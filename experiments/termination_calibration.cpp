#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>

float* read_fvecs(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb"); // "rb" ensures safe binary reading
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", fname); exit(1); }
    
    int d;
    // CORRECTED: size is sizeof(int), count is 1. Returns 1 on success.
    if (fread(&d, sizeof(int), 1, f) != 1) { 
        fprintf(stderr, "Error: failed to read dimension from %s\n", fname);
        fclose(f); return nullptr; 
    }
    
    *d_out = d;
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    *n_out = sz / (sizeof(int) + d * sizeof(float));
    float* x = new float[*n_out * d];
    
    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        fread(&d_check, sizeof(int), 1, f);
        fread(x + i * d, sizeof(float), d, f);
    }
    fclose(f);
    return x;
}

// --- Helper: Read .ivecs ---
int* read_ivecs(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb"); // "rb" ensures safe binary reading
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", fname); exit(1); }
    
    int d;
    // CORRECTED: size is sizeof(int), count is 1. Returns 1 on success.
    if (fread(&d, sizeof(int), 1, f) != 1) { 
        fprintf(stderr, "Error: failed to read dimension from %s\n", fname);
        fclose(f); return nullptr; 
    }
    
    *d_out = d;
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    *n_out = sz / (sizeof(int) + d * sizeof(int));
    int* x = new int[*n_out * d];
    
    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        fread(&d_check, sizeof(int), 1, f);
        fread(x + i * d, sizeof(int), d, f);
    }
    fclose(f);
    return x;
}

// --- Helper: Compute Squared L2 Distance ---
float compute_L2_sqr(const float* x, const float* y, size_t d) {
    float res = 0;
    for (size_t i = 0; i < d; i++) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

// --- Argument Parser ---
std::string get_arg_str(int argc, char** argv, std::string key, std::string def) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == key) return std::string(argv[i+1]);
    }
    return def;
}
int get_arg_int(int argc, char** argv, std::string key, int def) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == key) return std::atoi(argv[i+1]);
    }
    return def;
}

int main(int argc, char** argv) {
    // 1. Parse Arguments
    int k = get_arg_int(argc, argv, "--k", 10);
    int M = get_arg_int(argc, argv, "--M", 32);
    int efConstruction = get_arg_int(argc, argv, "--efConstruction", 40);
    int efSearch = get_arg_int(argc, argv, "--efSearch", 128); // Standard search effort for calibration

    std::string data_dir = get_arg_str(argc, argv, "--data", "../../sift1M");

    std::string base_path = data_dir + "/sift_base.fvecs";
    std::string query_calib_path = data_dir + "/sift_query_calib.fvecs";
    std::string gt_calib_path = data_dir + "/sift_groundtruth_calib.ivecs";

    // 2. Load Datasets
    std::cout << "[Calibration] Loading datasets..." << std::endl;
    size_t d_base, n_base;
    float* xb = read_fvecs(base_path.c_str(), &d_base, &n_base);

    size_t d_q_calib, nq_calib;
    float* calib_queries = read_fvecs(query_calib_path.c_str(), &d_q_calib, &nq_calib);

    size_t d_gt_calib, n_gt_calib;
    int* calib_gt = read_ivecs(gt_calib_path.c_str(), &d_gt_calib, &n_gt_calib);

    if (d_base != d_q_calib) {
        std::cerr << "Dimension mismatch! Base is " << d_base 
                  << " but Queries are " << d_q_calib << std::endl;
        return 1;
    }
    if (k > d_gt_calib) {
        std::cerr << "Requested k=" << k << " is larger than groundtruth size " << d_gt_calib << std::endl;
        return 1;
    }

    // =========================================================================
    // CALIBRATION 1: Optimal Radius Limit
    // Goal: Find the max distance to the k-th neighbor across all calib queries
    // =========================================================================
    std::cout << "[Calibration] Computing optimal radius cutoff for k=" << k << "..." << std::endl;
    
    float optimal_radius = 0.0f;
    float min_radius = std::numeric_limits<float>::max();

    for (size_t i = 0; i < nq_calib; i++) {
        // Find the groundtruth ID for the k-th closest neighbor (0-indexed, so k-1)
        int kth_neighbor_id = calib_gt[i * d_gt_calib + (k - 1)];
        
        // Compute L2 squared distance between query and its k-th ground truth neighbor
        float dist = compute_L2_sqr(
            calib_queries + i * d_base, 
            xb + kth_neighbor_id * d_base, 
            d_base
        );

        optimal_radius = std::max(optimal_radius, dist);
        min_radius = std::min(min_radius, dist);
    }
    
    // Add a tiny epsilon multiplier (e.g., 1.01) to account for float inaccuracies
    optimal_radius = optimal_radius * 1.01f;


    // =========================================================================
    // CALIBRATION 2: Optimal Hard Limit (Node Visit Count)
    // Goal: Run HNSW and track the max hops needed for standard convergence
    // =========================================================================
    std::cout << "[Calibration] Building HNSW Index for hop counting..." << std::endl;
    
    // Ensure we are using standard ef_search termination logic
    setenv("HNSW_TERMINATION_METHOD", "ef_search", 1);
    
    faiss::IndexHNSWFlat index(d_base, M);
    index.hnsw.efConstruction = efConstruction;
    index.hnsw.efSearch = efSearch;
    index.add(n_base, xb);

    std::cout << "[Calibration] Tracing node visits per query..." << std::endl;
    
    std::vector<faiss::idx_t> I(k);
    std::vector<float> D(k);
    
    size_t optimal_hardlimit_nodes = 0;
    size_t total_hops = 0;

    for (size_t i = 0; i < nq_calib; i++) {
        // Reset FAISS global HNSW stats before running the query
        faiss::hnsw_stats.reset();

        // Search single query
        index.search(1, calib_queries + i * d_base, k, D.data(), I.data());

        // Read how many nodes were visited during this search
        size_t hops = faiss::hnsw_stats.nhops;
        
        optimal_hardlimit_nodes = std::max(optimal_hardlimit_nodes, hops);
        total_hops += hops;
    }

    size_t avg_hops = total_hops / nq_calib;

    // =========================================================================
    // OUTPUT RESULTS
    // =========================================================================
    std::cout << "\n=========================================\n";
    std::cout << "          CALIBRATION RESULTS            \n";
    std::cout << "=========================================\n";
    std::cout << "Dataset: " << data_dir << "\n";
    std::cout << "Target k: " << k << "\n";
    std::cout << "Calibration Queries Processed: " << nq_calib << "\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "[Radius Limit]\n";
    std::cout << "  -> Min distance to k-th neighbor: " << min_radius << "\n";
    std::cout << "  -> MAX DISTANCE (Optimal Radius): " << optimal_radius << "\n";
    std::cout << "  (Use: --term_method radiuslimit --sweep_values " << optimal_radius << ")\n";
    std::cout << "-----------------------------------------\n";
    std::cout << "[Hard Limit / Node Visits]\n";
    std::cout << "  -> Average node visits: " << avg_hops << "\n";
    std::cout << "  -> MAX VISITS (Optimal Limit): " << optimal_hardlimit_nodes << "\n";
    std::cout << "  (Use: --term_method hardlimit --sweep_values " << optimal_hardlimit_nodes << ")\n";
    std::cout << "=========================================\n";

    // Cleanup
    delete[] xb;
    delete[] calib_queries;
    delete[] calib_gt;

    return 0;
}