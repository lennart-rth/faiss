#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <map>
#include <unordered_set>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>

// --- Helper: Read .fvecs ---
float* read_fvecs(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Err: cannot open %s\n", fname); exit(1); }
    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) { fclose(f); return nullptr; }
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

// --- Helper: Read .ivecs (Ground Truth) ---
int* read_ivecs(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", fname); exit(1); }
    int d;
    if (fread(&d, sizeof(int), 1, f) != 1) { fclose(f); return nullptr; }
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

// --- Helper: Calculate Recall@k ---
float calculate_recall(const std::vector<faiss::idx_t>& search_results, 
                       const int* ground_truth, 
                       size_t num_queries, int k, size_t gt_dim) {
    int total_hits = 0;
    for (size_t i = 0; i < num_queries; ++i) {
        std::unordered_set<int> gt_set;
        // Insert the top-k true neighbors for this query
        for (int j = 0; j < k; ++j) {
            gt_set.insert(ground_truth[i * gt_dim + j]);
        }
        // Check how many of the search results are in the true top-k
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(search_results[i * k + j])) {
                total_hits++;
            }
        }
    }
    return (float)total_hits / (num_queries * k);
}

// --- Argument Parser Helpers ---
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
    // 1. Parse Args (Removed n_queries)
    int M = get_arg_int(argc, argv, "--M", 32);
    int efConstruction = get_arg_int(argc, argv, "--efConstruction", 40);
    int k = get_arg_int(argc, argv, "--k", 10);
    
    std::string data_dir = get_arg_str(argc, argv, "--data", "../../sift1M");
    std::string csv_file = get_arg_str(argc, argv, "--csv_out", "frontier_results.csv");

    // 2. Define Hardcoded Sweep Configurations
    // Note: You must adjust the 'radiuslimit' values based on the output of your calibration script!
    std::map<std::string, std::vector<float>> sweep_configs = {
        {"ef_search",   {2, 4, 8 ,16, 32, 64, 128, 256}},
        {"patience",    {2, 4, 8, 16, 32, 64, 128, 256}},
        {"hardlimit",   {2, 4, 8, 16, 32, 64, 128, 256}},
        {"radiuslimit", {3225.0f, 5500.0f, 9500.0f, 16000.0f, 27000.0f, 46000.0f, 78000.0f, 132256.0f}}
    };

    // 3. Load Data
    size_t d, nb, nq, d_ignore, gt_dim, n_gt;
    std::string base_path = data_dir + "/sift_base.fvecs";
    std::string query_path = data_dir + "/sift_query.fvecs";
    std::string gt_path = data_dir + "/sift_groundtruth.ivecs";

    std::cout << "[C++] Loading datasets..." << std::endl;
    float* xb = read_fvecs(base_path.c_str(), &d, &nb);
    float* xq = read_fvecs(query_path.c_str(), &d_ignore, &nq);
    int* gt = read_ivecs(gt_path.c_str(), &gt_dim, &n_gt);

    std::cout << "[C++] Loaded " << nq << " queries." << std::endl;

    // 4. Build Index (ONCE)
    std::cout << "[C++] Building Index (M=" << M << ", efC=" << efConstruction << ")..." << std::endl;
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.verbose = false; 
    index.add(nb, xb);
    std::cout << "[C++] Index built." << std::endl;

    // Open CSV File for appending
    std::ofstream out_csv(csv_file, std::ios::app);
    // Write header if file is empty
    out_csv.seekp(0, std::ios::end);
    if (out_csv.tellp() == 0) {
        out_csv << "method,sweep_value,latency_ms,recall@" << k << "\n";
    }

    // 5. Sweep Over All Methods and Values
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    for (const auto& [term_method, sweep_values] : sweep_configs) {
        std::cout << "\n[C++] Starting sweep for method: " << term_method << std::endl;
        setenv("HNSW_TERMINATION_METHOD", term_method.c_str(), 1);

        for (float val : sweep_values) {
            // Configure the specific limits
            if (term_method == "ef_search") {
                index.hnsw.efSearch = (int)val;
            } 
            else if (term_method == "patience") {
                index.hnsw.efSearch = std::max(2000, k); // Infinite queue
                setenv("HNSW_PATIENCE", std::to_string((int)val).c_str(), 1);
            } 
            else if (term_method == "hardlimit") {
                index.hnsw.efSearch = std::max(3000, k); // Infinite queue
                setenv("HNSW_HARDLIMIT_MAX_NODES", std::to_string((int)val).c_str(), 1);
            } 
            else if (term_method == "radiuslimit") {
                index.hnsw.efSearch = std::max(2000, k); // Infinite queue
                setenv("HNSW_RADIUSLIMIT_RADIUS", std::to_string(val).c_str(), 1);
            }

            // Execute Search & Measure Time
            auto start = std::chrono::high_resolution_clock::now();
            index.search(nq, xq, k, D.data(), I.data());
            auto end = std::chrono::high_resolution_clock::now();

            double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double latency_per_query_ms = duration_ms / nq;

            // Calculate Recall
            float recall = calculate_recall(I, gt, nq, k, gt_dim);

            // Write to CSV and console
            out_csv << term_method << "," << val << "," << latency_per_query_ms << "," << recall << "\n";
            std::cout << "  -> Val: " << val << " | Latency: " << latency_per_query_ms 
                      << " ms/query | Recall: " << recall << std::endl;
        }
    }

    out_csv.close();
    delete[] xb;
    delete[] xq;
    delete[] gt;
    std::cout << "\n[C++] Sweep complete. Results saved to " << csv_file << std::endl;
    return 0;
}