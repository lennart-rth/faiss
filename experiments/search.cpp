#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <sys/stat.h>
#include <cstring>
#include <omp.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/AuxIndexStructures.h>

template <typename T>
T* vecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        std::cerr << "Could not open " << fname << std::endl;
        return nullptr;
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    fseek(f, 0, SEEK_SET);
    
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    size_t n = sz / ((d + 1) * 4);
    *d_out = d;
    *n_out = n;
    
    T* x = new T[n * d];
    std::vector<int> header(1);
    for (size_t i = 0; i < n; i++) {
        fread(header.data(), sizeof(int), 1, f);
        fread(x + i * d, sizeof(T), d, f);
    }
    fclose(f);
    return x;
}

struct ConannCalibParams {
    float lambda_hat;
    float gamma;
    int c_reg;
    float min_dist;
    float max_dist;
    std::vector<int> ef_schedule;
};

ConannCalibParams load_conann_params(const std::string& filename) {
    ConannCalibParams params;
    std::ifstream in(filename);
    
    if (!in.is_open()) {
        std::cerr << "Failed to open " << filename << " for reading. Run calibration first!" << std::endl;
        exit(-1);
    }
    
    in >> params.lambda_hat;
    in >> params.gamma;
    in >> params.c_reg;
    in >> params.min_dist;
    in >> params.max_dist;
    
    size_t schedule_size;
    in >> schedule_size;
    params.ef_schedule.resize(schedule_size);
    for (size_t i = 0; i < schedule_size; ++i) {
        in >> params.ef_schedule[i];
    }
    
    in.close();
    std::cout << "Loaded ConANN parameters from " << filename << std::endl;
    std::cout << "  - Lambda Hat: " << params.lambda_hat << "\n"
              << "  - Gamma: " << params.gamma << "\n"
              << "  - c_reg: " << params.c_reg << "\n"
              << "  - efSearch Steps: " << params.ef_schedule.size() << std::endl;
              
    return params;
}

void search_hnsw_conann_batch(
    faiss::IndexHNSWFlat& index,
    size_t nq,
    const float* queries,
    int k,
    const ConannCalibParams& params,
    faiss::idx_t* final_labels,
    float* final_distances,
    std::vector<int>& ef_used) 
{
    ef_used.resize(nq);

    // 1. Initialize the active set (all queries start as active)
    std::vector<int> active_indices(nq);
    std::iota(active_indices.begin(), active_indices.end(), 0);

    // Pre-allocate buffers to avoid allocations inside the loop.
    // They are sized to the maximum possible batch size (nq).
    std::vector<float> batch_queries(nq * index.d);
    std::vector<faiss::idx_t> batch_labels(nq * k);
    std::vector<float> batch_distances(nq * k);
    std::vector<int> next_active_indices;
    
    // Reserve space to prevent reallocation during push_back
    next_active_indices.reserve(nq);

    // 2. Iterate through the ef_search schedule
    for (size_t p = 0; p < params.ef_schedule.size(); ++p) {
        int current_ef = params.ef_schedule[p];
        size_t num_active = active_indices.size();
        
        // If all queries have satisfied the criteria, we are done!
        if (num_active == 0) break;

        // 3. Gather active queries into a contiguous buffer for FAISS
        // (Copying memory is trivially fast compared to graph traversal)
        for (size_t i = 0; i < num_active; ++i) {
            int orig_idx = active_indices[i];
            std::memcpy(&batch_queries[i * index.d], 
                        &queries[orig_idx * index.d], 
                        index.d * sizeof(float));
        }

        // 4. Perform the STANDARD FAISS BATCH SEARCH
        // FAISS will handle OpenMP parallelization and SIMD internally here
        faiss::SearchParametersHNSW hnsw_params;
        hnsw_params.efSearch = current_ef;

        index.search(num_active, batch_queries.data(), k, 
                     batch_distances.data(), batch_labels.data(), 
                     reinterpret_cast<faiss::SearchParameters*>(&hnsw_params));

        next_active_indices.clear();

        // 5. Evaluate the ConANN stopping criteria for this batch
        for (size_t i = 0; i < num_active; ++i) {
            int orig_idx = active_indices[i];
            float dist_k = batch_distances[i * k + (k - 1)];
            
            float norm_dist = (dist_k - params.min_dist) / (params.max_dist - params.min_dist);
            norm_dist = std::max(0.0f, std::min(1.0f, norm_dist));

            float base_score = norm_dist;
            float penalty = params.gamma * std::max(0, static_cast<int>(p) - params.c_reg);
            float pi_hat = base_score + penalty;

            // Check if it passes OR if we are on the very last fallback schedule step
            if (pi_hat <= params.lambda_hat || p == params.ef_schedule.size() - 1) {
                // Passed: Scatter results to the correct original index in the final arrays
                std::memcpy(&final_labels[orig_idx * k], &batch_labels[i * k], k * sizeof(faiss::idx_t));
                std::memcpy(&final_distances[orig_idx * k], &batch_distances[i * k], k * sizeof(float));
                ef_used[orig_idx] = current_ef;
            } else {
                // Failed: Add to the worklist for the next ef_search iteration
                next_active_indices.push_back(orig_idx);
            }
        }

        // 6. Update the active set for the next iteration
        active_indices = next_active_indices;
    }
}

// Fixed signature to accept raw pointers directly
float calculate_fnr(const faiss::idx_t* preds, const int* gt_row, int k) {
    std::set<int> gt_set(gt_row, gt_row + k);
    int intersection_size = 0;
    
    for (int j = 0; j < k; ++j) {
        if (gt_set.count(static_cast<int>(preds[j])) > 0) {
            intersection_size++;
        }
    }
    return 1.0f - (static_cast<float>(intersection_size) / static_cast<float>(k));
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <k> <alpha>" << std::endl;
        return 1; // Exit with an error code
    }

    int k = std::stoi(argv[1]);
    float target_alpha = std::stof(argv[2]);

    size_t d_base, nb, d_q, nq, d_gt, n_gt;
    
    std::cout << "1. Loading Datasets..." << std::endl;
    float* xb = vecs_read<float>("../sift1M/sift_base.fvecs", &d_base, &nb);
    float* xq = vecs_read<float>("../sift1M/sift_query.fvecs", &d_q, &nq);
    int* gt = vecs_read<int>("../sift1M/sift_groundtruth.ivecs", &d_gt, &n_gt);
    
    if (!xb || !xq || !gt) {
        std::cerr << "Failed to load files." << std::endl;
        return -1;
    }

    std::cout << "2. Building HNSW Index..." << std::endl;
    faiss::IndexHNSWFlat index(d_base, 32);
    index.add(nb, xb);
    
    std::cout << "3. Setting up ConANN parameters..." << std::endl;

    ConannCalibParams params = load_conann_params("conann_calibrated_params.txt");
    
    // FIXED: Removed the redundant loop that populated params.ef_schedule

    std::vector<faiss::idx_t> all_labels(nq * k);
    std::vector<float> all_distances(nq * k);
    std::vector<int> tracking_ef_used;

    std::cout << "4. Running Adaptive Online Search..." << std::endl;
    double t_start = omp_get_wtime();
    
    search_hnsw_conann_batch(
        index, 
        nq, 
        xq, 
        k, 
        params, 
        all_labels.data(), 
        all_distances.data(), 
        tracking_ef_used
    );
    
    double t_end = omp_get_wtime();
    std::cout << "Search completed in " << (t_end - t_start) << " seconds." << std::endl;

    std::cout << "5. Evaluating Accuracy (FNR) & Efficiency..." << std::endl;
    float total_fnr = 0.0f;
    long long total_ef = 0;
    
    for (size_t i = 0; i < nq; ++i) {
        // FIXED: Use d_gt instead of k to step through the ground truth rows correctly
        total_fnr += calculate_fnr(&all_labels[i * k], &gt[i * d_gt], k);
        total_ef += tracking_ef_used[i];
    }
    
    float empirical_fnr = total_fnr / nq;
    float avg_ef = static_cast<float>(total_ef) / nq;
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Empirical Expected FNR: " << (empirical_fnr * 100.0f) << "%" << std::endl;
    std::cout << "Average efSearch used:  " << avg_ef << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    delete[] xb; delete[] xq; delete[] gt;
    return 0;
}