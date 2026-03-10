#include <faiss/IndexHNSW.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <omp.h>
#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
#include <map>
#include <string>

// Helper to read .fvecs / .ivecs files
template <typename T>
T* vecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb"); 
    if (!f) {
        std::cerr << "Could not open " << fname << std::endl;
        return nullptr;
    }
    int d;
    (void)fread(&d, sizeof(int), 1, f);
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
        (void)fread(header.data(), sizeof(int), 1, f);
        (void)fread(x + i * d, sizeof(T), d, f);
    }
    fclose(f);
    return x;
}

// --- Argument Parser Helper ---
std::string get_arg_str(int argc, char** argv, std::string key, std::string def) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == key) return std::string(argv[i+1]);
    }
    return def;
}

// In-memory parameter struct 
struct ConannCalibParams {
    float lambda_hat;
    float gamma;
    int c_reg;
    float min_dist;
    float max_dist;
    std::vector<float> sweep_schedule; 
    std::string term_method; 
    float max_reg_val;         
};

float calculate_fnr(const faiss::idx_t* preds, const int* gt_row, int k) {
    std::set<int> gt_set(gt_row, gt_row + k);
    int intersection_size = 0;

    for (int j = 0; j < k; ++j) {
        if (gt_set.count(static_cast<int>(preds[j])) > 0) {
            intersection_size++;
        }
    }
    return 1.0f -
            (static_cast<float>(intersection_size) / static_cast<float>(k));
}

// Set termination environment correctly 
void set_termination_env(faiss::IndexHNSWFlat& index, faiss::SearchParametersHNSW& hnsw_params, 
                         const std::string& term_method, float val, int k) {
    if (term_method == "ef_search") {
        index.hnsw.efSearch = (int)val;
        hnsw_params.efSearch = (int)val;
    } else if (term_method == "patience") {
        int queue_size = std::max(2000, k);
        index.hnsw.efSearch = queue_size;
        hnsw_params.efSearch = queue_size;
        setenv("HNSW_PATIENCE", std::to_string((int)val).c_str(), 1);
    } else if (term_method == "hardlimit") {
        int queue_size = std::max(3000, k);
        index.hnsw.efSearch = queue_size;
        hnsw_params.efSearch = queue_size;
        setenv("HNSW_HARDLIMIT_MAX_NODES", std::to_string((int)val).c_str(), 1);
    } else if (term_method == "radiuslimit") {
        int queue_size = std::max(2000, k);
        index.hnsw.efSearch = queue_size;
        hnsw_params.efSearch = queue_size;
        setenv("HNSW_RADIUSLIMIT_RADIUS", std::to_string(val).c_str(), 1);
    }
}

// === NEW: Added test_gt and d_gt to the signature ===
void search_hnsw_conann_batch(
        faiss::IndexHNSWFlat& index,
        size_t nq,
        const float* queries,
        int k,
        const ConannCalibParams& params,
        faiss::idx_t* final_labels,
        float* final_distances,
        std::vector<float>& param_used, 
        std::vector<faiss::HNSWSearchCache>& all_caches,
        const int* test_gt, 
        size_t d_gt) {
    
    param_used.resize(nq);

    std::vector<int> active_indices(nq);
    std::iota(active_indices.begin(), active_indices.end(), 0);

    std::vector<float> batch_queries(nq * index.d);
    std::vector<faiss::idx_t> batch_labels(nq * k);
    std::vector<float> batch_distances(nq * k);
    std::vector<int> next_active_indices;
    next_active_indices.reserve(nq);

    // === NEW: Cache to hold the previous step's results ===
    std::vector<faiss::idx_t> prev_final_labels(nq * k);
    std::vector<float> prev_final_distances(nq * k);

    std::ofstream remaining_ids_csv("remaining_active_indices.csv");

    for (size_t p = 0; p < params.sweep_schedule.size(); ++p) {
        float current_val = params.sweep_schedule[p];
        size_t num_active = active_indices.size();

        if (num_active == 0) break;

        std::cerr << "\nProcessing " << params.term_method << " = " << current_val << " for "
                  << num_active << " active queries..." << std::endl;
        
        if (p == params.sweep_schedule.size() - 1) {
            remaining_ids_csv << "query_id\n";
            for (int query_id : active_indices) remaining_ids_csv << query_id << "\n";
        }

        std::vector<faiss::HNSWSearchCache*> active_caches(num_active);

        for (size_t i = 0; i < num_active; ++i) {
            int orig_idx = active_indices[i];
            std::memcpy(&batch_queries[i * index.d], &queries[orig_idx * index.d], index.d * sizeof(float));
            active_caches[i] = &all_caches[orig_idx];
        }

        faiss::SearchParametersHNSW hnsw_params;
        set_termination_env(index, hnsw_params, params.term_method, current_val, k);

        index.search_resume(
                num_active,
                batch_queries.data(),
                k,
                batch_distances.data(),
                batch_labels.data(),
                active_caches,
                reinterpret_cast<faiss::SearchParameters*>(&hnsw_params));

        next_active_indices.clear();

        for (size_t i = 0; i < num_active; ++i) {
            int orig_idx = active_indices[i];
            float dist_k = batch_distances[i * k + (k - 1)];
            faiss::idx_t label_k = batch_labels[i * k + (k - 1)]; // Check the ID!

            float norm_score;
            // Sanitize unfound neighbors
            if (label_k == -1 || std::isinf(dist_k) || dist_k >= std::numeric_limits<float>::max()) {
                norm_score = 1.0f;
            } else {
                norm_score = std::min(dist_k / params.max_dist, 1.0f);
            }

            float raw_reg_score = (1.0f - norm_score) + params.gamma * std::max(0.0f, static_cast<float>(p) - static_cast<float>(params.c_reg));
            float reg_score = raw_reg_score / params.max_reg_val;

            // Paper's logic: if score > lamhat, terminate and return PREVIOUS
            if (reg_score > params.lambda_hat) {
                if (p == 0) {
                    // Edge case: Threshold met on very first try, no previous state exists
                    std::memcpy(&final_labels[orig_idx * k], &batch_labels[i * k], k * sizeof(faiss::idx_t));
                    std::memcpy(&final_distances[orig_idx * k], &batch_distances[i * k], k * sizeof(float));
                    param_used[orig_idx] = current_val;
                } else {
                    // Normal case: We searched one too far, use the previous state!
                    std::memcpy(&final_labels[orig_idx * k], &prev_final_labels[orig_idx * k], k * sizeof(faiss::idx_t));
                    std::memcpy(&final_distances[orig_idx * k], &prev_final_distances[orig_idx * k], k * sizeof(float));
                    // Log the parameter that we actually used
                    param_used[orig_idx] = params.sweep_schedule[p - 1]; 
                }
            } 
            else if (p == params.sweep_schedule.size() - 1) {
                // Reached the absolute max limit without ever hitting the threshold, force terminate
                std::memcpy(&final_labels[orig_idx * k], &batch_labels[i * k], k * sizeof(faiss::idx_t));
                std::memcpy(&final_distances[orig_idx * k], &batch_distances[i * k], k * sizeof(float));
                param_used[orig_idx] = current_val;
            } 
            else {
                // Threshold not met yet. Save CURRENT state into PREVIOUS cache for the next loop
                std::memcpy(&prev_final_labels[orig_idx * k], &batch_labels[i * k], k * sizeof(faiss::idx_t));
                std::memcpy(&prev_final_distances[orig_idx * k], &batch_distances[i * k], k * sizeof(float));
                
                // Keep query active
                next_active_indices.push_back(orig_idx);
            }
        }
        
        // // === NEW: Print step statistics ===
        // if (test_gt) {
        //     float avg_rec_term = (count_terminated > 0) ? (sum_recall_terminated / count_terminated) : 0.0f;
        //     float avg_rec_cont = (count_continued > 0) ? (sum_recall_continued / count_continued) : 0.0f;
        //     float avg_pi_term = (count_terminated > 0) ? (sum_pi_terminated / count_terminated) : 0.0f;
        //     float avg_pi_cont = (count_continued > 0) ? (sum_pi_continued / count_continued) : 0.0f;
            
        //     std::cerr << "  -> Terminated: " << count_terminated << " queries (Avg Recall: " << avg_rec_term 
        //     << ", Avg pi_hat: " << avg_pi_term << ")\n";
        //     std::cerr << "  -> Continued:  " << count_continued << " queries (Avg Recall: " << avg_rec_cont 
        //     << ", Avg pi_hat: " << avg_pi_cont << ")\n";
        // }
        // // ==================================

        active_indices = next_active_indices;
    }
}

std::vector<float> generate_exponential_sweep(int start_val, int max_val, float growth_fraction = 0.5f) {
    std::vector<float> sweep;
    int current = start_val;
    
    while (current < max_val) {
        sweep.push_back(static_cast<float>(current));
        int step = static_cast<int>(current * growth_fraction);
        if (step < 1) step = 1;
        current += step;
    }
    sweep.push_back(static_cast<float>(max_val));
    return sweep;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <k> <alpha> [--term_method ef_search|patience|hardlimit|radiuslimit]" << std::endl;
        return 1;
    }

    int k = std::stoi(argv[1]);
    float target_alpha = std::stof(argv[2]);
    std::string term_method = get_arg_str(argc, argv, "--term_method", "ef_search");

    size_t d_base, nb, d_q_calib, nq_calib, d_gt_calib, n_gt_calib;
    size_t d_q_test, nq_test, d_gt_test, n_gt_test;

    // -------------------------------------------------------------
    // STEP 1: Load All Datasets
    // -------------------------------------------------------------
    std::cerr << "1. Loading Datasets..." << std::endl;
    float* xb = vecs_read<float>("../sift1M/sift_base.fvecs", &d_base, &nb);
    float* calib_queries = vecs_read<float>("../sift1M/sift_query_calib.fvecs", &d_q_calib, &nq_calib);
    int* calib_gt = vecs_read<int>("../sift1M/sift_groundtruth_calib.ivecs", &d_gt_calib, &n_gt_calib);
    float* test_queries = vecs_read<float>("../sift1M/sift_query.fvecs", &d_q_test, &nq_test);
    int* test_gt = vecs_read<int>("../sift1M/sift_groundtruth.ivecs", &d_gt_test, &n_gt_test);

    if (!xb || !calib_queries || !calib_gt || !test_queries || !test_gt) {
        std::cerr << "Failed to load one or more files. Check paths." << std::endl;
        return -1;
    }

    setenv("HNSW_TERMINATION_METHOD", term_method.c_str(), 1);

    // -------------------------------------------------------------
    // STEP 2: Build the HNSW Index
    // -------------------------------------------------------------
    int M = 32;
    faiss::IndexHNSWFlat index(d_base, M);
    index.hnsw.efConstruction = 300;

    std::vector<faiss::HNSWSearchCache> calib_caches;
    for (size_t i = 0; i < nq_calib; ++i) {
        calib_caches.emplace_back(2000); 
    }

    std::cerr << "2. Building HNSW index on " << nb << " vectors..." << std::endl;
    index.add(nb, xb);

    // -------------------------------------------------------------
    // STEP 3: Calibration Phase
    // -------------------------------------------------------------
    std::map<std::string, std::vector<float>> sweep_configs = {
        {"ef_search",   {10, 16, 24, 32, 64, 128, 256, 512, 1024}},
        {"patience",    {2, 5, 10, 20, 35, 50, 100, 200}},
        {"radiuslimit", {3225.0f, 5500.0f, 9500.0f, 16000.0f, 27000.0f, 46000.0f, 78000.0f, 132256.0f}}
    };

    int max_hardlimit = 140000;
    sweep_configs["hardlimit"] = generate_exponential_sweep(2, max_hardlimit, 0.2f);

    std::cerr << "Calibrating with " << sweep_configs["hardlimit"].size() << " hardlimit values." << std::endl;

    if (sweep_configs.find(term_method) == sweep_configs.end()) {
        std::cerr << "Invalid --term_method provided." << std::endl;
        return 1;
    }
    
    std::vector<float> sweep_schedule = sweep_configs[term_method];
    size_t num_steps = sweep_schedule.size();

    std::vector<std::vector<float>> nonconf_matrix(nq_calib, std::vector<float>(num_steps, 0.0f));
    std::vector<std::vector<std::vector<faiss::idx_t>>> all_preds_list(
            nq_calib, std::vector<std::vector<faiss::idx_t>>(num_steps, std::vector<faiss::idx_t>(k)));

    float global_min_dist = std::numeric_limits<float>::max();
    float global_max_dist = 0.0f;

    std::vector<float> distances(nq_calib * k);
    std::vector<faiss::idx_t> labels(nq_calib * k);
    faiss::SearchParametersHNSW hnsw_params;

    std::cerr << "3. Running Calibration (Building Non-Conformity Matrix for " << term_method << ")..." << std::endl;
    for (size_t col = 0; col < num_steps; ++col) {
        float val = sweep_schedule[col];
        
        set_termination_env(index, hnsw_params, term_method, val, k);

        std::vector<faiss::HNSWSearchCache*> batch_calib_caches(nq_calib);
        for (size_t i = 0; i < nq_calib; i++) {
            batch_calib_caches[i] = &calib_caches[i];
        }

        index.search_resume(
                nq_calib,
                calib_queries,
                k,
                distances.data(),
                labels.data(),
                batch_calib_caches,
                reinterpret_cast<faiss::SearchParameters*>(&hnsw_params));

        for (size_t row = 0; row < nq_calib; ++row) {
            float dist_k = distances[row * k + (k - 1)];
            nonconf_matrix[row][col] = dist_k;

            if (dist_k < std::numeric_limits<float>::max() && labels[row * k + (k-1)] != -1) {
                if (dist_k < global_min_dist) global_min_dist = dist_k;
                if (dist_k > global_max_dist) global_max_dist = dist_k;
            }

            for (int j = 0; j < k; ++j) {
                all_preds_list[row][col][j] = labels[row * k + j];
            }
        }
    }

    // 1. Calculate max_reg_val exactly as the paper does
    float gamma = 0.01f;
    int c_reg = 2;
    // Note: The paper uses `nlist` (total clusters). For us, it's `num_steps`.
    float max_reg_val = (1.0f + gamma * std::max(0.0f, static_cast<float>(num_steps - c_reg))) + 10.0f;

    std::vector<std::vector<float>> reg_nonconf_matrix(nq_calib, std::vector<float>(num_steps, 0.0f));


    // 2. Build the regularized matrix using the inverted conformity logic
    for (size_t row = 0; row < nq_calib; ++row) {
        for (size_t col = 0; col < num_steps; ++col) {
            float dist_k = nonconf_matrix[row][col];
            faiss::idx_t label_k = all_preds_list[row][col][k - 1]; // Check the ID!
            
            float norm_score;
            // If ID is -1 or distance is infinity, the neighbor was not found
            if (label_k == -1 || std::isinf(dist_k) || dist_k >= std::numeric_limits<float>::max()) {
                norm_score = 1.0f; // Maximum uncertainty
            } else {
                norm_score = std::min(dist_k / global_max_dist, 1.0f);
            }
            
            float raw_reg_score = (1.0f - norm_score) + gamma * std::max(0.0f, static_cast<float>(col) - static_cast<float>(c_reg));
            reg_nonconf_matrix[row][col] = raw_reg_score / max_reg_val;
            // if (col == 0) {
            //     std::cerr << "DEBUG: Row " << row << ", Col " << col 
            //               << " -> dist_k: " << dist_k 
            //               << ", norm_score: " << norm_score 
            //               << ", raw_reg_score: " << raw_reg_score 
            //               << ", reg_score: " << reg_nonconf_matrix[row][col] 
            //               << std::endl;
            // }
        }
    }


    // Print average regularized non-conformity scores per sweep parameter
    // std::cerr << "Average Regularized Non-Conformity Scores per Sweep Parameter:" << std::endl;
    // for (size_t col = 0; col < num_steps; ++col) {
    //     float col_sum = 0.0f;
    //     for (size_t row = 0; row < nq_calib; ++row) {
    //         col_sum += reg_nonconf_matrix[row][col];
    //     }
    //     float col_avg = col_sum / nq_calib;
    //     std::cerr << "  [" << sweep_schedule[col] << "]: " << col_avg << std::endl;
    // }

    // 3. CRC Optimization using the "One Step Back" logic
    float B = 1.0f;
    float best_lambda = -1.0f;

    // --- DEBUG: Check maximum possible performance ---
    // float max_effort_fnr_sum = 0.0f;
    // for (size_t row = 0; row < nq_calib; ++row) {
    //     const int* gt_row = &calib_gt[row * d_gt_calib];
    //     max_effort_fnr_sum += calculate_fnr(all_preds_list[row][num_steps - 1].data(), gt_row, k);
    // }
    // float best_possible_risk = (nq_calib * (max_effort_fnr_sum / nq_calib) + B) / (nq_calib + 1);
    // std::cerr << "DEBUG: The lowest possible risk this index can achieve is: " 
    //           << best_possible_risk << " (Target: " << target_alpha << ")\n";
    // -------------------------------------------------

    for (float lambda_cand = 0.0f; lambda_cand <= 1.0f; lambda_cand += 0.001f) {
        float empirical_risk_sum = 0.0f;

        for (size_t row = 0; row < nq_calib; ++row) {
            int selected_col = num_steps - 1; // Default to max effort if threshold never reached

            for (size_t col = 0; col < num_steps; ++col) {
                // If the conformity score gets HIGH enough to cross the threshold
                if (reg_nonconf_matrix[row][col] > lambda_cand) {
                    // We searched one step too far, fall back to the previous step
                    selected_col = (col == 0) ? 0 : col - 1;
                    break;
                }
            }

            const int* gt_row = &calib_gt[row * d_gt_calib];
            const auto& preds = all_preds_list[row][selected_col];
            empirical_risk_sum += calculate_fnr(preds.data(), gt_row, k);
        }

        float expected_risk = (nq_calib * (empirical_risk_sum / nq_calib) + B) / (nq_calib + 1);

        if (expected_risk <= target_alpha) {
            best_lambda = lambda_cand;
            // Note: Since R(\lambda) decreases as \lambda increases in this setup, 
            // we want the SMALLEST lambda that satisfies the risk.
            break; 
        }
    }

    ConannCalibParams params = {
            best_lambda,
            gamma,
            c_reg,
            global_min_dist,
            global_max_dist,
            sweep_schedule, 
            term_method,
            max_reg_val
    };

    std::cerr << "Optimal Lambda (lamhat) found: " << best_lambda << std::endl;

    // -------------------------------------------------------------
    // STEP 4: Online Search Phase
    // -------------------------------------------------------------
    std::cerr << "4. Running Adaptive Online Search..." << std::endl;
    std::vector<faiss::idx_t> all_labels(nq_test * k);
    std::vector<float> all_distances(nq_test * k);
    std::vector<float> tracking_param_used; 

    std::vector<faiss::HNSWSearchCache> test_caches;
    for (size_t i = 0; i < nq_test; ++i) {
        test_caches.emplace_back(2000);
    }

    double t_start = omp_get_wtime();
    // === NEW: Passed test_gt and d_gt_test into the online search ===
    search_hnsw_conann_batch(
            index,
            nq_test,
            test_queries,
            k,
            params,
            all_labels.data(),
            all_distances.data(),
            tracking_param_used,
            test_caches,
            test_gt,       
            d_gt_test);    
    double t_end = omp_get_wtime();
    double time_taken = t_end - t_start;

    std::cerr << "\nSearch completed in " << time_taken << " seconds." << std::endl;

    // -------------------------------------------------------------
    // STEP 5: Evaluate FNR & Efficiency
    // -------------------------------------------------------------
    std::cerr << "5. Evaluating Accuracy (FNR) & Efficiency..." << std::endl;
    float total_fnr = 0.0f;
    double total_param_effort = 0;

    for (size_t i = 0; i < nq_test; ++i) {
        total_fnr += calculate_fnr(&all_labels[i * k], &test_gt[i * d_gt_test], k);
        total_param_effort += tracking_param_used[i];
    }

    float empirical_fnr = total_fnr / nq_test;
    float avg_param = total_param_effort / nq_test;

    std::cerr << "------------------------------------------------" << std::endl;
    std::cerr << "Empirical Expected FNR: " << (empirical_fnr * 100.0f) << "%" << std::endl;
    std::cerr << "Average " << term_method << " value used: " << avg_param << std::endl;
    std::cerr << "------------------------------------------------" << std::endl;

    std::cout << target_alpha << "," << best_lambda << "," << time_taken << ","
              << avg_param << "," << empirical_fnr << std::endl;

    delete[] xb;
    delete[] calib_queries;
    delete[] calib_gt;
    delete[] test_queries;
    delete[] test_gt;

    return 0;
}