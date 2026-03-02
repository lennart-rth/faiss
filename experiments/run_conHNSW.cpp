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

// Helper to read .fvecs / .ivecs files
template <typename T>
T* vecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        std::cerr << "Could not open " << fname << std::endl;
        return nullptr;
    }
    int d;
    (void)fread(&d, 1, sizeof(int), f);
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

// In-memory parameter struct
struct ConannCalibParams {
    float lambda_hat;
    float gamma;
    int c_reg;
    float min_dist;
    float max_dist;
    std::vector<int> ef_schedule;
};

// Fixed signature to accept raw pointers directly
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

void search_hnsw_conann_batch(
        faiss::IndexHNSWFlat& index,
        size_t nq,
        const float* queries,
        int k,
        const ConannCalibParams& params,
        faiss::idx_t* final_labels,
        float* final_distances,
        std::vector<int>& ef_used,
        std::vector<faiss::HNSWSearchCache>& all_caches) {
    ef_used.resize(nq);

    std::vector<int> active_indices(nq);
    std::iota(active_indices.begin(), active_indices.end(), 0);

    std::vector<float> batch_queries(nq * index.d);
    std::vector<faiss::idx_t> batch_labels(nq * k);
    std::vector<float> batch_distances(nq * k);
    std::vector<int> next_active_indices;
    next_active_indices.reserve(nq);

    std::ofstream remaining_ids_csv("remaining_active_indices.csv");
    if (!remaining_ids_csv.is_open()) {
        std::cerr << "Failed to open remaining_active_indices.csv for writing." << std::endl;
    } 

    for (size_t p = 0; p < params.ef_schedule.size(); ++p) {
        int current_ef = params.ef_schedule[p];
        size_t num_active = active_indices.size();

        if (num_active == 0)
            break;

        std::cerr << "Processing efSearch = " << current_ef << " for "
                  << num_active << " active queries..." << std::endl;
        
        if (p == params.ef_schedule.size() - 1) {
            remaining_ids_csv << "query_id\n";
            for (int query_id : active_indices) {
                remaining_ids_csv << query_id << "\n";
            }
        }

        std::vector<faiss::HNSWSearchCache*> active_caches(num_active);

        for (size_t i = 0; i < num_active; ++i) {
            int orig_idx = active_indices[i];
            std::memcpy(
                    &batch_queries[i * index.d],
                    &queries[orig_idx * index.d],
                    index.d * sizeof(float));

            active_caches[i] = &all_caches[orig_idx];
        }

        faiss::SearchParametersHNSW hnsw_params;
        hnsw_params.efSearch = current_ef;
        index.hnsw.efSearch = current_ef;

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

            float norm_dist = (dist_k - params.min_dist) /
                    (params.max_dist - params.min_dist);
            norm_dist = std::max(0.0f, std::min(1.0f, norm_dist));

            float base_score = norm_dist;
            float penalty = params.gamma *
                    std::max(0, static_cast<int>(p) - params.c_reg);
            float pi_hat = base_score + penalty;

            if (pi_hat <= params.lambda_hat ||
                p == params.ef_schedule.size() - 1) {
                std::memcpy(
                        &final_labels[orig_idx * k],
                        &batch_labels[i * k],
                        k * sizeof(faiss::idx_t));
                std::memcpy(
                        &final_distances[orig_idx * k],
                        &batch_distances[i * k],
                        k * sizeof(float));
                ef_used[orig_idx] = current_ef;
            } else {
                next_active_indices.push_back(orig_idx);
            }
        }
        active_indices = next_active_indices;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <k> <alpha>" << std::endl;
        return 1;
    }

    int k = std::stoi(argv[1]);
    float target_alpha = std::stof(argv[2]);

    size_t d_base, nb, d_q_calib, nq_calib, d_gt_calib, n_gt_calib;
    size_t d_q_test, nq_test, d_gt_test, n_gt_test;

    // -------------------------------------------------------------
    // STEP 1: Load All Datasets
    // -------------------------------------------------------------
    std::cerr << "1. Loading Datasets..." << std::endl;
    float* xb = vecs_read<float>("../sift1M/sift_base.fvecs", &d_base, &nb);

    float* calib_queries = vecs_read<float>(
            "../sift1M/sift_query_calib.fvecs", &d_q_calib, &nq_calib);
    int* calib_gt = vecs_read<int>(
            "../sift1M/sift_groundtruth_calib.ivecs", &d_gt_calib, &n_gt_calib);

    float* test_queries =
            vecs_read<float>("../sift1M/sift_query.fvecs", &d_q_test, &nq_test);
    int* test_gt = vecs_read<int>(
            "../sift1M/sift_groundtruth.ivecs", &d_gt_test, &n_gt_test);

    if (!xb || !calib_queries || !calib_gt || !test_queries || !test_gt) {
        std::cerr << "Failed to load one or more files. Check paths."
                  << std::endl;
        return -1;
    }

    // -------------------------------------------------------------
    // STEP 2: Build the HNSW Index
    // -------------------------------------------------------------
    int M = 32;
    faiss::IndexHNSWFlat index(d_base, M);

    std::vector<faiss::HNSWSearchCache> calib_caches;
    for (size_t i = 0; i < nq_calib; ++i) {
        calib_caches.emplace_back(1200);
    }

    std::cerr << "2. Building HNSW index on " << nb << " vectors..."
              << std::endl;
    index.add(nb, xb);

    // -------------------------------------------------------------
    // STEP 3: Calibration Phase
    // -------------------------------------------------------------
    std::vector<int> efSearch_values;
    for (int ef = 1; ef <= 1024; ef += 20) {
        efSearch_values.push_back(ef);
    }
    size_t num_ef = efSearch_values.size();

    std::vector<std::vector<float>> nonconf_matrix(
            nq_calib, std::vector<float>(num_ef, 0.0f));
    std::vector<std::vector<std::vector<faiss::idx_t>>> all_preds_list(
            nq_calib,
            std::vector<std::vector<faiss::idx_t>>(
                    num_ef, std::vector<faiss::idx_t>(k)));

    float global_min_dist = std::numeric_limits<float>::max();
    float global_max_dist = 0.0f;

    std::vector<float> distances(nq_calib * k);
    std::vector<faiss::idx_t> labels(nq_calib * k);

    std::cerr << "3. Running Calibration (Building Non-Conformity Matrix)..."
              << std::endl;
    for (size_t col = 0; col < num_ef; ++col) {
        int ef = efSearch_values[col];
        index.hnsw.efSearch = ef;

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
                batch_calib_caches);

        for (size_t row = 0; row < nq_calib; ++row) {
            float dist_k = distances[row * k + (k - 1)];
            nonconf_matrix[row][col] = dist_k;

            if (dist_k < std::numeric_limits<float>::max()) {
                if (dist_k < global_min_dist)
                    global_min_dist = dist_k;
                if (dist_k > global_max_dist)
                    global_max_dist = dist_k;
            }

            for (int j = 0; j < k; ++j) {
                all_preds_list[row][col][j] = labels[row * k + j];
            }
        }
    }

    // Normalize and Regularize Matrix
    float gamma = 0.01f;
    int c_reg = 2;
    std::vector<std::vector<float>> reg_nonconf_matrix(
            nq_calib, std::vector<float>(num_ef, 0.0f));

    for (size_t row = 0; row < nq_calib; ++row) {
        for (size_t col = 0; col < num_ef; ++col) {
            float norm_score = (nonconf_matrix[row][col] - global_min_dist) /
                    (global_max_dist - global_min_dist);

            norm_score = std::max(0.0f, std::min(1.0f, norm_score));
            float penalty = gamma * std::max(0, static_cast<int>(col) - c_reg);
            reg_nonconf_matrix[row][col] = norm_score + penalty;
        }
    }

    // CRC Optimization
    float B = 1.0f;
    float best_lambda = -1.0f;

    for (float lambda_cand = 0.0f; lambda_cand <= 1.5f; lambda_cand += 0.005f) {
        float empirical_risk_sum = 0.0f;

        for (size_t row = 0; row < nq_calib; ++row) {
            int selected_col = num_ef - 1;

            for (size_t col = 0; col < num_ef; ++col) {
                if (reg_nonconf_matrix[row][col] <= lambda_cand) {
                    selected_col = col;
                    break;
                }
            }

            const int* gt_row = &calib_gt[row * d_gt_calib];
            const auto& preds = all_preds_list[row][selected_col];

            float fnr = calculate_fnr(preds.data(), gt_row, k);
            empirical_risk_sum += fnr;
        }

        float expected_risk = (nq_calib * (empirical_risk_sum / nq_calib) + B) /
                (nq_calib + 1);

        if (expected_risk <= target_alpha) {
            best_lambda = lambda_cand;
        }
    }

    ConannCalibParams params = {
            best_lambda,
            gamma,
            c_reg,
            global_min_dist,
            global_max_dist,
            efSearch_values};
    std::cerr << "Optimal Lambda (lamhat) found: " << best_lambda << std::endl;

    // -------------------------------------------------------------
    // STEP 4: Online Search Phase
    // -------------------------------------------------------------
    std::cerr << "4. Running Adaptive Online Search..." << std::endl;
    std::vector<faiss::idx_t> all_labels(nq_test * k);
    std::vector<float> all_distances(nq_test * k);
    std::vector<int> tracking_ef_used;

    std::vector<faiss::HNSWSearchCache> test_caches;
    for (size_t i = 0; i < nq_test; ++i) {
        test_caches.emplace_back(1200);
    }

    double t_start = omp_get_wtime();
    search_hnsw_conann_batch(
            index,
            nq_test,
            test_queries,
            k,
            params,
            all_labels.data(),
            all_distances.data(),
            tracking_ef_used,
            test_caches);
    double t_end = omp_get_wtime();
    double time_taken = t_end - t_start;

    std::cerr << "Search completed in " << time_taken << " seconds."
              << std::endl;

    // -------------------------------------------------------------
    // STEP 5: Evaluate FNR & Efficiency
    // -------------------------------------------------------------
    std::cerr << "5. Evaluating Accuracy (FNR) & Efficiency..." << std::endl;
    float total_fnr = 0.0f;
    long long total_ef = 0;

    for (size_t i = 0; i < nq_test; ++i) {
        total_fnr +=
                calculate_fnr(&all_labels[i * k], &test_gt[i * d_gt_test], k);
        total_ef += tracking_ef_used[i];
    }

    float empirical_fnr = total_fnr / nq_test;
    float avg_ef = static_cast<float>(total_ef) / nq_test;

    std::cerr << "------------------------------------------------"
              << std::endl;
    std::cerr << "Empirical Expected FNR: " << (empirical_fnr * 100.0f) << "%"
              << std::endl;
    std::cerr << "Average efSearch used:  " << avg_ef << std::endl;
    std::cerr << "------------------------------------------------"
              << std::endl;

    // Output final data to stdout for bash to capture
    std::cout << target_alpha << "," << best_lambda << "," << time_taken << ","
              << avg_ef << "," << empirical_fnr << std::endl;

    // Cleanup
    delete[] xb;
    delete[] calib_queries;
    delete[] calib_gt;
    delete[] test_queries;
    delete[] test_gt;

    return 0;
}