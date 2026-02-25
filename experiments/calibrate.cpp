#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <sys/stat.h>
#include <cstring>
#include <faiss/IndexHNSW.h>

// Helper to read .fvecs / .ivecs files
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

void save_conann_params(const std::string& filename, float lambda_hat, float gamma, int c_reg, float min_dist, float max_dist, const std::vector<int>& ef_schedule) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write scalar parameters
    out << lambda_hat << "\n";
    out << gamma << "\n";
    out << c_reg << "\n";
    out << min_dist << "\n";
    out << max_dist << "\n";
    
    // Write schedule size, then the schedule itself
    out << ef_schedule.size() << "\n";
    for (int ef : ef_schedule) {
        out << ef << " ";
    }
    out << "\n";
    
    out.close();
    std::cout << "Successfully saved ConANN parameters to " << filename << std::endl;
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


    size_t d_base, nb, d_q, nq_calib, d_gt, n_gt_calib;
    
    // -------------------------------------------------------------
    // STEP 1: Load Datasets
    // -------------------------------------------------------------
    std::cout << "Loading base dataset..." << std::endl;
    float* xb = vecs_read<float>("../sift1M/sift_base.fvecs", &d_base, &nb); 
    
    std::cout << "Loading calibration queries and ground truth..." << std::endl;
    float* calib_queries = vecs_read<float>("../sift1M/sift_query_calib.fvecs", &d_q, &nq_calib);
    int* calib_gt = vecs_read<int>("../sift1M/sift_groundtruth_calib.ivecs", &d_gt, &n_gt_calib);
    
    if (!xb || !calib_queries || !calib_gt) {
        std::cerr << "Failed to load one or more files. Check paths." << std::endl;
        return -1;
    }

    // -------------------------------------------------------------
    // STEP 2: Build the HNSW Index
    // -------------------------------------------------------------
    int M = 32;
    faiss::IndexHNSWFlat index(d_base, M);
    std::cout << "Building HNSW index on " << nb << " vectors..." << std::endl;
    index.add(nb, xb);
    
    // -------------------------------------------------------------
    // STEP 3: Define Parameter Space & Build Non-Conformity Matrix
    // -------------------------------------------------------------
    std::vector<int> efSearch_values;
    // Stepping by 5 instead of 20 for a smoother CRC curve
    for (int ef = 1; ef <= 1024; ef += 20) {
        efSearch_values.push_back(ef);
    }
    size_t num_ef = efSearch_values.size();
    
    std::vector<std::vector<float>> nonconf_matrix(nq_calib, std::vector<float>(num_ef, 0.0f));
    
    std::vector<std::vector<std::vector<faiss::idx_t>>> all_preds_list(
        nq_calib, std::vector<std::vector<faiss::idx_t>>(num_ef, std::vector<faiss::idx_t>(k))
    );
    
    float global_min_dist = std::numeric_limits<float>::max();
    float global_max_dist = 0.0f;
    
    std::vector<float> distances(nq_calib * k);
    std::vector<faiss::idx_t> labels(nq_calib * k);

    std::cout << "Building Non-Conformity Matrix across " << num_ef << " efSearch values..." << std::endl;
    for (size_t col = 0; col < num_ef; ++col) {
        int ef = efSearch_values[col];
        index.hnsw.efSearch = ef; 
        
        index.search(nq_calib, calib_queries, k, distances.data(), labels.data());
        
        for (size_t row = 0; row < nq_calib; ++row) {
            float dist_k = distances[row * k + (k - 1)];
            nonconf_matrix[row][col] = dist_k;
            
            if (dist_k < global_min_dist) global_min_dist = dist_k;
            if (dist_k > global_max_dist) global_max_dist = dist_k;
            
            for (int j = 0; j < k; ++j) {
                all_preds_list[row][col][j] = labels[row * k + j];
            }
        }
    }
    
    // Normalize Matrix
    for (size_t row = 0; row < nq_calib; ++row) {
        for (size_t col = 0; col < num_ef; ++col) {
            nonconf_matrix[row][col] = (nonconf_matrix[row][col] - global_min_dist) / 
                                       (global_max_dist - global_min_dist);
        }
    }

    // -------------------------------------------------------------
    // STEP 4: Implement RAPS Regularization
    // -------------------------------------------------------------
    float gamma = 0.01f;
    int c_reg = 2;
    
    std::vector<std::vector<float>> reg_nonconf_matrix(nq_calib, std::vector<float>(num_ef, 0.0f));
    
    for (size_t row = 0; row < nq_calib; ++row) {
        for (size_t col = 0; col < num_ef; ++col) {
            // FIX 1: Removed the 1.0f - inversion. 
            // The score should simply be the normalized distance.
            float base_score = nonconf_matrix[row][col];
            float penalty = gamma * std::max(0, static_cast<int>(col) - c_reg);
            reg_nonconf_matrix[row][col] = base_score + penalty;
        }
    }

    // -------------------------------------------------------------
    // STEP 5: Perform CRC Optimization & Export Plot Data
    // -------------------------------------------------------------
    float B = 1.0f; // Upper bound of the FNR
    
    std::cout << "Running CRC Optimization for Alpha = " << target_alpha << "..." << std::endl;
    
    float best_lambda = -1.0f;
    std::ofstream csv("risk_vs_lambda.csv");
    csv << "lambda,risk\n";
    
    for (float lambda_cand = 0.0f; lambda_cand <= 1.5f; lambda_cand += 0.005f) {
        float empirical_risk_sum = 0.0f;
        
        for (size_t row = 0; row < nq_calib; ++row) {
            // FIX 2: Default to the maximum effort if the lambda threshold is never met.
            int selected_col = num_ef - 1; 
            
            for (size_t col = 0; col < num_ef; ++col) {
                if (reg_nonconf_matrix[row][col] <= lambda_cand) {
                    selected_col = col; 
                    break; // FIX 2: Break immediately! This simulates early stopping perfectly.
                }
            }
            
            const int* gt_row = &calib_gt[row * d_gt]; 
            const auto& preds = all_preds_list[row][selected_col]; 
            
            float fnr = calculate_fnr(preds.data(), gt_row, k); 
            empirical_risk_sum += fnr;
        }
        
        // Apply CRC finite-sample correction term
        float expected_risk = (nq_calib * (empirical_risk_sum / nq_calib) + B) / (nq_calib + 1);
        
        csv << lambda_cand << "," << expected_risk << "\n";
        
        // FIX 3: Removed `best_lambda < 0.0f` check to find the largest valid lambda.
        // As lambda_cand increases, risk increases. We want the highest lambda that stays below alpha.
        if (expected_risk <= target_alpha) {
            best_lambda = lambda_cand;
        }
    }
    csv.close();

    save_conann_params("conann_calibrated_params.txt", 
                       best_lambda, gamma, c_reg, 
                       global_min_dist, global_max_dist, efSearch_values);
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Optimal Lambda (lamhat) found: " << best_lambda << std::endl;
    std::cout << "Data saved to risk_vs_lambda.csv. Ready for Python plotting." << std::endl;

    delete[] xb; delete[] calib_queries; delete[] calib_gt;
    return 0;
}