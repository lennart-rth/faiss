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
    // Changed arguments to take k and ef_search
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <k> <ef_search>" << std::endl;
        return 1;
    }

    int k = std::stoi(argv[1]);
    int ef_search = std::stoi(argv[2]);

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
    // M=32 to match your original script
    faiss::IndexHNSWFlat index(d_base, 32);
    index.add(nb, xb);
    
    std::cout << "3. Setting up Standard HNSW parameters..." << std::endl;
    // Set the global efSearch parameter for the FAISS index
    index.hnsw.efSearch = ef_search;
    std::cout << "  - k: " << k << "\n"
              << "  - efSearch: " << ef_search << std::endl;

    std::vector<faiss::idx_t> all_labels(nq * k);
    std::vector<float> all_distances(nq * k);

    std::cout << "4. Running Standard Batch Search..." << std::endl;
    double t_start = omp_get_wtime();
    
    // FAISS naturally parallelizes batch searches via OpenMP internally,
    // so we don't need a custom #pragma omp parallel for loop here.
    index.search(nq, xq, k, all_distances.data(), all_labels.data());
    
    double t_end = omp_get_wtime();
    std::cout << "Search completed in " << (t_end - t_start) << " seconds." << std::endl;

    std::cout << "5. Evaluating Accuracy (FNR)..." << std::endl;
    float total_fnr = 0.0f;
    
    for (size_t i = 0; i < nq; ++i) {
        total_fnr += calculate_fnr(&all_labels[i * k], &gt[i * d_gt], k);
    }
    
    float empirical_fnr = total_fnr / nq;
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Empirical Expected FNR: " << (empirical_fnr * 100.0f) << "%" << std::endl;
    std::cout << "Fixed efSearch used:    " << ef_search << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    delete[] xb; delete[] xq; delete[] gt;
    return 0;
}