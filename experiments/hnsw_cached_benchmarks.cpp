#include <iostream>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <sys/stat.h>
#include <faiss/IndexHNSW.h>
#include <chrono>
#include <unistd.h> // For sysconf

// Linux-specific helper to get current memory usage (Resident Set Size) in MB
double get_current_rss_mb() {
    long rss = 0L;
    FILE* fp = nullptr;
    if ((fp = fopen("/proc/self/statm", "r")) == nullptr) return 0.0;
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return 0.0;
    }
    fclose(fp);
    return (double)rss * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
}

float calculate_recall(const int* gt, const faiss::idx_t* labels, size_t nq, int k, size_t d_gt) {
    size_t correct = 0;
    for (size_t i = 0; i < nq; i++) {
        std::unordered_set<int> ground_truth_set;
        // SIFT ground truth often provides 100 neighbors, we only check top K
        for (int j = 0; j < k; j++) {
            ground_truth_set.insert(gt[i * d_gt + j]);
        }
        for (int j = 0; j < k; j++) {
            if (ground_truth_set.count(labels[i * k + j])) {
                correct++;
            }
        }
    }
    return (float)correct / (nq * k);
}

// Helper to read .fvecs files
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

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <ef1> <ef2>\n";
        return 1;
    }

    int ef1 = std::stoi(argv[1]);
    int ef2 = std::stoi(argv[2]);
    int k = 10;
    int M = 32;

    size_t d, nb, d_q, nq, d_gt, n_gt;
    std::cerr << "Loading SIFT dataset...\n";
    float* xb = vecs_read<float>("../sift1M/sift_base.fvecs", &d, &nb);
    float* xq = vecs_read<float>("../sift1M/sift_query.fvecs", &d_q, &nq);
    int* gt = vecs_read<int>("../sift1M/sift_groundtruth.ivecs", &d_gt, &n_gt);
    
    if (!xb || !xq || !gt) return 1;

    // -------------------------------------------------------------
    // Build HNSW
    // -------------------------------------------------------------
    std::cerr << "Building HNSW index...\n";
    faiss::IndexHNSWFlat index(d, M, faiss::METRIC_L2);
    index.add(nb, xb);

    std::vector<faiss::idx_t> labels_norm1(nq * k), labels_norm2(nq * k);
    std::vector<float> dists_norm1(nq * k), dists_norm2(nq * k);

    std::vector<faiss::idx_t> labels_res1(nq * k), labels_res2(nq * k);
    std::vector<float> dists_res1(nq * k), dists_res2(nq * k);

    // -------------------------------------------------------------
    // NORMAL SEARCH PIPELINE
    // -------------------------------------------------------------
    std::cerr << "Running Normal Search Pipeline...\n";
    index.hnsw.efSearch = ef1;
    index.search(nq, xq, k, dists_norm1.data(), labels_norm1.data());

    // Measure memory overhead & time for the full normal search
    double mem_before_normal = get_current_rss_mb();
    index.hnsw.efSearch = ef2;
    
    auto start_norm = std::chrono::steady_clock::now();
    index.search(nq, xq, k, dists_norm2.data(), labels_norm2.data());
    auto end_norm = std::chrono::steady_clock::now();
    
    double memory_normal_overhead_mb = get_current_rss_mb() - mem_before_normal;
    double time_normal = std::chrono::duration<double>(end_norm - start_norm).count();

    // -------------------------------------------------------------
    // RESUME SEARCH PIPELINE
    // -------------------------------------------------------------
    std::cerr << "Running Resume Search Pipeline...\n";
    
    double mem_before_resume = get_current_rss_mb();
    
    // Allocate Caches
    std::vector<faiss::HNSWSearchCache> caches;
    for (size_t i = 0; i < nq; ++i) {
        caches.emplace_back(ef2 + 100); 
    }
    std::vector<faiss::HNSWSearchCache*> cache_ptrs(nq);
    for (size_t i = 0; i < nq; ++i) cache_ptrs[i] = &caches[i];

    // Search 1 (Initial search up to ef1)
    index.hnsw.efSearch = ef1;
    index.search_resume(nq, xq, k, dists_res1.data(), labels_res1.data(), cache_ptrs);

    // Measure memory overhead AFTER caches are fully populated
    double memory_resume_overhead_mb = get_current_rss_mb() - mem_before_resume;

    // Search 2 (Resuming from ef1 to ef2)
    index.hnsw.efSearch = ef2;
    auto start_res = std::chrono::steady_clock::now();
    index.search_resume(nq, xq, k, dists_res2.data(), labels_res2.data(), cache_ptrs);
    auto end_res = std::chrono::steady_clock::now();
    
    double time_resume = std::chrono::duration<double>(end_res - start_res).count();


    float recall_norm_ef2 = calculate_recall(gt, labels_norm2.data(), nq, k, d_gt);
    float recall_res_ef1 = calculate_recall(gt, labels_res1.data(), nq, k, d_gt);
    float recall_res_ef2 = calculate_recall(gt, labels_res2.data(), nq, k, d_gt);


    // -------------------------------------------------------------
    // OVERLAP CALCULATION
    // -------------------------------------------------------------
    long long total_overlap = 0;
    for (size_t i = 0; i < nq; i++) {
        std::unordered_set<faiss::idx_t> set1;
        for (int j = 0; j < k; j++) set1.insert(labels_res1[i * k + j]);

        for (int j = 0; j < k; j++) {
            if (set1.find(labels_res2[i * k + j]) != set1.end()) {
                total_overlap++;
            }
        }
    }
    float overlap_pct = (static_cast<float>(total_overlap) / (nq * k)) * 100.0f;

    // -------------------------------------------------------------
    // OUTPUT TO STDOUT (CSV FORMAT)
    // -------------------------------------------------------------
    std::cout << ef1 << "," << ef2 << "," 
              << time_normal << "," << time_resume << "," 
              << overlap_pct << "," 
              << memory_normal_overhead_mb << "," << memory_resume_overhead_mb << "," 
              << recall_norm_ef2 << "," << recall_res_ef1 << "," << recall_res_ef2 << "\n";

    delete[] xb;
    delete[] xq;
    delete[] gt;
    return 0;
}