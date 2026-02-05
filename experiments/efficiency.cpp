#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <chrono>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>

// --- Helper: Read fvecs ---
float* read_fvecs(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) { fprintf(stderr, "Err: cannot open %s\n", fname); exit(1); }
    int d;
    fread(&d, 1, sizeof(int), f);
    *d_out = d;
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    *n_out = sz / (sizeof(int) + d * sizeof(float));
    float* x = new float[*n_out * d];
    for (size_t i = 0; i < *n_out; i++) {
        int d_check;
        fread(&d_check, 1, sizeof(int), f);
        fread(x + i * d, sizeof(float), d, f);
    }
    fclose(f);
    return x;
}

// --- Argument Parser Helper ---
int get_arg(int argc, char** argv, std::string key, int def) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == key) return std::atoi(argv[i+1]);
    }
    return def;
}
std::string get_arg_str(int argc, char** argv, std::string key, std::string def) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == key) return std::string(argv[i+1]);
    }
    return def;
}

int main(int argc, char** argv) {
    // 1. Parse Args
    int M = get_arg(argc, argv, "--M", 32);
    int efConstruction = get_arg(argc, argv, "--efConstruction", 40);
    int efSearch = get_arg(argc, argv, "--efSearch", 64);
    int k = get_arg(argc, argv, "--k", 10);
    int n_queries = get_arg(argc, argv, "--n_queries", 5);
    std::string data_dir = get_arg_str(argc, argv, "--data", "sift1M");
    std::string result_dir = get_arg_str(argc, argv, "--result_dir", "undefined_exp");
    std::string exp_name = get_arg_str(argc, argv, "--exp_name", "undefined_exp");

    // 2. Load Data
    size_t d, nb, nq, d_ignore;
    std::string base_path = data_dir + "/sift_base.fvecs";
    std::string query_path = data_dir + "/sift_query.fvecs";

    float* xb = read_fvecs(base_path.c_str(), &d, &nb);
    float* xq = read_fvecs(query_path.c_str(), &d_ignore, &nq);

    // 3. Build Index
    // We turn verbose OFF here so it doesn't spam build times
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.verbose = false; 
    
    // std::cout << "Building index..." << std::endl;
    index.add(nb, xb);

    // 4. Search
    index.hnsw.efSearch = efSearch;
    
    std::vector<faiss::idx_t> I(n_queries * k);
    std::vector<float> D(n_queries * k);

    // --- TIMING START ---
    auto start = std::chrono::high_resolution_clock::now();
    
    index.search(n_queries, xq, k, D.data(), I.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    // --- TIMING END ---

    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 5. Output for Python
    // IMPORTANT: This is the exact line the python script looks for
    std::cout << "TIME_MS=" << duration_ms << std::endl;

    // 6. Save results for recall calculation
    // Ensure the directory exists
    struct stat st = {0};
    if (stat("results", &st) == -1) mkdir("results", 0700);

    FILE* f_out = fopen((result_dir + "/" + exp_name + ".bin").c_str(), "wb");
    if (f_out) {
        fwrite(I.data(), sizeof(faiss::idx_t), n_queries * k, f_out);
        fclose(f_out);
    } else {
        std::cerr << "Error writing " << result_dir << "/" << exp_name << ".bin" << std::endl;
    }

    delete[] xb;
    delete[] xq;
    return 0;
}