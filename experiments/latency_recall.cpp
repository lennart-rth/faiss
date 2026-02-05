#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <chrono>
#include <algorithm>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/utils.h>

// --- Helper: Read fvecs (Unchanged) ---
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

// --- Helper: Parse comma-separated integers ---
std::vector<int> parse_int_list(std::string text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        values.push_back(std::stoi(item));
    }
    return values;
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
    // 1. Parse Args
    int M = get_arg_int(argc, argv, "--M", 32);
    int efConstruction = get_arg_int(argc, argv, "--efConstruction", 40);
    int k = get_arg_int(argc, argv, "--k", 10);
    int n_queries = get_arg_int(argc, argv, "--n_queries", 5);
    
    std::string data_dir = get_arg_str(argc, argv, "--data", "sift1M");
    std::string result_dir = get_arg_str(argc, argv, "--result_dir", "results/efficiency");

    // Parse the list of efSearch values (e.g., "16,32,64,128")
    std::string ef_str = get_arg_str(argc, argv, "--efSearch", "16,32,64,128,256");
    std::string exp_name = get_arg_str(argc, argv, "--exp_name", "undefinded_exp");
    int patience = get_arg_int(argc, argv, "--patience", 100);
    std::vector<int> ef_values = parse_int_list(ef_str);

    // 2. Load Data
    size_t d, nb, nq, d_ignore;
    std::string base_path = data_dir + "/sift_base.fvecs";
    std::string query_path = data_dir + "/sift_query.fvecs";

    std::cout << "[C++] Loading data..." << std::endl;
    float* xb = read_fvecs(base_path.c_str(), &d, &nb);
    float* xq = read_fvecs(query_path.c_str(), &d_ignore, &nq);

    // 3. Build Index (ONCE)
    std::cout << "[C++] Building Index (M=" << M << ", efC=" << efConstruction << ")..." << std::endl;
    faiss::IndexHNSWFlat index(d, M);
    index.hnsw.efConstruction = efConstruction;
    index.verbose = false; 
    index.add(nb, xb);
    std::cout << "[C++] Index built." << std::endl;

    // Ensure result directory exists
    struct stat st = {0};
    if (stat(result_dir.c_str(), &st) == -1) mkdir(result_dir.c_str(), 0700);

    // 4. Sweep Loop
    std::vector<faiss::idx_t> I(n_queries * k);
    std::vector<float> D(n_queries * k);



    std::cout << "[C++] Starting Sweep..." << std::endl;
    
    for (int ef : ef_values) {
        // A. Configure
        index.hnsw.efSearch = ef;

        // B. Search & Time
        auto start = std::chrono::high_resolution_clock::now();
        index.search(n_queries, xq, k, D.data(), I.data());
        auto end = std::chrono::high_resolution_clock::now();

        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // C. Generate Output Filename
        std::string filename = result_dir + "/" + exp_name + ".bin";

        // D. Save Results
        FILE* f_out = fopen(filename.c_str(), "wb");
        if (f_out) {
            fwrite(I.data(), sizeof(faiss::idx_t), n_queries * k, f_out);
            fclose(f_out);
        } else {
            std::cerr << "Error writing " << filename << std::endl;
        }

        // E. Print Result for Python Parsing
        // Format: EF_SEARCH=[val] TIME_MS=[val] FILE=[path]
        std::cout << "RESULT: efSearch=" << ef 
                  << " TIME_MS=" << duration_ms 
                  << " FILE=" << filename << std::endl;
    }

    delete[] xb;
    delete[] xq;
    return 0;
}