#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <sys/stat.h>
#include <cstring>

// Read .fvecs or .ivecs
template <typename T>
T* vecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) return nullptr;
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

// Write .fvecs or .ivecs
template <typename T>
void vecs_write(const char* fname, size_t d, size_t n, const T* data) {
    FILE* f = fopen(fname, "w");
    int d_int = (int)d;
    for (size_t i = 0; i < n; i++) {
        fwrite(&d_int, 1, sizeof(int), f);
        fwrite(data + i * d, sizeof(T), d, f);
    }
    fclose(f);
}

int main() {
    size_t d, nq, d_gt, n_gt;
    
    // Read original files
    float* xq = vecs_read<float>("../../sift1M/sift_query.fvecs", &d, &nq);
    int* gt = vecs_read<int>("../../sift1M/sift_groundtruth.ivecs", &d_gt, &n_gt);
    
    size_t calib_size = nq / 2; // 50%
    
    // Write 50% to new files
    vecs_write<float>("../../sift1M/sift_query_calib.fvecs", d, calib_size, xq);
    vecs_write<int>("../../sift1M/sift_groundtruth_calib.ivecs", d_gt, calib_size, gt);
    
    std::cout << "Successfully extracted " << calib_size << " queries for calibration." << std::endl;
    
    delete[] xq; delete[] gt;
    return 0;
}