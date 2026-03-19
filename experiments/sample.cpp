#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>

struct OutputSet {
    std::string filename;
    double ratio;
    size_t count;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage (2 inputs): " << argv[0] << " <in1.fvecs> <in2.fvecs> <out1.fvecs> <ratio1> ...\n";
        std::cerr << "Usage (1 input):  " << argv[0] << " <in1.fvecs> <out1.fvecs> <ratio1> ...\n";
        return 1;
    }

    std::string file1 = argv[1];
    std::string file2 = "";
    int start_idx = 2;

    // THE FIX: Check argument count parity (Even vs Odd)
    // 1 Input file  -> ./sample in out 0.5 out 0.5 -> 6 args (Even)
    // 2 Input files -> ./sample in1 in2 out 0.5 out 0.5 -> 7 args (Odd)
    if (argc % 2 != 0) {
        file2 = argv[2];
        start_idx = 3;
    }

    std::vector<OutputSet> outputs;
    double total_ratio = 0.0;
    for (int i = start_idx; i < argc; i += 2) {
        double r = std::stod(argv[i+1]);
        outputs.push_back({argv[i], r, 0});
        total_ratio += r;
    }

    if (std::abs(total_ratio - 1.0) > 1e-5) {
        std::cerr << "Error: Ratios must sum to 1.0. Current sum: " << total_ratio << "\n";
        return 1;
    }

    // 1. Get dimension from first file
    std::ifstream in1;
    in1.open(file1, std::ios::binary);
    if (!in1) { std::cerr << "Error opening " << file1 << "\n"; return 1; }

    uint32_t d;
    in1.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
    size_t row_bytes = sizeof(uint32_t) + d * sizeof(float);
    
    in1.seekg(0, std::ios::end);
    size_t N1 = in1.tellg() / row_bytes;
    in1.seekg(0, std::ios::beg);

    // 2. Load data from file1
    size_t N2 = 0;
    if (!file2.empty()) {
        std::ifstream in2_probe;
        in2_probe.open(file2, std::ios::binary);
        if (!in2_probe) { std::cerr << "Error opening " << file2 << "\n"; return 1; }
        in2_probe.seekg(0, std::ios::end);
        N2 = in2_probe.tellg() / row_bytes;
        in2_probe.close();
    }

    size_t N = N1 + N2;
    std::cout << "Loading " << N << " vectors total..." << std::endl;
    std::vector<char> data(N * row_bytes);
    
    // Read file 1
    in1.read(data.data(), N1 * row_bytes);
    in1.close();
    
    // Read file 2 if it exists
    if (!file2.empty()) {
        std::ifstream in2;
        in2.open(file2, std::ios::binary);
        in2.read(data.data() + N1 * row_bytes, N2 * row_bytes);
        in2.close();
    }

    // 3. Shuffle
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));

    // 4. Split and Write
    size_t distributed = 0;
    size_t current_idx = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i].count = (i == outputs.size() - 1) ? (N - distributed) : (size_t)std::round(N * outputs[i].ratio);
        distributed += outputs[i].count;

        std::ofstream out_f;
        out_f.open(outputs[i].filename, std::ios::binary);
        if (!out_f) { std::cerr << "Error writing to " << outputs[i].filename << "\n"; return 1; }

        for (size_t j = 0; j < outputs[i].count; ++j) {
            size_t pos = indices[current_idx++];
            out_f.write(data.data() + pos * row_bytes, row_bytes);
        }
        std::cout << "Saved " << outputs[i].count << " to " << outputs[i].filename << "\n";
        out_f.close();
    }

    return 0;
}