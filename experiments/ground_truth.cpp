#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>

// Structure to hold distance and index for our max-heap
struct Neighbor {
    float distance;
    uint32_t index;
    bool operator<(const Neighbor& other) const {
        return distance < other.distance; // Max-heap: largest distance stays at the top
    }
};

// Helper function to read .fvecs files into a flat std::vector<float>
bool read_fvecs(const std::string& filename, std::vector<float>& data, uint32_t& num_vectors, uint32_t& dim) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) { std::cerr << "Cannot open " << filename << "\n"; return false; }
    
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    
    in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    in.seekg(0, std::ios::beg);
    
    size_t row_bytes = sizeof(uint32_t) + dim * sizeof(float);
    num_vectors = file_size / row_bytes;
    
    data.resize(num_vectors * dim);
    
    for (size_t i = 0; i < num_vectors; ++i) {
        uint32_t d;
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(float));
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <query.fvecs> <base.fvecs> <groundtruth.ivecs> <k>\n";
        return 1;
    }

    std::string query_file = argv[1];
    std::string base_file = argv[2];
    std::string gt_file = argv[3];
    uint32_t k = std::stoi(argv[4]);

    std::vector<float> queries, base;
    uint32_t num_queries, num_base, dim_q, dim_b;

    std::cout << "Loading base vectors...\n";
    if (!read_fvecs(base_file, base, num_base, dim_b)) return 1;

    std::cout << "Loading query vectors...\n";
    if (!read_fvecs(query_file, queries, num_queries, dim_q)) return 1;

    if (dim_q != dim_b) {
        std::cerr << "Error: Dimension mismatch! Queries: " << dim_q << ", Base: " << dim_b << "\n";
        return 1;
    }

    if (k > num_base) {
        std::cerr << "Error: k (" << k << ") is larger than base dataset size (" << num_base << ").\n";
        return 1;
    }

    uint32_t dim = dim_q;
    std::vector<std::vector<uint32_t>> ground_truth(num_queries, std::vector<uint32_t>(k));

    std::cout << "Computing exact " << k << "-NN for " << num_queries << " queries against " << num_base << " base vectors...\n";
    std::cout << "This may take some time depending on your CPU.\n";

    // OpenMP directive to multithread the outer loop
    #pragma omp parallel for
    for (int i = 0; i < (int)num_queries; ++i) {
        std::priority_queue<Neighbor> max_heap;
        const float* q_vec = queries.data() + i * dim;

        for (uint32_t j = 0; j < num_base; ++j) {
            const float* b_vec = base.data() + j * dim;
            float dist_sq = 0.0f;
            
            // Calculate Squared L2 Distance
            for (uint32_t d = 0; d < dim; ++d) {
                float diff = q_vec[d] - b_vec[d];
                dist_sq += diff * diff;
            }

            // Maintain a max-heap of size k
            if (max_heap.size() < k) {
                max_heap.push({dist_sq, j});
            } else if (dist_sq < max_heap.top().distance) {
                max_heap.pop();
                max_heap.push({dist_sq, j});
            }
        }

        // Extract from heap (they come out in reverse order, largest first)
        for (int j = k - 1; j >= 0; --j) {
            ground_truth[i][j] = max_heap.top().index;
            max_heap.pop();
        }
        
        // Progress tracker for the main thread
        #pragma omp critical
        {
            static int done = 0;
            done++;
            if (done % 100 == 0 || done == num_queries) {
                std::cout << "\rProcessed " << done << " / " << num_queries << " queries" << std::flush;
            }
        }
    }
    std::cout << "\n";

    // Write the results to .ivecs format
    std::cout << "Writing ground truth to " << gt_file << "...\n";
    std::ofstream out(gt_file, std::ios::binary);
    if (!out) { std::cerr << "Cannot open output file.\n"; return 1; }

    for (uint32_t i = 0; i < num_queries; ++i) {
        out.write(reinterpret_cast<const char*>(&k), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(ground_truth[i].data()), k * sizeof(uint32_t));
    }

    std::cout << "Done!\n";
    return 0;
}