import os
import subprocess
import numpy as np
import json
import struct

# ================= CONFIGURATION =================
# We sweep 'efSearch'. 
# Low values = Fast but imprecise. High values = Slow but accurate.
SWEEP_PARAMS = [4,8,16, 32, 64, 128]
EXP_NAME = "hnsw"

CONFIG = {
    "binary_path": "experiments/build/efficiency",
    "ground_truth_path": "sift1M/sift_groundtruth.ivecs",
    "results_bin_path": f"results/{EXP_NAME}.bin",
    "output_dir": "results/latency_recall",
    "M": 32,
    "efConstruction": 40,
    "k": 10,
    "n_queries": 10000  # Use all 10k queries for stable stats
}
# =================================================

def read_ivecs(fname):
    """
    Reads a standard .ivecs file (the format used for SIFT1M Ground Truth).
    Returns a numpy array of shape (n_queries, n_neighbors).
    """
    if not os.path.exists(fname):
        print(f"Error: Ground truth file not found at {fname}")
        exit(1)
        
    a = np.fromfile(fname, dtype='int32')
    d = a[0] # The first 4 bytes indicate dimension (number of neighbors provided)
    # The file format is [d, id1, id2, ... id_d, d, id1, ...]
    # We reshape it to rows of (d + 1) columns, then drop the first column (which is just 'd')
    return a.reshape(-1, d + 1)[:, 1:].copy()

def compute_recall_at_k(results, ground_truth, k):
    """
    Computes Recall@K (Intersection / K).
    Since we ask for top-K candidates, we check how many of the *actual* top-K
    are present in our results.
    """
    n_queries = results.shape[0]
    recall_sum = 0
    
    # Slice to top-K just in case
    GT = ground_truth[:, :k]
    I = results[:, :k]
    
    for i in range(n_queries):
        # Intersection of two sets
        # (Using python sets is fast enough for 10k queries with small k)
        truth_set = set(GT[i])
        found_set = set(I[i])
        recall_sum += len(truth_set.intersection(found_set))
        
    return recall_sum / (n_queries * k)

def run_single_experiment(efSearch):
    """
    Runs the C++ binary for a specific parameter setting.
    Returns: (latency_ms, indices_array)
    """
    cmd = [
        CONFIG["binary_path"],
        "--data", "sift1M",
        "--M", str(CONFIG["M"]),
        "--efConstruction", str(CONFIG["efConstruction"]),
        "--efSearch", str(efSearch),
        "--k", str(CONFIG["k"]),
        "--n_queries", str(CONFIG["n_queries"])
    ]

    my_env = os.environ.copy()
    my_env["HNSW_ENABLE_LOGGING"] = "0"

    if EXP_NAME == "naiveES":
        my_env["HNSW_NAIVE_ES"] = "0"
        my_env["HNSW_PATIENCE"] = "0"
    
    # 1. Run C++ Binary
    # We assume the binary prints "TIME_MS=123.45" to stdout
    result = subprocess.run(cmd, capture_output=True, text=True, env=my_env)
    
    if result.returncode != 0:
        print(f"Binary failed for ef={efSearch}!")
        print(result.stderr)
        return None, None

    # 2. Parse Latency from Stdout
    latency_ms = None
    for line in result.stdout.split('\n'):
        if line.startswith("TIME_MS="):
            latency_ms = float(line.split("=")[1])
            break
            
    if latency_ms is None:
        print(f"Could not find 'TIME_MS=' in output for ef={efSearch}")
        return None, None
        
    # 3. Read Result Indices from Binary File
    # The C++ code writes raw int64 (faiss::idx_t) to results/search_output.bin
    with open(CONFIG["results_bin_path"], "rb") as f:
        data = f.read()
        indices = np.frombuffer(data, dtype=np.int64)
        indices = indices.reshape(CONFIG["n_queries"], CONFIG["k"])
        
    return latency_ms, indices

def main():
    # Setup Output
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    out_file = os.path.join(CONFIG["output_dir"], f"{EXP_NAME}.json")
    
    # Load Ground Truth
    print("Loading Ground Truth...")
    gt_indices = read_ivecs(CONFIG["ground_truth_path"])
    print(f"Ground Truth Loaded. Shape: {gt_indices.shape}")
    
    results_log = []
    
    print(f"{'efSearch':<10} | {'Latency (ms/q)':<15} | {'Recall@10':<10}")
    print("-" * 45)
    
    for ef in SWEEP_PARAMS:
        total_time_ms, indices = run_single_experiment(ef)
        
        if indices is None:
            continue
            
        # Calc Recall
        recall = compute_recall_at_k(indices, gt_indices, CONFIG["k"])
        
        # Calc Latency per Query
        latency_per_query = total_time_ms / CONFIG["n_queries"]
        
        print(f"{ef:<10} | {latency_per_query:<15.4f} | {recall:<10.4f}")
        
        results_log.append({
            "efSearch": ef,
            "latency_ms": latency_per_query,
            "recall": recall,
            "k": CONFIG["k"]
        })

    # Save Results
    with open(out_file, "w") as f:
        json.dump(results_log, f, indent=4)
        
    print(f"\nSweep complete. Data saved to {out_file}")

if __name__ == "__main__":
    main()