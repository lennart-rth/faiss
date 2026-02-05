import os
import subprocess
import numpy as np
import json
import shutil

PAT = 6
# ================= CONFIGURATION =================
CONFIG = {
    # Experiment Settings
    "exp_name": f"naiveES-{PAT}",          # Name for this run (output filename)
    "use_naive_es": True,               # Enable your C++ modification?
    "patience": PAT,                    # How many steps to wait
    "enable_logging": False,            # Write per-step CSVs? (Slows down search!)

    # Binary & Data Paths
    "binary_path": "experiments/build/latency_recall",  # Path to compiled C++ binary
    "ground_truth_path": "sift1M/sift_groundtruth.ivecs",
    "results_dir": "results/latency_recall",
    
    # HNSW Parameters
    "M": 32,
    "efConstruction": 40,
    "efSearch_list": "4,8,16,32,64,128",
    "k": 10,
    "n_queries": 10000
}
# =================================================

def read_ivecs(fname):
    """Parses standard .ivecs file format"""
    if not os.path.exists(fname):
        print(f"Error: GT file {fname} not found.")
        exit(1)
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def compute_recall(results_bin_path, gt_indices, k, n_queries):
    """Reads binary output from C++ and compares with Ground Truth"""
    if not os.path.exists(results_bin_path):
        print(f"Warning: Result file {results_bin_path} missing!")
        return 0.0
        
    with open(results_bin_path, "rb") as f:
        data = f.read()
        indices = np.frombuffer(data, dtype=np.int64) # faiss::idx_t is int64
        indices = indices.reshape(n_queries, k)
    
    recall_sum = 0
    GT = gt_indices[:, :k]
    I = indices[:, :k]
    
    for i in range(n_queries):
        recall_sum += len(set(GT[i]).intersection(set(I[i])))
        
    return recall_sum / (n_queries * k)

def main():
    # 1. Setup Directories
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    
    # Output file: results/latency_recall/naiveES-200.json
    output_json = os.path.join(CONFIG["results_dir"], f"{CONFIG['exp_name']}.json")

    # 2. Load Ground Truth
    print(f"Loading Ground Truth from {CONFIG['ground_truth_path']}...")
    gt_indices = read_ivecs(CONFIG['ground_truth_path'])

    # 3. Prepare Command
    cmd = [
        CONFIG["binary_path"],
        "--data", "sift1M",
        "--M", str(CONFIG["M"]),
        "--efConstruction", str(CONFIG["efConstruction"]),
        "--k", str(CONFIG["k"]),
        "--n_queries", str(CONFIG["n_queries"]),
        "--efSearch", CONFIG["efSearch_list"],
        "--result_dir", CONFIG["results_dir"], # Save bins directly here temporarily
        "--exp_name", CONFIG["exp_name"]
    ]

    # 4. Prepare Environment Variables
    env_vars = os.environ.copy()
    env_vars["HNSW_NAIVE_ES"] = "1" if CONFIG["use_naive_es"] else "0"
    env_vars["HNSW_PATIENCE"] = str(CONFIG["patience"])
    env_vars["HNSW_ENABLE_LOGGING"] = "1" if CONFIG["enable_logging"] else "0"

    print(f"\n🚀 Running Sweep: {CONFIG['exp_name']}")
    print(f"   - Naive ES: {CONFIG['use_naive_es']} (Patience: {CONFIG['patience']})")
    print(f"   - Command: {' '.join(cmd)}\n")

    # 5. Run Process
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        text=True, 
        env=env_vars 
    )
    
    results_log = []

    # 6. Parse C++ Output
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            # print(line.strip()) # Uncomment to see raw C++ output
            
            # Expected C++ Output format: 
            # RESULT: efSearch=16 TIME_MS=123.4 FILE=results/...
            if line.startswith("RESULT:"):
                try:
                    parts = line.split()
                    ef = int(parts[1].split("=")[1])
                    time_ms = float(parts[2].split("=")[1])
                    bin_file = parts[3].split("=")[1]
                    
                    recall = compute_recall(bin_file, gt_indices, CONFIG["k"], CONFIG["n_queries"])
                    latency = time_ms / CONFIG["n_queries"]
                    
                    print(f"   -> efSearch={ef:<4} | Latency={latency:.5f}ms | Recall={recall:.5f}")

                    # EXACT JSON STRUCTURE REQUESTED
                    results_log.append({
                        "efSearch": ef,
                        "latency_ms": latency,
                        "recall": recall,
                        "k": CONFIG["k"]
                    })
                    
                    # Optional: Cleanup the temporary .bin file to save space
                    # if os.path.exists(bin_file):
                    #     os.remove(bin_file)

                except Exception as e:
                    print(f"   [Error Parsing Line]: {e}")

    # 7. Save Final JSON
    with open(output_json, "w") as f:
        json.dump(results_log, f, indent=4)

    print(f"\n✅ Experiment Complete. Data saved to {output_json}")

if __name__ == "__main__":
    main()