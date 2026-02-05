import os
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
EXP_CONFIG = {
    "data_dir": "sift1M",
    "M": 32,                 # HNSW Graph Connections
    "efConstruction": 40,    # Build depth
    "efSearch": 200,         # Search depth (High value to see long convergence)
    "k": 10,                 # Neighbors to find
    "n_queries": 50           # Number of queries to log
}

PATHS = {
    "exp_src_dir": "experiments",
    "exp_build_dir": "experiments/build",
    "results_dir": "results/efficiency",
    "binary_name": "efficiency"
}
# =================================================

def compile_cpp():
    """Builds the C++ experiment binary using CMake."""
    print("🔨 Compiling C++ experiments...")
    os.makedirs(PATHS["exp_build_dir"], exist_ok=True)
    
    # Run CMake
    subprocess.check_call(
        ["cmake", ".."], 
        cwd=PATHS["exp_build_dir"]
    )
    
    # Run Make
    subprocess.check_call(
        ["make", "-j" + str(os.cpu_count())], 
        cwd=PATHS["exp_build_dir"]
    )
    print("✅ Compilation successful.\n")

def run_cpp_experiment():
    """Calls the compiled binary with config arguments."""
    binary_path = os.path.join(PATHS["exp_build_dir"], PATHS["binary_name"])
    
    cmd = [
        binary_path,
        "--M", str(EXP_CONFIG["M"]),
        "--efConstruction", str(EXP_CONFIG["efConstruction"]),
        "--efSearch", str(EXP_CONFIG["efSearch"]),
        "--k", str(EXP_CONFIG["k"]),
        "--n_queries", str(EXP_CONFIG["n_queries"]),
        "--data", EXP_CONFIG["data_dir"]
    ]

    my_env = os.environ.copy()
    my_env["HNSW_ENABLE_LOGGING"] = "1"
    
    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=my_env)
def plot_results():
    print("\n📊 Generating Plots...")
    log_files = sorted([f for f in os.listdir(PATHS["results_dir"]) if f.endswith(".csv")])
    
    if not log_files:
        print("No log files found! Did the C++ code run?")
        return

    plt.figure(figsize=(12, 7))
    
    for fname in log_files:
        path = os.path.join(PATHS["results_dir"], fname)
        try:
            # Load CSV: Col 0 = ndis, Col 1 = radius
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            if data.ndim < 2: continue
            
            plt.plot(data[:, 0], data[:, 1], alpha=0.7, linewidth=1.5, label=f"{fname}")
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    plt.xlabel("Distance Computations (ndis) - [Effort]")
    plt.ylabel("Distance to k-th Neighbor - [Accuracy]")
    plt.title(f"HNSW Early Stopping Analysis\n(M={EXP_CONFIG['M']}, efSearch={EXP_CONFIG['efSearch']})")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    
    out_file = os.path.join(PATHS["results_dir"], "convergence_plot.png")
    plt.savefig(out_file)
    print(f"✅ Plot saved to: {out_file}")

def main():
    # 1. Clean previous results
    if os.path.exists(PATHS["results_dir"]):
        shutil.rmtree(PATHS["results_dir"])
    os.makedirs(PATHS["results_dir"], exist_ok=True)

    # 2. Compile
    compile_cpp()

    # 3. Run
    run_cpp_experiment()

    # 4. Plot
    plot_results()

if __name__ == "__main__":
    main()