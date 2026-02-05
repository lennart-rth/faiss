import os
import json
import matplotlib.pyplot as plt
import glob

# ================= CONFIGURATION =================
RESULTS_DIR = "latency_recall"
OUTPUT_FILENAME = "latency_recall_comparison.png"
# =================================================

def load_data(results_dir):
    """
    Reads all .json files in the directory.
    Returns a dict: { "filename_without_extension": [data_points] }
    """
    all_data = {}
    
    # Find all json files
    json_pattern = os.path.join(results_dir, "*.json")
    files = glob.glob(json_pattern)
    
    if not files:
        print(f"No JSON files found in {results_dir}")
        return {}

    for filepath in files:
        filename = os.path.basename(filepath)
        exp_name = os.path.splitext(filename)[0] 
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list) and len(data) > 0 and "recall" in data[0]:
                data.sort(key=lambda x: x["recall"])
                all_data[exp_name] = data
            else:
                print(f"Skipping {filename}: Invalid format or empty.")
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return all_data

def plot_comparison(all_data):
    plt.figure(figsize=(10, 7))
    
    # Define a set of markers to cycle through
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    
    # --- MODIFICATION: SORT KEYS ALPHABETICALLY ---
    sorted_exp_names = sorted(all_data.keys())
    
    for i, exp_name in enumerate(sorted_exp_names):
        points = all_data[exp_name]
        
        recalls = [p["recall"] for p in points]
        latencies = [p["latency_ms"] for p in points]
        
        # Select marker based on index
        marker = markers[i % len(markers)]
        
        # Plot line
        plt.plot(recalls, latencies, marker=marker, markersize=5, 
                 linewidth=2, label=exp_name, alpha=0.8)

    # Styling
    plt.xlabel("Recall@10", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.title("HNSW Latency vs. Recall Comparison", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    
    # Legend will now automatically follow the plot order (Alphabetical)
    plt.legend(title="Experiment Name")
    
    plt.tight_layout()
    
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"Plot saved to {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    print(f"Reading data from {RESULTS_DIR}...")
    data = load_data(RESULTS_DIR)
    
    if data:
        plot_comparison(data)
    else:
        print("No valid data found to plot.")