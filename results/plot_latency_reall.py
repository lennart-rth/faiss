import os
import json
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
CONFIG = {
    "results_file": "results/latency_recall/sweep_results.json",
    "output_plot": "results/latency_recall/latency_recall_plot.png",
    "title": "HNSW Trade-off: Latency vs. Recall",
    "x_label": "Recall@10",
    "y_label": "Latency per Query (ms)"
}
# =================================================

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Results file not found at {filepath}")
        print("Did you run 'run_sweep.py' first?")
        exit(1)
        
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_tradeoff(data):
    # Extract points
    # Sort by recall to ensure the line connects correctly
    # (HNSW behavior usually implies higher efSearch = higher recall)
    data.sort(key=lambda x: x["recall"])

    recalls = [d["recall"] for d in data]
    latencies = [d["latency_ms"] for d in data]
    ef_values = [d["efSearch"] for d in data]

    # Plot Setup
    plt.figure(figsize=(10, 6))
    
    # 1. Main Line
    plt.plot(recalls, latencies, marker='o', linestyle='-', linewidth=2, color='#1f77b4', label='HNSW Index')

    # 2. Annotate specific points (efSearch values)
    # We annotate every other point to avoid clutter if there are many
    for i, ef in enumerate(ef_values):
        # Annotate specific key points (start, end, and some middle ones)
        if i == 0 or i == len(ef_values)-1 or i % 2 == 0:
            plt.annotate(f"ef={ef}", 
                         (recalls[i], latencies[i]), 
                         textcoords="offset points", 
                         xytext=(0, 10), 
                         ha='center',
                         fontsize=9,
                         fontweight='bold',
                         alpha=0.7)

    # 3. Styling
    plt.xlabel(CONFIG["x_label"], fontsize=12)
    plt.ylabel(CONFIG["y_label"], fontsize=12)
    plt.title(CONFIG["title"], fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Optional: Highlight the "Pareto Frontier" feel (Top Right is usually bad)
    # Usually we want High Recall (Right) and Low Latency (Down) -> Bottom Right is ideal
    
    plt.tight_layout()
    
    # 4. Save
    os.makedirs(os.path.dirname(CONFIG["output_plot"]), exist_ok=True)
    plt.savefig(CONFIG["output_plot"], dpi=300)
    print(f"Plot saved to: {CONFIG["output_plot"]}")
    plt.show()

if __name__ == "__main__":
    print("Generating Plot...")
    data = load_data(CONFIG["results_file"])
    plot_tradeoff(data)