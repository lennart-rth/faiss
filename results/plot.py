import os
import json
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict

# ================= CONFIGURATION =================
RESULTS_DIR = "latency_recall"
OUTPUT_FILENAME = "latency_recall_patience_pivot.png"
# =================================================

def load_data(results_dir):
    """
    Reads all .json files.
    Returns a list of dictionaries: 
    [ { "name": "NaiveES", "patience": 10, "data": [...] }, ... ]
    """
    loaded_files = []
    
    # Find all json files
    json_pattern = os.path.join(results_dir, "*.json")
    files = glob.glob(json_pattern)
    
    if not files:
        print(f"No JSON files found in {results_dir}")
        return []

    for filepath in files:
        filename = os.path.basename(filepath)
        name_no_ext = os.path.splitext(filename)[0]
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list) and len(data) > 0:
                # Determine experiment type and parameters from filename
                patience = None
                exp_type = name_no_ext
                
                # Check if this is a NaiveES file and extract patience
                # Matches "NaiveES-10", "NaiveES-50", etc.
                match = re.match(r"NaiveES-(\d+)", name_no_ext, re.IGNORECASE)
                if match:
                    exp_type = "NaiveES"
                    patience = int(match.group(1))
                elif name_no_ext.lower() == "hnsw":
                    exp_type = "HNSW"

                loaded_files.append({
                    "type": exp_type,
                    "patience": patience, # Will be None for HNSW
                    "original_filename": name_no_ext,
                    "data": data
                })
            else:
                print(f"Skipping {filename}: Invalid format or empty.")
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return loaded_files

def get_ef_value(point):
    """Helper to find the ef/ef_search key in the json point."""
    for key in ['ef', 'ef_search', 'efSearch', 'search_k']:
        if key in point:
            return point[key]
    return None

def process_data(loaded_files):
    """
    Pivots the data.
    1. HNSW is kept as a simple curve (varying EF).
    2. NaiveES is grouped by EF, so we can plot Patience on the curve.
    """
    processed_curves = {}

    # 1. Handle NaiveES: Group by EF value across all files
    naive_by_ef = defaultdict(list)
    
    # 2. Handle HNSW: Keep as is
    hnsw_data = []

    for file_info in loaded_files:
        if file_info['type'] == 'HNSW':
            # Just sort HNSW by recall (standard ROC-like curve)
            sorted_data = sorted(file_info['data'], key=lambda x: x.get('recall', 0))
            processed_curves["HNSW (Baseline)"] = sorted_data
            
        elif file_info['type'] == 'NaiveES':
            patience = file_info['patience']
            
            # Go through every point in this file (which varies EF)
            # and distribute them into buckets based on their EF value
            for point in file_info['data']:
                ef_val = get_ef_value(point)
                
                if ef_val is not None:
                    # We store the patience with the point so we can sort by it later
                    point_with_meta = point.copy()
                    point_with_meta['patience_param'] = patience
                    naive_by_ef[ef_val].append(point_with_meta)

    # 3. Finalize NaiveES curves: Sort each EF bucket by Patience
    for ef_val, points in naive_by_ef.items():
        # Sort by patience so the line connects increasing patience values
        points.sort(key=lambda x: x['patience_param'])
        label = f"NaiveES (Fixed ef={ef_val})"
        processed_curves[label] = points

    return processed_curves

def plot_comparison(curves):
    plt.figure(figsize=(12, 8))
    
    # Define markers and colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    # Use a colormap to ensure distinct colors if we have many EF values
    colormap = plt.cm.tab10 
    
    # Sort keys to make legend consistent (Put HNSW first, then NaiveES sorted by EF)
    def sort_key(k):
        if "HNSW" in k: return (0, 0)
        # Extract the ef number for sorting
        try:
            num = int(re.search(r"ef=(\d+)", k).group(1))
            return (1, num)
        except:
            return (1, 9999)

    sorted_keys = sorted(curves.keys(), key=sort_key)
    
    for i, label in enumerate(sorted_keys):
        points = curves[label]
        
        recalls = [p["recall"] for p in points]
        latencies = [p["latency_ms"] for p in points]
        
        # Style logic
        if "HNSW" in label:
            # Make HNSW stand out (thick black or distinct line)
            color = 'black'
            linestyle = '--'
            marker = 'x'
            lw = 3
        else:
            color = colormap(i % 10)
            linestyle = '-'
            marker = markers[i % len(markers)]
            lw = 2
        
        plt.plot(recalls, latencies, marker=marker, markersize=6, 
                 linewidth=lw, label=label, color=color, alpha=0.8)

    # Styling
    plt.xlabel("Recall@10", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.title("HNSW vs. NaiveES\n(NaiveES curves vary Patience for fixed EF)", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    
    plt.legend(title="Experiment Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    print(f"Reading data from {RESULTS_DIR}...")
    raw_files = load_data(RESULTS_DIR)
    
    if raw_files:
        print("Processing and pivoting data...")
        curves = process_data(raw_files)
        
        if curves:
            print(f"Found {len(curves)} distinct curves to plot.")
            plot_comparison(curves)
        else:
            print("No valid curves generated after processing.")
    else:
        print("No valid data found to plot.")