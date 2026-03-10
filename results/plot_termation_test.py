import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_frontier(csv_path, output_image):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    recall_col = [col for col in df.columns if 'recall' in col.lower()][0]

    # 1. Use a clean white background with a dense, subtle grid
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(True, which="both", ls="--", alpha=0.5)

    methods = df['method'].unique()
    
    # 2. Hardcode high-contrast colors, distinct styles, and shapes
    # Red, Blue, Green, Orange, Purple
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, method in enumerate(methods):
        subset = df[df['method'] == method].copy()
        subset = subset.sort_values(by=recall_col)
        
        ax.plot(
            subset[recall_col], 
            subset['latency_ms'], 
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)], 
            linewidth=2.5,          # Thicker lines
            markersize=9,           # Larger markers
            markeredgecolor='white', # White borders around markers prevent blending
            markeredgewidth=1.5,
            label=method
        )

    # 3. Axis Formatting
    ax.set_title('HNSW Early Termination: Latency-Recall Frontier', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(f'{recall_col.capitalize()}', fontsize=14, labelpad=10)
    ax.set_ylabel('Latency per Query (ms)', fontsize=14, labelpad=10)
    
    # TIP: If your lines are overlapping at the top, un-comment the next line to spread them out vertically!
    ax.set_yscale('log')

    # TIP: If all methods achieve >90% recall, zoom in on the right side of the graph
    # min_recall = df[recall_col].min()
    # ax.set_xlim(left=max(0.5, min_recall - 0.05), right=1.01)

    # 4. Legend styling
    ax.legend(
        title='Termination Method', 
        title_fontsize='13', 
        fontsize='12', 
        loc='upper left', 
        frameon=True, 
        shadow=True
    )
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"High-contrast plot successfully saved to {output_image}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HNSW Latency-Recall Frontier")
    parser.add_argument("--csv", type=str, default="../experiments/build/frontier_results.csv", help="Path to the benchmark CSV file")
    parser.add_argument("--out", type=str, default="frontier_plot.png", help="Output path for the saved image")
    
    args = parser.parse_args()
    plot_frontier(args.csv, args.out)