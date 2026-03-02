import argparse
import csv
import pathlib
import numpy as np
import matplotlib.pyplot as plt

def read_vecs(path: pathlib.Path, dtype: np.dtype) -> np.ndarray:
    """Reads both fvecs and ivecs formats."""
    data = np.fromfile(path, dtype=np.int32)
    if data.size == 0:
        raise ValueError(f"Empty file: {path}")

    d = int(data[0])
    row_size = d + 1
    if data.size % row_size != 0:
        raise ValueError(f"Malformed file: {path}")

    raw = data.reshape(-1, row_size)
    if not np.all(raw[:, 0] == d):
        raise ValueError(f"Inconsistent dimensions in file: {path}")

    # Return the payload cast to the requested dtype
    return raw[:, 1:].view(dtype) if dtype == np.float32 else raw[:, 1:]

def read_query_ids(csv_path: pathlib.Path) -> np.ndarray:
    """Extracts unique query IDs from the first column of a CSV."""
    ids = []
    with csv_path.open("r", newline="") as f:
        for row in csv.reader(f):
            if row and row[0].strip() and row[0].strip().lstrip("-").isdigit():
                ids.append(int(row[0].strip()))
    
    if not ids:
        raise ValueError(f"No valid query IDs found in {csv_path}")
    
    return np.array(sorted(set(ids)), dtype=np.int64)

def compute_distances(query_ids: np.ndarray, queries: np.ndarray, base: np.ndarray, gt: np.ndarray, topk: int = 10):
    """Computes squared L2 distances for the specified top-k groundtruth neighbors."""
    q = queries[query_ids]
    neighbors = base[gt[query_ids, :topk]]
    
    # Calculate squared L2 distance: sum((neighbor_vec - query_vec)^2)
    dists = np.sum((neighbors - q[:, None, :]) ** 2, axis=2)
    
    avg_topk = np.mean(dists, axis=1).astype(np.float64)
    dist_10th = dists[:, topk - 1].astype(np.float64)
    return avg_topk, dist_10th

def analyze_difference(x: np.ndarray, y: np.ndarray, n_iter: int = 5000, seed: int = 123) -> dict:
    """Performs both a permutation test and bootstrap CI for the difference in means (x - y)."""
    rng = np.random.default_rng(seed)
    obs_diff = float(np.mean(x) - np.mean(y))

    # Permutation Test
    combined = np.concatenate([x, y])
    n_x = len(x)
    extreme_count = 0
    for _ in range(n_iter):
        perm = rng.permutation(combined)
        if np.mean(perm[:n_x]) - np.mean(perm[n_x:]) >= obs_diff:
            extreme_count += 1
    p_val = (extreme_count + 1) / (n_iter + 1)

    # Bootstrap 95% Confidence Interval
    boot_diffs = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        boot_diffs[i] = np.mean(xb) - np.mean(yb)
    
    ci_low, ci_high = np.quantile(boot_diffs, [0.025, 0.975])
    
    return {"diff": obs_diff, "p_val": p_val, "ci_low": ci_low, "ci_high": ci_high}

def plot_10th_distances(in_10th: np.ndarray, out_10th: np.ndarray, output_path: pathlib.Path) -> None:
    """Saves a single histogram comparing the 10th neighbor distances."""
    plt.figure(figsize=(6, 5))
    plt.hist(out_10th, bins=80, alpha=0.6, label="Lambda Satisfied")
    plt.hist(in_10th, bins=80, alpha=0.6, label="Lambda Not Satisfied")
    plt.title("Per-query 10th Groundtruth Distance")
    plt.xlabel("Squared L2 Distance")
    plt.ylabel("Count")
    plt.legend(fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute distance stats vs groundtruth.")
    parser.add_argument("--ids-csv", type=pathlib.Path, default="remaining_active_indices.csv")
    parser.add_argument("--base-fvecs", type=pathlib.Path, default="../sift1M/sift_base.fvecs")
    parser.add_argument("--query-fvecs", type=pathlib.Path, default="../sift1M/sift_query.fvecs")
    parser.add_argument("--gt-ivecs", type=pathlib.Path, default="../sift1M/sift_groundtruth.ivecs")
    parser.add_argument("--range-end", type=int, default=10000)
    parser.add_argument("--plot-out", type=pathlib.Path, default="remaining_vs_nonremaining_distances_0.6.png")
    parser.add_argument("--n-iter", type=int, default=5000, help="Iterations for bootstrap and perm tests")
    args = parser.parse_args()

    # Load data
    ids = read_query_ids(args.ids_csv)
    base = read_vecs(args.base_fvecs, np.float32)
    queries = read_vecs(args.query_fvecs, np.float32)
    gt = read_vecs(args.gt_ivecs, np.int32)

    # Validate limits
    if args.range_end > queries.shape[0] or args.range_end > gt.shape[0]:
        raise ValueError("range-end exceeds available queries or groundtruth rows.")

    # Split indices
    all_ids = np.arange(args.range_end, dtype=np.int64)
    in_mask = np.isin(all_ids, ids)
    in_ids = all_ids[in_mask]
    out_ids = all_ids[~in_mask]

    # Compute distances
    in_avg_top10, in_10th = compute_distances(in_ids, queries, base, gt)
    out_avg_top10, out_10th = compute_distances(out_ids, queries, base, gt)

    # Statistical analysis
    stats_avg = analyze_difference(in_avg_top10, out_avg_top10, n_iter=args.n_iter, seed=123)
    stats_10th = analyze_difference(in_10th, out_10th, n_iter=args.n_iter, seed=124)

    # Generate the single plot
    plot_10th_distances(in_10th, out_10th, args.plot_out)

    # Output results
    print("Distances are squared L2.")
    print(f"LAMBDA_NOT_SATISFIED | count={len(in_ids)}, avg_top10={np.mean(in_avg_top10):.6f}, avg_10th={np.mean(in_10th):.6f}")
    print(f"LAMBDA_SATISFIED     | count={len(out_ids)}, avg_top10={np.mean(out_avg_top10):.6f}, avg_10th={np.mean(out_10th):.6f}")
    
    print("\nPERM_TEST(mean IN_CSV > NOT_IN_CSV, avg_top10):")
    print(f"  diff={stats_avg['diff']:.6f}, p_value={stats_avg['p_val']:.8f}, boot95CI=[{stats_avg['ci_low']:.6f}, {stats_avg['ci_high']:.6f}]")
    
    print("PERM_TEST(mean IN_CSV > NOT_IN_CSV, dist_10th):")
    print(f"  diff={stats_10th['diff']:.6f}, p_value={stats_10th['p_val']:.8f}, boot95CI=[{stats_10th['ci_low']:.6f}, {stats_10th['ci_high']:.6f}]")
    
    print(f"\nPlot written to: {args.plot_out}")

if __name__ == "__main__":
    main()