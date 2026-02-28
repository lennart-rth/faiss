import pandas as pd
import matplotlib.pyplot as plt

# Load data (assuming your script output to resume_benchmark.csv)
df = pd.read_csv("../experiments/resume_benchmark.csv")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary Y-Axis: Search Times
ax1.set_xlabel('Initial Search Depth (ef1)', fontsize=12)
ax1.set_ylabel('Search Time (seconds)', color='black', fontsize=12)
line1 = ax1.plot(df['ef1'], df['time_normal'], label='Normal Search Time', color='crimson', linestyle='--', linewidth=2)
line2 = ax1.plot(df['ef1'], df['time_resume'], label='Resumed Search Time', color='royalblue', marker='o', linewidth=2)
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.set_ylim(0, df['time_normal'].max() + 0.5)

# Secondary Y-Axis: Memory Overhead
ax2 = ax1.twinx()  
ax2.set_ylabel('Cache Memory Overhead (MB)', color='purple', fontsize=12)  
line3 = ax2.plot(df['ef1'], df['memory_resume_mb'], label='Resume Cache RAM (MB)', color='purple', marker='s', linestyle='-', linewidth=2)
ax2.tick_params(axis='y', labelcolor='purple')

# Add legends
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left')

plt.title('HNSW Space/Time Tradeoff: Resumed vs Normal (Target ef2=700)', fontsize=14, fontweight='bold')
fig.tight_layout()

plt.savefig("hnsw_space_time_tradeoff.png", dpi=300)
plt.show()