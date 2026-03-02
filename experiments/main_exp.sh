# Configuration
K_VALUE=10
OUTPUT_FILE="experiment_results.csv"

# Write the CSV headers
echo "target_alpha,found_lambda,time_seconds,avg_ef_search,empirical_fnr" > "$OUTPUT_FILE"

# Define the alpha values to loop through
ALPHAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for alpha in "${ALPHAS[@]}"; do
    echo "Running experiment with alpha = $alpha..."
    
    # Run the program. Standard Error (logs) will print to the terminal.
    # Standard Output (the CSV line) gets captured in the 'result' variable.
    result=$(./build/conHNSW $K_VALUE $alpha)
    
    # Append the result to our CSV file
    echo "$result" >> "$OUTPUT_FILE"
done

echo "All experiments finished! Data saved to $OUTPUT_FILE."