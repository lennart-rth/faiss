CSV_FILE="resume_benchmark.csv"
echo "ef1,ef2,time_normal,time_resume,overlap_pct,memory_normal_mb,memory_resume_mb,normal_recall,resume_recall_ef1,resume_recall_ef2" > $CSV_FILE

EF2=700
echo "Starting benchmarks. Target ef2 is fixed at $EF2."
echo "Results will be appended to $CSV_FILE"
echo "--------------------------------------------------------"

# 3. Loop through ef1 values (from 10 to 690 in steps of 50)
for EF1 in $(seq 10 68 690); do
    echo "Testing ef1 = $EF1 ..."
    
    ./build/hnsw_cached_benchmarks $EF1 $EF2 >> $CSV_FILE
done

echo "--------------------------------------------------------"
echo "Done! Check $CSV_FILE for your plot data."