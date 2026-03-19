# --- CONFIGURATION ---
USB_PATH="/run/media/lennart/SANDISK/data"
# Points to the build folder relative to where this script is (experiments/)
SAMPLE_BIN="./build/sample"
GT_BIN="./build/ground_truth"

# 1. Safety Check: Do the binaries exist?
if [ ! -f "$SAMPLE_BIN" ] || [ ! -f "$GT_BIN" ]; then
    echo "Error: Binaries not found in ./build/"
    echo "Make sure you ran 'make' inside the build folder first."
    exit 1
fi

# 2. Safety Check: Is the USB mounted?
if [ ! -d "$USB_PATH" ]; then
    echo "Error: USB path $USB_PATH not found."
    exit 1
fi

# Folder|BaseFile|QueryFile (Empty query if only 1 file)
datasets=(
    # "sift|sift/sift_base.fvecs|sift/sift_query.fvecs"
    # "gist|gist/gist_base.fvecs|gist/gist_query.fvecs"
    "fasttext|fasttext/wiki-news-300d-1M-subword.vec.vec|"
    "bert|bert/bert_30522.fvecs|"
)

for entry in "${datasets[@]}"; do
    IFS="|" read -r dir base query <<< "$entry"
    
    echo "================================================"
    echo "PROCESSING: $dir"
    echo "================================================"
    
    # We stay in experiments/ but use absolute or relative paths for the USB
    # This keeps our binary paths (./build/...) valid.
    
    IN_BASE="$USB_PATH/$base"
    IN_QUERY=""
    if [ -n "$query" ]; then IN_QUERY="$USB_PATH/$query"; fi
    
    # Define outputs on the USB
    OUT_BASE="$USB_PATH/sampled/${dir}/${dir}_base.fvecs"
    OUT_QUERY="$USB_PATH/sampled/${dir}/${dir}_query.fvecs"
    OUT_CALIB="$USB_PATH/sampled/${dir}/${dir}_calib.fvecs"
    OUT_RAPS="$USB_PATH/sampled/${dir}/${dir}_raps.fvecs"

    # 1. Run Sampling
    if [ -n "$IN_QUERY" ] && [ -f "$IN_QUERY" ]; then
        echo "Merging and sampling $dir..."
        "$SAMPLE_BIN" "$IN_BASE" "$IN_QUERY" \
            "$OUT_BASE" 0.75 "$OUT_QUERY" 0.1 "$OUT_CALIB" 0.1 "$OUT_RAPS" 0.05
    else
        echo "Sampling single file $dir..."
        "$SAMPLE_BIN" "$IN_BASE" \
            "$OUT_BASE" 0.75 "$OUT_QUERY" 0.1 "$OUT_CALIB" 0.1 "$OUT_RAPS" 0.05
    fi

    # 2. Compute Ground Truths (Top 100 neighbors)
    for split in "query" "calib" "raps"; do
        CUR_FILE="$USB_PATH/sampled/${dir}/${dir}_${split}.fvecs"
        GT_FILE="$USB_PATH/sampled/${dir}/${dir}_${split}_gt.ivecs"

        echo "Computing Ground Truth: $split vs Base..."
        "$GT_BIN" "$CUR_FILE" "$OUT_BASE" "$GT_FILE" 100
    done

    echo "Finished $dir successfully."
done