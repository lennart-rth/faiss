# 1. Enter environment
nix-shell

# 2. Configure (Disable Python, Enable Optimization)
cmake -B build . \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_OPT_LEVEL=avx2 \
    -DCMAKE_BUILD_TYPE=Release

# 3. Build the library (libfaiss.a)
make -C build -j$(nproc) faiss




--- 
# Build Experiment Scripts

cd experiments/build
cmake ..
make -j$(nproc)
cd ../..