import os
import urllib.request
import tarfile
import numpy as np
import requests
import zipfile

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_fvecs(filename, vecs):
    """Converts a numpy array of vectors into the standard .fvecs binary format."""
    print(f"Writing {len(vecs)} vectors to {filename}...")
    vecs = vecs.astype(np.float32)
    _, d = vecs.shape
    with open(filename, "wb") as f:
        for vec in vecs:
            # fvecs format: [dimension (int32)] [vector data (float32)]
            f.write(np.int32(d).tobytes())
            f.write(vec.tobytes())
    print(f"Finished writing {filename}\n")

def download_and_extract_texmex(name, url, output_dir):
    """Downloads and extracts SIFT/GIST datasets which are already .fvecs."""
    tar_path = os.path.join(output_dir, f"{name}.tar.gz")
    if not os.path.exists(tar_path):
        print(f"Downloading {name} from {url}...")
        urllib.request.urlretrieve(url, tar_path)
    
    print(f"Extracting {name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    print(f"{name} extracted successfully.\n")

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"File {dest} already exists. Skipping download.")


def process_fasttext_manual(base_dir):
    """Downloads and parses FastText .vec file without Gensim."""
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip"
    zip_path = os.path.join(base_dir, "fasttext.zip")
    
    download_file(url, zip_path)
    
    fasttext_dir = os.path.join(base_dir, "fasttext")
    ensure_dir(fasttext_dir)

    print(f"Extracting all files from {zip_path} to {fasttext_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(fasttext_dir)

    print("FastText zip extracted successfully.\n")

def process_bert_no_torch(output_path,base_dir):
    """Extracts BERT embeddings using safetensors without requiring PyTorch."""
    print("Extracting BERT embeddings (No-Torch mode)...")
    # Download the small safetensors weight file directly from HuggingFace
    url = "https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors"
    download_file(url, os.path.join(base_dir, "bert.safetensors"))
    sf_path = os.path.join(base_dir, "bert.safetensors")
    # Load the weights directly into a dictionary of numpy arrays
    from safetensors import safe_open
    with safe_open(sf_path, framework="np", device="cpu") as f:
        # The word embeddings are stored under this specific key in BERT
        weights = f.get_tensor("bert.embeddings.word_embeddings.weight")
        write_fvecs(output_path, weights)

def main():
    base_dir = "/run/media/lennart/SANDISK/data"
    ensure_dir(base_dir)

    # 1. SIFT 1M (Files are hosted on IRISA FTP server)
    sift_url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    # download_and_extract_texmex("sift", sift_url, base_dir)

    # 2. GIST 1M (Files are hosted on IRISA FTP server)
    gist_url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    # download_and_extract_texmex("gist", gist_url, base_dir)

    # 3. FastText 1M
    # process_fasttext_manual(base_dir)

    # 4. BERT 30522
    process_bert_no_torch(os.path.join(base_dir, "bert", "bert_30522.fvecs"), base_dir)

    # 5. DEEP 10M Note
    print("="*50)
    print("NOTE REGARDING DEEP (10M):")
    print("The PKU disk link (https://disk.pku.edu.cn/...) requires a browser ")
    print("environment with JavaScript and session cookies to bypass security checks.")
    print("You will need to manually download the DEEP dataset locally, upload it to ")
    print("this server, and extract it.")
    print("="*50)

if __name__ == "__main__":
    main()