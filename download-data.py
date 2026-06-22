from huggingface_hub import hf_hub_download
import os

os.makedirs("./data/SteerEval/personality", exist_ok=True)

# Download train.json
hf_hub_download(
    repo_id="zjunlp/SteerEval",
    filename="personality/train.json",
    local_dir="./data/SteerEval/personality"
)

# Download valid.json
hf_hub_download(
    repo_id="zjunlp/SteerEval",
    filename="personality/valid.json",
    local_dir="./data/SteerEval/personality"
)

print("Download complete!")