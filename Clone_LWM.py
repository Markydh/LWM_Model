import subprocess
import os

# Step 1: Clone the model repository (if not already cloned)
model_repo_url = "git@hf.co:wi-lab/lwm"

model_repo_dir = "./LWM"

if not os.path.exists(model_repo_dir):
    print(f"Cloning model repository from {model_repo_url}...")
    subprocess.run(["git", "clone", model_repo_url, model_repo_dir], check=True)


