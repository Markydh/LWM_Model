import subprocess
import os
import numpy as np
from FirstStep import clone_dataset_scenarios

# 用于克隆特定的数据集场景文件夹
def clone_dataset_scenario(scenario_name, repo_url, model_repo_dir="./LWM", scenarios_dir="scenarios_test"):
    current_dir = os.path.basename(os.getcwd())
    if current_dir == "LWM":
        model_repo_dir = "."

    # Create the scenarios_test directory if it doesn't exist
    scenarios_path = os.path.join(model_repo_dir, scenarios_dir)
    if not os.path.exists(scenarios_path):
        os.makedirs(scenarios_path)

    scenario_path = os.path.join(scenarios_path, scenario_name)

    # Initialize sparse checkout for the dataset repository
    if not os.path.exists(os.path.join(scenarios_path, ".git")):
        print(f"Initializing sparse checkout in {scenarios_path}...")
        subprocess.run(["git", "clone", "--sparse", repo_url, "."], cwd=scenarios_path, check=True)
        subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=scenarios_path, check=True)
        subprocess.run(["git", "lfs", "install"], cwd=scenarios_path, check=True)  # Install Git LFS if needed

    # Add the requested scenario folder to sparse checkout
    print(f"Adding {scenario_name} to sparse checkout...")
    subprocess.run(["git", "sparse-checkout", "add", scenario_name], cwd=scenarios_path, check=True)
    
    # Pull large files if needed (using Git LFS)
    subprocess.run(["git", "lfs", "pull"], cwd=scenarios_path, check=True)

    print(f"Successfully cloned {scenario_name} into {scenarios_path}.")

def clone_dataset_scenarios(selected_scenario_names, dataset_repo_url, model_repo_dir="./LWM"):
    for scenario_name in selected_scenario_names:
        clone_dataset_scenario(scenario_name, dataset_repo_url, model_repo_dir)
    
    
dataset_repo_url = "https://huggingface.co/datasets/wi-lab/lwm"  # Base URL for dataset repo
scenario_names = np.array([
    "city_18_denver", "city_15_indianapolis", "city_19_oklahoma", 
    "city_12_fortworth", "city_11_santaclara", "city_7_sandiego"
])

scenario_idxs = np.array([0, 1, 2, 3, 4, 5])  # Select the scenario indexes
selected_scenario_names = scenario_names[scenario_idxs]

# Clone the requested scenarios_test
clone_dataset_scenarios(selected_scenario_names, dataset_repo_url)