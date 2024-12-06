from input_preprocess import tokenizer
from lwm_model import lwm
import torch
import numpy as np

scenario_names = np.array([
    "city_18_denver", "city_15_indianapolis", "city_19_oklahoma",
    "city_12_fortworth", "city_11_santaclara", "city_7_sandiego"
])

scenario_idxs = np.array([0, 1, 2, 3, 4, 5])  # Select the scenario indexes
selected_scenario_names = scenario_names[scenario_idxs]

# 对数据进行标记
preprocessed_chs = tokenizer(
    selected_scenario_names=selected_scenario_names,  # Selects predefined DeepMIMOv3 scenarios_test. Set to None to load your own dataset.
    manual_data=None,  # If using a custom dataset, ensure it is a wireless channel dataset of size (N,32,32) based on the settings provided above.
    gen_raw=True  # Set gen_raw=False to apply masked channel modeling (MCM), as used in LWM pre-training. For inference, masking is unnecessary unless you want to evaluate LWM's ability to handle noisy inputs.
)

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading the LWM model on {device}...")
model = lwm.from_pretrained(device=device)


# 进行推理
from inference import lwm_inference, create_raw_dataset
input_types = ['cls_emb', 'channel_emb', 'raw']
selected_input_type = input_types[0]  # Change the index to select LWM CLS embeddings, LWM channel embeddings, or the original input channels.

if selected_input_type in ['cls_emb', 'channel_emb']:
    dataset = lwm_inference(preprocessed_chs, selected_input_type, model, device)
else:
    dataset = create_raw_dataset(preprocessed_chs, device)



# # 生成标签
# from input_preprocess import create_labels
# tasks = ['LoS/NLoS Classification', 'Beam Prediction']
# task = tasks[1] # Choose 0 for LoS/NLoS labels or 1 for beam prediction labels.
# labels = create_labels(task, selected_scenario_names, n_beams=64) 
