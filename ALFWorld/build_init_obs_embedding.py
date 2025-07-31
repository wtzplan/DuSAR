import os
import json
import torch
import numpy as np
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer
current_path = os.path.realpath(__file__)
path= os.path.dirname(current_path)
# print(path)
device = torch.device("cuda:0")
#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
model = SentenceTransformer(path+"/../../src/all-MiniLM-L6-v2").to(device)

os.makedirs(path+"/data/", exist_ok=True)
traj_files = os.listdir(path+"/refined_data")
obs = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
}
for filename in tqdm(traj_files):
    idx = int(filename[0])
    json_path = os.path.join(path+"/refined_data", filename)
    with open(json_path, "r") as f:
        traj_dict = json.load(f)
    task, traj = traj_dict["task"], traj_dict["traj"]
    init_ob = traj[0]["obs"] + '\nYour task is to: ' + task
    obs[idx].append(init_ob)

for i in trange(6):
    emb = model.encode(
        obs[i], batch_size=512, convert_to_numpy=True, normalize_embeddings=True)
    pathn = path+f"/data/{i}_init_obs_embedding.npy"
    np.save(pathn, emb)

# mkdir /pretrained_model/all-MiniLM-L6-v2
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-tranls
# sformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main_config.json
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/train_script.py
# wget -P /pretrained_model/all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt
# mkdir /pretrained_model/all-MiniLM-L6-v2/1_Pooling
# wget -P /pretrained_model/all-MiniLM-L6-v2/1_Pooling https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json
