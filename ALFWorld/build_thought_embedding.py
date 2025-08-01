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

thoughts = {
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
    for step in traj[:-1]:
        thoughts[idx].append(step["thought"])
 
for i in trange(6):
    emb = model.encode(
        thoughts[i], batch_size=512, convert_to_numpy=True, normalize_embeddings=True)
    pathn = path+f"/data/{i}_thought_embedding.npy"
    np.save(pathn, emb)
    # np.save(f"data/{i}_thought_embedding.npy", emb)