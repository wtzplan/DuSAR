import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Pre-load the embedding model
current_path = os.path.realpath(__file__)
path= os.path.dirname(current_path)
# print(path)
device = torch.device("cuda:0")
#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
model = SentenceTransformer("/home/axzc/TRAD/Mind2Web/pretreined_model/all-MiniLM-L6-v2")



# Prompt in retrieval for thought
def process_ob(ob):
    if ob.startswith("In the middle of the room"):
        ob = "You are in the middle of the room."
    return ob

# Trajectory-level random retrieval
def retrieve_demonstration_random(expert_demos, args):
    order = np.random.permutation(len(expert_demos))
    retrieved_demos = []
    
    retrieved_demos, len_trajs = [], []
    for idx in order[:args.K]:
        step_idx = len(expert_demos[idx]["traj"]) - 1
        retrieved_demos.append({
            "doc_id": idx,
            "trajectory": expert_demos[idx],
            "step_idx": step_idx,
            "similarity": None
        })
        len_trajs.append(step_idx)
    
    order = np.argsort(len_trajs)
    retrieved_demos = [retrieved_demos[i] for i in order]
    
    return retrieved_demos

# Trajectory-level retrieval with initial observation + task (Synapse)
@torch.no_grad()
def retrieve_demonstration_task_meta(init_obs, target_emb, expert_demos, args):
    # Embed task metadata into dense vectors, compute similarity, and rank
    query_emb = model.encode([init_obs], convert_to_tensor=True, normalize_embeddings=True)
    similarity = torch.matmul(query_emb, target_emb.T).view(-1).cpu().numpy()
    order = np.argsort(-similarity)
    
    retrieved_demos, len_trajs = [], []
    for idx in order[:args.K]:
        step_idx = len(expert_demos[idx]["traj"]) - 1
        retrieved_demos.append({
            "doc_id": idx,
            "trajectory": expert_demos[idx],
            "step_idx": step_idx,
            "similarity": similarity
        })
        len_trajs.append(step_idx)
    
    # Re-rank the retrieved demostrations by its step_idx or length
    order = np.argsort(len_trajs)
    retrieved_demos = [retrieved_demos[i] for i in order]
    
    return retrieved_demos

# Step-level retrieval with thought
@torch.no_grad()
def retrieve_demonstration_thought(thought, target_emb, expert_demos, args):
    # Embed thought into dense vectors, compute similarity, and rank
    query_emb = model.encode([thought], convert_to_tensor=True, normalize_embeddings=True)
    similarity = torch.matmul(query_emb, target_emb.T).view(-1).cpu().numpy()
    order = np.argsort(-similarity)
    
    # Index for concatnated embedding
    raw_traj_idx = np.repeat(np.arange(len(expert_demos)), [len(d["traj"])-1 for d in expert_demos]) # 记录每个步骤所属的轨迹索引。
    raw_step_idx = np.concatenate([np.arange(len(d["traj"])-1) for d in expert_demos], axis=0) # 记录每个步骤在轨迹中的索引位置。

    # Collect trajectory-independent demonstrations
    # traj_collection 用于存储已选择的轨迹索引，列表 retrieved_demos 用于存储检索到的演示，列表 step_indices 用于存储步骤索引
    traj_collection, retrieved_demos, step_indices = set(), [], []
    for idx in order:
    # 遍历相似度排序后的索引 order，选择前 K 个未被选择过的轨迹。
    # 将选中的轨迹及其相关信息添加到 retrieved_demos 列表中。
        traj_idx, step_idx = raw_traj_idx[idx], raw_step_idx[idx]
        if traj_idx in traj_collection:
            continue
        traj_collection.add(traj_idx)
        retrieved_demos.append({
            "doc_id": traj_idx,
            "trajectory": expert_demos[traj_idx],
            "step_idx": step_idx,
            "similarity": similarity
        })
        step_indices.append(step_idx)
        if len(traj_collection) >= args.K:
            break

    # Re-rank the retrieved demostrations by its step_idx or length
    order = np.argsort(step_indices)
    retrieved_demos = [retrieved_demos[i] for i in order]
    return retrieved_demos

# Step-level retrieval with initial observation + task (Synapse) 
@torch.no_grad()
def retrieve_demonstration_task_thought(init_obs, target_emb, expert_demos, args):
    # Embed task metadata into dense vectors, compute similarity, rank, and get step-by-step thought
    query_emb = model.encode([init_obs], convert_to_tensor=True, normalize_embeddings=True)
    similarity = torch.matmul(query_emb, target_emb.T).view(-1).cpu().numpy()
    order = np.argsort(-similarity)

    # Index for concatnated embedding
    raw_traj_idx = np.repeat(np.arange(len(expert_demos)), [len(d["traj"])-1 for d in expert_demos]) # 记录每个步骤所属的轨迹索引。
    raw_step_idx = np.concatenate([np.arange(len(d["traj"])-1) for d in expert_demos], axis=0) # 记录每个步骤在轨迹中的索引位置。

    
    traj_collection, retrieved_demos, step_indices = set(), [], []
    for idx in order:
    # 遍历相似度排序后的索引 order，选择前 K 个未被选择过的轨迹。
    # 将选中的轨迹及其相关信息添加到 retrieved_demos 列表中。
        traj_idx, step_idx = raw_traj_idx[idx], raw_step_idx[idx]
        if traj_idx in traj_collection:
            continue
        traj_collection.add(traj_idx)
        retrieved_demos.append({
            "doc_id": traj_idx,
            "trajectory": expert_demos[traj_idx],
            "step_idx": step_idx,
            "similarity": similarity
        })
        step_indices.append(step_idx)
        if len(traj_collection) >= args.K:
            break

    # Re-rank the retrieved demostrations by its step_idx or length
    order = np.argsort(step_indices)
    retrieved_demos = [retrieved_demos[i] for i in order]
    return retrieved_demos
