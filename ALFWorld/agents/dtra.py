from utils import load_data
from collections import deque
import json
from typing import Dict, List, Any
import random
import time
import torch
from agents.llm import (
    generate_response
)
from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda:0")
#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
model = SentenceTransformer("/home/axzc/TRAD/Mind2Web/pretreined_model/all-MiniLM-L6-v2")


class StrategyManager:
    def __init__(self, args):
        self.strategy_history = []
        self.stey_by_step_history =[]
        self.evaluation_threshold = 0.3
        self.success_count = 0
        self.total_attempts = 0
        self.args = args
        self.model_chat = args.model

    def update_holistic_strategy(self, current_strategy: str, evaluation_result: float, 
                               environment_feedback: str, holistic_prompt: str, analysis: str, iniobs:str) -> str:
        if evaluation_result >= self.evaluation_threshold:
            update_prompt = f"""
            Your previous strategy is: {current_strategy}.
            Here is your analysis trace is : {environment_feedback}.
            And the recent status analysis is : {analysis}
            Your initial environment and task is: " {iniobs}.
            
            Please give me your new strategy to slove your task, which should be concise and clear, about 100 words.
            The example is: 
            {holistic_prompt}
            """

            message = [ 
                    {   "role": "system",
                        "content": "You are a strategy optimization assistant. Help improve the current strategy."
                    },

                    {
                        "role": "user",
                        "content": update_prompt
                    }
            ]
            new_strategy, info = generate_response(
                messages=message,
                model=self.model_chat,
                args=self.args
            )
            self.strategy_history.append({
                'old_strategy': current_strategy,
                'new_strategy': new_strategy,
                'evaluation': evaluation_result,
                'observation_changes': environment_feedback,
                'timestamp': time.time()
            })
            
            return new_strategy, info
        return current_strategy, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def format_action(self, ob):
        flattened_list = [item for sublist in ob for item in sublist]
        if len(flattened_list) > 1:
            result = ', '.join(flattened_list[:-1]) + ', and ' + flattened_list[-1] +".\n"
        else:
            result = flattened_list[0] + ".\n"
        return result
    
    def semantic_match(self, user_text, action_space):
        embed_actions = model.encode(action_space, convert_to_tensor=True)
        emb_user = model.encode(user_text, convert_to_tensor=True)
        scores = util.cos_sim(emb_user, embed_actions)[0]
        best_idx = scores.argmax()
        return action_space[best_idx]
