import time
import logging
from bs4 import BeautifulSoup
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.llm import (
    generate_response,
    num_tokens_from_messages,
    get_mode,
    MAX_TOKENS,
    extract_from_response,
)

class StrategyManager:
    def __init__(self, args, model):
        self.strategy_history = []
        self.stey_by_step_history =[]
        self.evaluation_threshold = 0.3
        self.success_count = 0
        self.total_attempts = 0
        self.args = args
        self.model = model

    def format_action(self, all_candidates):
        """
        1. `CLICK [id]`: Click on an HTML element with its id.
        2. `TYPE [id] [value]`: Type a string into the element with the id.
        3. `SELECT [id] [value]`: Select a value for an HTML element by its id.
        """
        actions = []
        for cid in all_candidates:
            actions.append(f'CLICK [{cid}]')
            actions.append(f'TYPE [{cid}] [value]')
            actions.append(f'SELECT [{cid}] [value]')

        # Concatenate into a natural language style string
        if len(actions) > 1:
            result = ', '.join(actions[:-1]) + ', and ' + actions[-1] + ".\n"
        else:
            result = actions[0] + ".\n"

        return result+ "You need to replace the [value] with a [string] without affecting the format.\n"

    def generate_actions(self, html: str, all_candidates:list) -> List[str]:
        supported_tags = {
            'a': ['CLICK'],
            'button': ['CLICK'],
            'input': {
                'text': ['TYPE'],
                'password': ['TYPE'],
                'email': ['TYPE'],
                None: ['TYPE'] 
            },
            'textarea': ['TYPE'],
            'select': ['SELECT'],
            'option': ['SELECT'],
            'label': ['CLICK'],
            # 可选：将 div/span 设为不可操作
            'div': [],  # 不推荐点击
            'span': [],
        }

        soup = BeautifulSoup(html, "html.parser")
        actions = []
        have_Type = False

        for element in soup.find_all(True):  # 遍历所有标签
            tag_name = element.name.lower()
            element_id = element.get('id')

            if not element_id:
                continue  # 忽略没有 id 的元素

            if tag_name not in supported_tags:
                allowed_actions = ['CLICK']
                #continue
            else:
                allowed_actions = supported_tags[tag_name]

            # 处理 input 的不同 type
            if tag_name == 'input':
                input_type = element.get('type')
                if isinstance(allowed_actions, dict):
                    allowed_actions = allowed_actions.get(input_type, [])

            if isinstance(allowed_actions, list):
                for action in allowed_actions:
                    if action == 'CLICK':
                        actions.append(f"CLICK [{element_id}]")
                    elif action == 'TYPE':
                        have_Type = True
                        actions.append(f"TYPE [{element_id}] [value]")
                    elif action == 'SELECT':
                        actions.append(f"SELECT [{element_id}] [value]")

        note = """''TYPE [id] [value]' is to type a string into the element with the id. You need to replace the value with a string, such as' TYPE [id] [string] ', without affecting the format.\n"""

        if len(actions) > 1:
            result = ', '.join(actions[:-1]) + ', and ' + actions[-1] + ".\n"
        elif len(actions)>0:
            result = actions[0] + ".\n"
        else:
            print(html)
            print(all_candidates)
            result = self.format_action(all_candidates)
        if have_Type:
            return result + "Where, "+note+"\n"
        else:
            return result


    def compare_similarity(self, text1, text2):
        if text1==None:
            text1 = " "
        if text2==None:
            text2 = " "
        tfidf = TfidfVectorizer().fit_transform([text1, text2])
        return cosine_similarity(tfidf[0], tfidf[1])[0][0]

    def feedback(self, pred_op, target_op, pred_val, target_val, pred_id, idflag):
        try:
            result = ""
            op_score = self.compare_similarity(pred_op, target_op)
            if target_val!=None:
                if pred_val!=None:
                    val_score = self.compare_similarity(pred_val, target_val)
                else:
                    val_score = self.compare_similarity(" ", target_val)
            elif target_val==None and pred_val!=None:
                result += f"You don't have to type {pred_val} for action {pred_op}."
                val_score = 0
            else:
                val_score = 1
            if idflag:
                id_score = 1
            else:
                id_score = self.compare_similarity(str(pred_id), str(idflag))
            
            if op_score==0 and val_score==0 and id_score==0:
                result = "The output of your action is illegal, please reselect your action."
                #return 0,result
            if op_score<1:
                result += f"The '{pred_op}' you selected is invalid.\n"
            else:
                result += f"The '{pred_op}' you selected is correct.\n"
            if val_score<1:
                result += f"The '{pred_val}' you entered is invalid. Note that you must include the string with [] and nothing else.\n"
            else:
                result += f"The '{pred_val}' you entered is correct.\n"
            if id_score<1:
                result += f"The element '{pred_id}' you selected is invalid. Please change the element id. Note that you must use [] to include only the id and nothing else.\n"
            else:
                result += f"The '{pred_id}' you selected is correct.\n"

            return 1/3*op_score + 1/3*val_score + 1/3*id_score, result
        except:
            return  0, "The output of your action is illegal, please reselect your action."  
         
    def analyze_feedback(self, obs, actions, reward, feedback):
        #return f"In '{obs}'\nYour choice '{actions}'\nReward is {reward}\nYou must have a reward greater than 0.8 to proceed. Please reselect your action\n"
        if reward < 0.8:
            #return f"In '{obs}'\n Your choice '{actions}'\nReward is {reward}\nYou must have a reward greater than 0.8 to proceed. \n{feedback} Please reselect your choice.\n"
            return f"When your choice '{actions}', {feedback}. \n Then, you get reward is {reward}.\nYou must have a reward greater than 0.8 to proceed. \n Please reselect your choice.\n"

    def format_history(self, prev_obs, prev_actions):
        history_lines = []
        for obs, action in zip(prev_obs, prev_actions):
            history_lines.append(f"In '{obs}'\nYour choice '{action}'")
        
        history_str = "\n".join(history_lines)
        #history_str.insert(0,"The history feedback is :\n")
        return f"{history_str}"

    def update_holistic_strategy(self, current_strategy: str, evaluation_result: float, 
                               environment_feedback: str, holistic_prompt: str, analysis: str, iniobs:str) -> str:
        if evaluation_result >= self.evaluation_threshold:
            # 分析观察变化
            #recent_changes = observation_changes[-3:] if len(observation_changes) >= 3 else observation_changes
            # changes_analysis = "\n".join([
            #     f"Change {i+1}: {change['old_observation']} -> {change['new_observation']}"
            #     for i, change in enumerate(recent_changes)
            # ])
            #changes_observation = "\n".join(observation_changes)
            """
            输入局部策略的更新迭代情况，让它总结并输出下一步动作
            """
            update_prompt = f"""
            Your previous strategy is: {current_strategy}.
            Here is your analysis trace is : {environment_feedback}.
            And the recent status analysis is : {analysis}
            Your task is: " {iniobs}.
            
            Please give me your new strategy to slove your task, which should be concise and clear, about 100 words or less.
            The example is: 
            {holistic_prompt}
            
            """
            # Your can't seem to continue to complete your task. According to the original exploration strategy, please help me generate a new strategy to complete the task based on the following example, which should be concise and clear, about 100 words.
            messages = [
                {"role": "system", "content": "You are a strategy optimization assistant. Help improve the current strategy."},
                {"role": "user", "content": update_prompt}
            ]

            new_strategy, info = generate_response(
                messages=messages,
                model=self.model,
                temperature=self.args.temperature,
                stop_tokens=["Task:", "obs:", "act:"],
                seed=self.args.seed
            )
            self.strategy_history.append({
                'old_strategy': current_strategy,
                'new_strategy': new_strategy,
                'evaluation': evaluation_result,
                'observation_changes': environment_feedback,
                'timestamp': time.time()
            })
            
            #logging.info("Strategy updated based on evaluation and observation changes")
            return new_strategy, info
        return current_strategy

    def evaluate_strategy(self, observations: str, action: str, reward: float, analysis:str) -> float:
        self.total_attempts += 1
        if reward>0:
            self.success_count += 1
        try:
            if(int(analysis>0)):
                 self.success_count += 1
        except:
            print("The analysis is useless.")
        
        success_rate = self.success_count / self.total_attempts
        logging.info(f"Strategy evaluation - Success rate: {success_rate:.2f}")
        return success_rate
