import os
import time
import yaml
import openai
import argparse
import alfworld
import alfworld.agents.environment
from alfworld.agents.environment import get_environment
from fuzzywuzzy import process
import dashscope
import sys
import numpy as np
from agents.dtra import StrategyManager
from utils import load_data
from collections import deque
import json
import random
import re
from utils import load_data
from thought_retrieval import (
    retrieve_demonstration_task_meta,
    retrieve_demonstration_thought,
    retrieve_demonstration_task_thought
)
from agents.llm import (
    generate_response
)
from prompts.sys_prompt import sys_message, sys_message_with_mark

current_path = os.path.realpath(__file__)
path= os.path.dirname(current_path)+"/"
summery_test = """Summary of actions and observations:
- went to A.
- Found a ...
- example B.
- Found a ...
"""
    
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=path+"./refined_data")
    parser.add_argument("--emb_dir", type=str, default=path+"./data")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--forward_step", type=int, default=1)
    parser.add_argument("--backward_step", type=int, default=0)
    parser.add_argument("--with_mark", action="store_true")
    parser.add_argument("--with_prev", action="store_true")
    parser.add_argument("--with_double", action="store_true")
    parser.add_argument("--with_action", action="store_true")
    parser.add_argument("--with_local", action="store_true")
    parser.add_argument("--with_global", action="store_true")
    parser.add_argument("--model", type=str, default="@cf/google/gemma-3-12b-it")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=int, default=0.3)
    args = parser.parse_args()
    return args

#parser = create_parser()
args = parse_args()
with open(path+'base_config.yaml') as reader:
    config = yaml.safe_load(reader)
    
split = "eval_out_of_distribution"
#config = load_config(filen=path+'base_config.yaml')
env_type = config['env']['type']
env = get_environment(env_type)(config, train_eval=split)
env = env.init_env(batch_size=1)

sys_mess = sys_message_with_mark if args.with_mark else sys_message

def process_ob(ob, process_init=False):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    elif ob.startswith('You are in the middle ') and process_init:
        ob = "In the middle of the room" + ob[ob.find(', '):]
    return ob

def format_step(step, idx, task):
    if idx == 0:
        template = f"{step['obs']}\nYour task is to: {task}\n> think: {step['thought']}\n> act: {step['act']}\n"
    else:
        template = f"{process_ob(step['obs'])}\n> think: {step['thought']}\n> act: {step['act']}\n"
    return template

def make_prompt_thought(template, demos, prev_prompts, ob, args):
    demo_prompts = []
    for demo in demos:
        traj, step_idx = demo["trajectory"], demo["step_idx"]
        task, traj = traj["task"], traj["traj"]
        demo_section = ""
        for idx in range(len(traj)-1):
            demo_section += format_step(traj[idx], idx, task)
        demo_prompts.append(demo_section)
    
    prev_steps = "\n".join(prev_prompts) + "\n"
    
    result = template.format(*demo_prompts) + prev_steps + f"{ob}\n> think:"+"Determine the nodes necessary for the current task."
    return result
    

def make_prompt(template, demos, prev_prompts, ob, thought, args):
    demo_prompts = []
    for demo in demos:
        traj, step_idx = demo["trajectory"], demo["step_idx"]
        task, traj = traj["task"], traj["traj"]
        start_idx = max(0, step_idx-args.backward_step)
        end_idx = min(step_idx+1+args.forward_step, len(traj)-1)
        demo_section = ""
        if start_idx > 0:
            init_ob = process_ob(traj[0]['obs'], True)
            demo_section += f"{init_ob}\n" + f"Your task is to: {task}\n"
        for idx in range(start_idx, end_idx):
            demo_section += f"[Step {idx-step_idx}]\n" if args.with_mark else ""
            demo_section += format_step(traj[idx], idx, task)
        demo_section += process_ob(traj[end_idx]["obs"]) + '\n'
        demo_prompts.append(demo_section)
    
    # Insert history piece before current step
    if args.with_prev and len(prev_prompts):
        ahead_step = args.backward_step + args.forward_step
        start_idx = max(0, len(prev_prompts)-ahead_step)
        prev_ = prev_prompts.copy()
        if start_idx > 0:
            meta_info = process_ob(prev_[0], True)
            meta_info = meta_info[:meta_info.find("> think:")]
            prev_[start_idx] = meta_info + prev_[start_idx]
        prev_steps = "\n".join(prev_[start_idx:]) + "\n"
    # No history piece, but specify the task still
    elif len(prev_prompts):
        meta_info = process_ob(prev_prompts[0], True)
        prev_steps = meta_info[:meta_info.find("> think:")]
    # Step 0
    else:
        prev_steps = ""
    
    result = template.format(*demo_prompts) + prev_steps + f"{ob}\n> think: {thought}\n> act:"
    result = (result, f"{ob}\n> think: {thought}\n> act:")
    return result

def alfworld_dtra(thought_embeddings,
    ob_embeddings,
    expert_demos,
    init_ob='',
    oinfo='',):
    strategy_manager = StrategyManager(args=args)
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    print_stuff, actions, prev_prompts, observation = [init_ob], [], [], init_ob
    thought_demos = retrieve_demonstration_task_meta(
        init_ob, ob_embeddings, expert_demos, args
    )
    
    demo_section = '\n'.join([str(i)+'): {'+str(i)+'}' for i in range(args.K)])
    prompt_template = f"Here are some examples: <'\n{demo_section}'>\n"
    
    if args.with_global:
        if args.with_prev:
            holistic_prompt = make_prompt_thought(
            prompt_template, thought_demos, prev_prompts, observation, args
        )
        else:
            holistic_prompt = " "
        holistic_thought, sinfo = strategy_manager.update_holistic_strategy(  
                        "",
                        1,
                        "",
                        holistic_prompt,
                        "",
                        init_ob
                    )
        for k, v in sinfo.items():
            token_stats[k] += v
        holistic_thought = holistic_thought.replace(" in ", " in/on ").replace(" on ", " in/on ")
    else:
        holistic_thought = ""

    prev_analysis = []
    analysis = ""
    
    for episode in range(50):
        #try:
        admissible_commands = list(oinfo['admissible_commands'])

        limit = strategy_manager.format_action(admissible_commands)
        Action_limit = "Here is your action space: " + limit + ". Try not to re-select actions you've already selected."

        if args.with_prev and args.with_mark: # local thinking prompt demo
            step_by_step_demos =  make_prompt_thought(
            prompt_template, thought_demos, prev_prompts, observation, args
        ) 
            example_step = "Your step demos is : " + ",".join(step_by_step_demos) +".\n"
        else:
            example_step = ""

        if len(analysis)>0:
            analysis = "Your current analysis is: "+ analysis
        if args.with_global:
            holistic_strategy = f"<<Your holistic strategy is: {holistic_thought}>>"
        else:
            holistic_strategy = ""
        if args.with_action:
            step_by_step_prompt = f"""
            {holistic_strategy}
            {Action_limit}
            The actions you have done so far and the feedback is:
            {".".join(prev_prompts)}
            {example_step}
            {analysis}
            Please give me your next action based on the above information, about 10 words.
            """
            message = [ 
                    {   "role": "system","content": sys_message},
                    {   "role": "user","content": step_by_step_prompt}
            ]
            step_by_step_thought, sinfo = generate_response(
                messages=message,
                model=strategy_manager.model_chat,
                args=args
            )
            for k, v in sinfo.items():
                token_stats[k] += v
            step_by_step_thought = step_by_step_thought.replace(" in ", " in/on ").replace(" on ", " in/on ")
            thought = "Your thinking is: " + step_by_step_thought + "\n"
        else:
            step_by_step_prompt = ""
            thought = f"""{holistic_strategy}\n{analysis}\n"""
        
        # action prompt
        
        
        decision_prompt = step_by_step_prompt +thought + "The actions you have done so far and the feedback is : <"+"\n".join(prev_prompts) + ">\n"+ Action_limit  + "Base on above information, select your action.\n"
                                    
        # decision action
        message = [ 
                {   "role": "system","content": sys_message},
                {   "role": "user","content": decision_prompt}
        ]
        decision_action, sinfo = generate_response(
            messages=message,
            model=strategy_manager.model_chat,
            args=args
        )
        for k, v in sinfo.items():
            token_stats[k] += v
        decision_action = decision_action.replace(" in ", " in/on ").replace(" on ", " in/on ")

        print(decision_action)
        
        action_next = strategy_manager.semantic_match(decision_action, admissible_commands[0])
        
        # action
        obs, scores, dones, oinfo = env.step([action_next])
        #print("Action: {}, Obs: {}".format(action_next, obs[0]))

        # feedback
        feedback =  f"The action '{action_next}' have done, {obs[0]}. Your get reward is {scores[0]}."
        prev_prompts.append(feedback)

        if args.with_local:
            analysis_prompt = (holistic_strategy
                    + "The feedback is :"+"\n".join(prev_prompts) + "\n" 
                    + "Your previous analysis is : "+ analysis + "\n" 
                    + "Your current observation is :" + obs[0] + "\n"
                    + Action_limit +"\n"
                    +"Please return a score from 0 to 100 based on the current observation as an evaluation of the feedback for your holistic strategy. The score has five levels, with a score of 0 indicating no progress or no repetition of meaningless actions; A score of 25 indicates that the exploration task is ongoing; A score of 50 indicates that the overall strategy has been explored or has been repeated in recent actions, and the overall strategy needs to be changed. A score of 75 indicates critical progress and requires further guidance; A score equal to 100 indicates the task has been completed.\n"
                    + "Your just give me a number and your analysis, like: Score: 0 \nanalysis is : in the step " + str(episode+1)+". At the end, Summarize what you've found and what you've got in your hands, and indicate what you've done and what you've found."
                    +"The example of actions and observations Summary is :"+summery_test+"\n"
                    + "And in the end, give me the next step plan.\n"
                    +"Your initial environment and task is: " + init_ob +"\n")

            message = [ 
                    {   "role": "system","content": "You are a helpful assistant."},
                    {   "role": "user","content": analysis_prompt}
            ]
            analysis, sinfo = generate_response(
                messages=message,
                model=strategy_manager.model_chat,
                args=args
            )
            for k, v in sinfo.items():
                token_stats[k] += v
        else:
            analysis = "Score: 100"

        #print("analysis is :", analysis)
        
        anascore = re.search(r'Score:\s*([-+]?\d+)', analysis)
        if anascore:
            score_str =  int(anascore.group(1))
            score_int = int(score_str)
            anascore = score_int
        else:
            print("Can not find Score.")
            anascore = 0

        observation = obs[0]
        # Update global strategy
        if args.with_global:
            if anascore>=50: 
                if args.with_prev:
                    holistic_prompt = make_prompt_thought(
                        prompt_template, thought_demos, prev_prompts, init_ob, args
                    ) 
                else:
                    holistic_prompt = ""
                holistic_thought, sinfo = strategy_manager.update_holistic_strategy(  
                    holistic_thought,
                    anascore/100,
                    json.dumps(prev_analysis),
                    holistic_prompt,
                    analysis,
                    init_ob
                )
                for k, v in sinfo.items():
                    token_stats[k] += v
        #print("analysis is :", analysis)
        prev_analysis.append(analysis)
        actions.append(decision_action)
        
        # fail if repeat often
        # if len(actions) >= 3 and (actions[-1] == actions[-2]) and (actions[-2] == actions[-3]):
        #     return 0, print_stuff
        if len(actions) >= 6 and (actions[-1] == actions[-3]) and (actions[-3] == actions[-5]) \
            and (actions[-2] == actions[-4]) and (actions[-4] == actions[-6]):
            return 0, print_stuff
        
        # log
        print(f'Global Reason {episode+1}: {holistic_thought}')
        print(f'Local Reason {episode+1}: {analysis}')
        print(f'Act {episode+1}: {action_next}')
        print(f'Obs {episode+1}: {observation}')
        print_stuff.append(f'\nGlobal Reaso {episode+1}: {holistic_thought}')
        print_stuff.append(f'\nLocal Reason {episode+1}: {analysis}')
        print_stuff.append(f'\nAct {episode+1}: {action_next}')
        print_stuff.append(f'\nObs {episode+1}: {observation}')
        
        # done
        if dones[0]:
            print_stuff.append(f'\nTokens : {token_stats}')
            return scores[0], print_stuff
            
                
        # except Exception as e:
        #     print(f"Error in episode {episode}: {str(e)}")
        #     continue
    print("Task completed in {} episodes!".format(episode + 1))



def alfworld_run(
    thought_embeddings,
    ob_embeddings,
    expert_demos,
    init_ob='',
    info={'admissible_commands':[0,0,0]},
):
    strategy_manager = StrategyManager(args=args)
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    print_stuff, actions, prev_prompts, observation = [init_ob], [], [], init_ob
    
    # retrieval for thought generation,从专家示例中找到最相似当前场景的前K个案例
    thought_demos = retrieve_demonstration_task_meta(
        init_ob, ob_embeddings, expert_demos, args
    )
    
    for i in range(50):
        # reason
        admissible_commands = list(info['admissible_commands'])
        prompt = make_prompt_thought(
            prompt_template, thought_demos, prev_prompts, observation, args
        ) 
        
        #try:
        message = [ 
            {   "role": "system","content": sys_message},
            {   "role": "user","content": prompt}
        ]
        thought, sinfo = generate_response(
            messages=message,
            model=strategy_manager.model_chat,
            args=args
        )
        for k, v in sinfo.items():
            token_stats[k] += v
        #thought = llm(sys_message, prompt, stop=['\n>']).strip() 
        # except:
        #     print("Thought Production Error...")
        #     return 0, print_stuff
        
        thought = thought.replace(" in ", " in/on ").replace(" on ", " in/on ")
        
        # thought retrieval
        retrieved_demos = retrieve_demonstration_thought(
            thought, thought_embeddings, expert_demos, args
        )
        
        # decision
        prompt, prompt_cut = make_prompt(
            prompt_template, retrieved_demos, prev_prompts, observation, thought, args
        )
        
        #try:
        message = [ 
            {   "role": "system","content": sys_mess},
            {   "role": "user","content": prompt}
        ]
        action, sinfo = generate_response(
            messages=message,
            model=strategy_manager.model_chat,
            args=args
        )
        action = strategy_manager.semantic_match(action, admissible_commands[0])
        for k, v in sinfo.items():
            token_stats[k] += v
        #action = llm(sys_mess, prompt, stop=['\n']).strip()
        # except:
        #     print("Action Production Error...")
        #     return 0, print_stuff
        
        # hack for GPT wrong output
        action = action.replace(" in ", " in/on ").replace(" on ", " in/on ")
        actions.append(action)
        
        # for alignment
        prev_prompts.append(prompt_cut + ' ' + action)
        
        # fail if repeat often
        # if len(actions) >= 3 and (actions[-1] == actions[-2]) and (actions[-2] == actions[-3]):
        #     return 0, print_stuff
        if len(actions) >= 6 and (actions[-1] == actions[-3]) and (actions[-3] == actions[-5]) \
            and (actions[-2] == actions[-4]) and (actions[-4] == actions[-6]):
            return 0, print_stuff
        
        # transition
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        
        # log

        print(f'Reason {i+1}: {thought}')
        print(f'Act {i+1}: {action}')
        print(f'Obs {i+1}: {observation}')
        print_stuff.append(f'\nReason {i+1}: {thought}')
        print_stuff.append(f'\nAct {i+1}: {action}')
        print_stuff.append(f'\nObs {i+1}: {observation}')
        
        # done
        if done:
            print_stuff.append(f'\nTokens : {token_stats}')
            return reward, print_stuff
    
    return 0, print_stuff

prefixes = {
    'pick_and_place': 'put',
    'look_at_obj': 'examine',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'pick_two_obj': 'puttwo'
}
cnts = [0] * 6
rs = [0] * 6

mark_tag = "with_mark" if args.with_mark else "no_mark"
prev_tag = "with_prev" if args.with_prev else "no_prev"
exp_name = f"trad_{args.forward_step}f{args.backward_step}b_{mark_tag}_{prev_tag}_{args.exp_name}"
os.makedirs(path+f"results/{args.model}/{exp_name}", exist_ok=True)
#os.makedirs(path+f"gpt-4-results/{exp_name}", exist_ok=True)
expert_demos, target_thoughts, ob_embeddings, thought_embeddings = load_data(args)

demo_section = '\n'.join(['{'+str(i)+'}' for i in range(args.K)])
prompt_template = f"Here are two examples.\n{demo_section}\nHere is the task.\n"

for idx in range(134):
    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    print(f"{idx}:", name)
    print("ob0", ob)
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            if args.with_double:
                r, log_ = alfworld_dtra(
                    thought_embeddings[i],
                    ob_embeddings[i],
                    expert_demos[i],
                    init_ob=ob,
                    oinfo=info
                )
            else:
                r, log_ = alfworld_run(
                    thought_embeddings[i],
                    ob_embeddings[i],
                    expert_demos[i],
                    init_ob=ob,
                    info=info
                )
            rs[i] += r
            cnts[i] += 1
            break
    with open(path+f"results/{args.model}/{exp_name}/{idx}.log", "w") as f:
        f.writelines(log_)
        f.write(f"\nSuccess: {r}")
    print(idx, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
    print('------------')
with open(path+f"results/{args.model}/{exp_name}/result_.log", "w") as f:
    f.write(f"{idx} r {r} rs {rs} cnts {cnts} sum(rs)/sum(cnts) {sum(rs)/sum(cnts)}")