import logging
import re
import time
import inspect
import tiktoken
import ollama
import backoff
import openai
import httpx
import requests
import os
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import ssl

class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2 # Enforce minimum TLS versions
        kwargs['ssl_context'] = ctx
        return super(TLSAdapter, self).init_poolmanager(*args, **kwargs)

# Send the request using a custom adapter
session = requests.Session()
session.mount('https://', TLSAdapter())

current_path = os.getcwd()
host="127.0.0.1"
port="11434"
client_oll= ollama.Client(host=f"http://{host}:{port}")
#client_gpt = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client_gpt = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
ACCOUNT_ID = os.environ.get("ACCOUNT_ID")
AUTH_TOKEN = os.environ.get("CLOUDFLARE_API_KEY")

from typing import List, Dict, Tuple, Union

logger = logging.getLogger("main")

def get_mode(model: str) -> str:
    """Check if the model is a chat model."""

    if model in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-turbo",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    ]:
        return "chat"
    
    elif model in [
        "davinci-002",
        "gpt-3.5-turbo-instruct-0914",
    ]:
        return "completion"
    elif model in [
        "qwen2.5:latest",
        "deepseek-r1:7b",
    ]:
        return "ollama"

    elif "7b" in model:
        return "7B"
    elif "32b" in model:
        return "32B"
    elif "8b" in model:
        return "8B"
    elif "70b" in model:
        return "70B"
    elif "12b" in model:
        return "12B"
    elif "14b" in model:
        return "14B"
    else:
        raise ValueError(f"Unknown model: {model}")

def generate_response(
    messages: List[Dict[str, str]],
    model: str,
    args,
    stop_tokens=['\n>']
) -> Tuple[str, Dict[str, int]]:
    """Send a request to the OpenAI API."""

    logger.info(
        f"Send a request to the language model from {inspect.stack()[1].function}"
    )
    seed =  args.seed
    temperature = args.temperature
    

    while True:
        try:
            if get_mode(model) == "chat":
                response = client_gpt.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stop=stop_tokens if stop_tokens else None,
                    # **seed
                )
                #message = response["choices"][0]["message"]["content"]
                message = response.choices[0].message.content
            elif get_mode(model)=="completion":
                prompt = "\n\n".join(m["content"] for m in messages) + "\n\n"
                response = client_gpt.chat.completions.create(
                    prompt=prompt,
                    engine=model,
                    temperature=temperature,
                    stop=stop_tokens if stop_tokens else None,
                    **seed
                )
                message = response["choices"][0]["text"]
            elif get_mode(model)=="ollama":
                response = client_oll.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, 
                         "top_p": 0.8, "presence_penalty": 0.1,"frequency_penalty": 0.1}
                )
                #cleaned_text = response['message']['content']
                message = re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL)
            else:
                response = session.post(
                f"https://gateway.ai.cloudflare.com/v1/{ACCOUNT_ID}/planning/workers-ai/{model}",
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                json={"messages": messages,
                      "temperature": 0.3, 
                        "top_p": 0.8, 
                        "presence_penalty": 0.1,
                        "frequency_penalty": 0.1,
                        "max_tokens":2000,
                    }
                )
                # response = requests.post(
                # f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model}",
                #     headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                #     json={
                #         "messages": messages,
                #         "temperature": 0.3, 
                #         "top_p": 0.8, 
                #         "presence_penalty": 0.1,
                #         "frequency_penalty": 0.1
                #     }
                #)
                time.sleep(1)
                #print(response)
                try:
                    response = response.json()['result']
                    message = response['response']
                    message = re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL)
                except:
                    response = {'response': '', 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
                    message = ""
            break
        except Exception as e:
            raise e
        except:
            time.sleep(1)
    info = get_usage_info(response)

    return message, info

def get_usage_info(response):
    if isinstance(response, dict):
        #  Cloudflare-like dic
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    elif hasattr(response, "usage"):
        # OpenAI-like object
        usage = response.usage
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    elif hasattr(response, "prompt_eval_count"):
        # Ollama-like object
        prompt_tokens = response.prompt_eval_count
        completion_tokens = response.eval_count
        total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    else:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

def clean_quotes(text: str) -> str:
    text = text.strip()
    if (text.startswith("'") and text.endswith("'")) or \
       (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]
    return text

def extract_and_clean(response: str, backtick="```") -> str:
    extracted = extract_from_response(response, backtick)

    cleaned = clean_quotes(extracted)

    return cleaned

def extract_from_response(response: str, backtick="```") -> str:
    if backtick == "```":
        # Matches anything between ```<optional label>\n and \n```
        pattern = r"```(?:[a-zA-Z]*)\n?(.*?)\n?```"
    elif backtick == "`":
        pattern = r"`(.*?)`"
    else:
        raise ValueError(f"Unknown backtick: {backtick}")
    match = re.search(
        pattern, response, re.DOTALL
    )  # re.DOTALL makes . match also newlines
    if match:
        extracted_string = match.group(1)
    else:
        extracted_string = response

    return extracted_string
