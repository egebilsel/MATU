import os
import json
import random
import pickle
import numpy as np
from tqdm import tqdm
from camel.societies import RolePlaying
from camel.types import ModelType, ModelPlatformType
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from openai import OpenAI
from camel.prompts.ai_society import AISocietyPromptTemplateDict
from camel.prompts.base import TextPrompt  
from camel.generators import SystemMessageGenerator
from camel.types import TaskType, RoleType
from sentence_transformers import SentenceTransformer


base_gen = SystemMessageGenerator(task_type=TaskType.AI_SOCIETY)
base_prompts = dict(base_gen.sys_prompts)             
allowed_keys = set(base_gen.sys_msg_meta_dict_keys)   


os.environ["OPENAI_API_KEY"] = "xxx"

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=1.0).as_dict(),
)

embedding_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    # cache_folder=cache_dir,
)

def get_embedding(text: str) -> np.ndarray:
    emb = embedding_model.encode(text, normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)


def run_conversation(society, round_limit: int = 6, client=None):
    conversation_log = []
    user_embeddings = []
    assistant_embeddings = []
    
    input_msg = society.init_chat()
    
    for _ in range(round_limit):
        assistant_response, user_response = society.step(input_msg)
        
        if getattr(assistant_response, 'terminated', False):
            print(f"Assistant terminated: {assistant_response.info.get('termination_reasons', 'unknown')}")
            break
        if getattr(user_response, 'terminated', False):
            print(f"User terminated: {user_response.info.get('termination_reasons', 'unknown')}")
            break
            
        conversation_log.append({
            'role': 'user',
            'output': user_response.msg.content
        })
        conversation_log.append({
            'role': 'assistant',
            'output': assistant_response.msg.content
        })
        
        if client is not None:
            user_emb = get_embedding(user_response.msg.content)
            assistant_emb = get_embedding(assistant_response.msg.content)
        else:
            user_emb = None
            assistant_emb = None
        
        if user_emb is not None:
            user_embeddings.append(user_emb)
        if assistant_emb is not None:
            assistant_embeddings.append(assistant_emb)
        
        if 'CAMEL_TASK_DONE' in user_response.msg.content:
            break
        
        input_msg = assistant_response.msg
        
    return conversation_log, user_embeddings, assistant_embeddings

def evaluate_response(problem, solution, assistant_last_response, client):
    prompt = f"""Evaluate the following math problem and the assistant's answer.
Problem: {problem}
Reference Solution: {solution}
Assistant's Answer: {assistant_last_response}
Is the assistant's answer correct? Answer with "Correct" or "Incorrect" based on the final answer."""
    
    completion = client.chat.completions.create(
      model="gpt-5",
      messages=[
          {"role": "developer", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt}
      ]
    )
    evaluation = completion.choices[0].message.content.strip()
    if "incorrect" in evaluation.lower():
        return "Incorrect"
    else:
        return "Correct"

data_root = "MATH/test" 
user_embedding_pickle_file = "user_embedding_matrices_Math_gpt4o.pkl"
assistant_embedding_pickle_file = "assistant_embedding_matrices_Math_gpt4o.pkl"
accuracy_pickle_file = "accuracy_dict_Math_gpt4o.pkl"
conversation_log_file = "conversation_logs_Math_gpt4o.json"


user_embedding_dict = {}     
assistant_embedding_dict = {} 
accuracy_dict = {}          
all_conversation_logs = {}   

openai_client = OpenAI()

categories = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
for category in categories:
    category_path = os.path.join(data_root, category)
    json_files = [f for f in os.listdir(category_path) if f.endswith(".json")]
    questions = []
    for jf in json_files:
        file_id = os.path.splitext(jf)[0] 
        file_path = os.path.join(category_path, jf)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            questions.append((file_id, data))

    sample_size = min(300, len(questions))
    random
    sampled_questions = random.sample(questions, sample_size)
    random.seed(42)
    idx = 0 
    for file_id, question in tqdm(sampled_questions[:300], desc=f"Processing {category}"):
        idx += 1
        key = f"{category}_{file_id}" 
        problem_text = question["problem"]
        solution_text = question["solution"]
        safe_problem_text = problem_text.replace("{", "{{").replace("}", "}}")

        task_prompt = f"Answer the following math problem: \"{safe_problem_text}\". Thinking Mode: Minimal."
        task_kwargs = {
            'task_prompt': task_prompt,
            'with_task_specify': True,
            'task_specify_agent_kwargs': {'model': model}
        }
        user_role_kwargs = {
            'user_role_name': 'The Curious Math Enthusiast.',
            'user_agent_kwargs': {'model': model}
        }
        assistant_role_kwargs = {
            'assistant_role_name': 'The Master Mathematician.',
            'assistant_agent_kwargs': {'model': model}
        }
        
        conversation_logs_list = []
        user_emb_matrices = []
        assistant_emb_matrices = []
        accuracy_list = []
        
        for run in range(10):
            society = RolePlaying(
                **task_kwargs,
                **user_role_kwargs,
                **assistant_role_kwargs,      
                 )
            conv_log, user_emb_list, assistant_emb_list = run_conversation(society, round_limit=10, client=openai_client)
            conversation_logs_list.append(conv_log)
            
            if user_emb_list:
                user_matrix = np.vstack(user_emb_list)
            else:
                user_matrix = np.array([])
            if assistant_emb_list:
                assistant_matrix = np.vstack(assistant_emb_list)
            else:
                assistant_matrix = np.array([])
            user_emb_matrices.append(user_matrix)
            assistant_emb_matrices.append(assistant_matrix)
            
            last_assistant_response = None
            for idx, msg in enumerate(conv_log):
                if msg['role'] == 'user' and "<CAMEL_TASK_DONE>" in msg['output']:
                    if idx > 0 and conv_log[idx - 1]['role'] == 'assistant':
                        last_assistant_response = conv_log[idx - 1]['output']
                    break
            if last_assistant_response is None:
                for msg in reversed(conv_log):
                    if msg['role'] == 'assistant':
                        last_assistant_response = msg['output']
                        break
            print(last_assistant_response)
            if last_assistant_response is None:
                eval_result = "No Response"
            else:
                eval_result = evaluate_response(problem_text, solution_text, last_assistant_response, openai_client)
            accuracy_list.append(eval_result)
        
        user_embedding_dict[key] = user_emb_matrices
        assistant_embedding_dict[key] = assistant_emb_matrices
        accuracy_dict[key] = accuracy_list
        all_conversation_logs[key] = conversation_logs_list
        
        if idx % 5 == 0:
            with open(user_embedding_pickle_file, "wb") as f:
                pickle.dump(user_embedding_dict, f)
            with open(assistant_embedding_pickle_file, "wb") as f:
                pickle.dump(assistant_embedding_dict, f)
            with open(accuracy_pickle_file, "wb") as f:
                pickle.dump(accuracy_dict, f)
            with open(conversation_log_file, "w", encoding="utf-8") as f:
                json.dump(all_conversation_logs, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved after processing {idx+1} problems in category {category}.")

with open(user_embedding_pickle_file, "wb") as f:
    pickle.dump(user_embedding_dict, f)
with open(assistant_embedding_pickle_file, "wb") as f:
    pickle.dump(assistant_embedding_dict, f)
with open(accuracy_pickle_file, "wb") as f:
    pickle.dump(accuracy_dict, f)
with open(conversation_log_file, "w", encoding="utf-8") as f:
    json.dump(all_conversation_logs, f, ensure_ascii=False, indent=2)
print("All data saved.")
