import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MATH_PROMPT = (
    "You are an expert mathematical text summarizer. "
    "Your task is to extract ONLY the core mathematical reasoning, equations, and keywords from the text provided. "
    "REMOVE all conversational filler words (e.g. 'so', 'then', 'let us', 'we can see that', 'obviously', 'the answer is'). "
    "DO NOT explain your reasoning, just output the condensed mathematical keywords and terms separated by spaces."
)

HARMBENCH_PROMPT = (
    "You are an expert text summarizer focusing on intent and core content. "
    "Your task is to extract ONLY the core concepts, entities, and action verbs from the text provided. "
    "REMOVE all conversational filler words, greetings, apologies, or generic AI disclaimers (e.g. 'I cannot fulfill this request', 'As an AI language model', 'Here is the information'). "
    "DO NOT explain your reasoning, just output the condensed keywords and core message separated by spaces."
)

def build_prompt(text: str, task: str = "math") -> str:
    system_prompt = HARMBENCH_PROMPT if task == "harmbench" else MATH_PROMPT
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nHere is the text:\n{text}<|im_end|>\n<|im_start|>assistant\n"

def main():
    parser = argparse.ArgumentParser(description="Remove stopwords using LLM")
    parser.add_argument("--input_logs", type=Path, required=True)
    parser.add_argument("--output_logs", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=args.cache_dir,
    )
    
    print(f"Loading JSON from {args.input_logs}...")
    with args.input_logs.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all assistant outputs to batch process
    print("Collecting assistant texts...")
    to_process = [] # List of tuples: (problem_key, run_idx, turn_idx, original_text)
    
    for key, runs in data.items():
        for r_idx, run in enumerate(runs):
            for t_idx, turn in enumerate(run):
                if turn.get("role") == "assistant":
                    to_process.append((key, r_idx, t_idx, turn["output"]))

    print(f"Found {len(to_process)} assistant responses to clean.")
    
    task = "harmbench" if "harmbench" in str(args.input_logs).lower() else "math"
    print(f"Inferred task: {task} from input path.")
    
    # Process in batches
    pbar = tqdm(total=len(to_process), desc="Cleaning texts")
    
    for i in range(0, len(to_process), args.batch_size):
        batch = to_process[i:i+args.batch_size]
        prompts = [build_prompt(item[3], task) for item in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1, # Low temperature for consistent extraction
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode only the newly generated tokens
        input_lengths = inputs["input_ids"].shape[1]
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_lengths:], skip_special_tokens=True)
        
        # Update the data dictionary with cleaned texts
        for j, (key, r_idx, t_idx, _) in enumerate(batch):
            cleaned_text = decoded_outputs[j].strip()
            data[key][r_idx][t_idx]["output"] = cleaned_text
            
        pbar.update(len(batch))
        
    pbar.close()

    print(f"Saving cleaned logs to {args.output_logs}...")
    args.output_logs.parent.mkdir(parents=True, exist_ok=True)
    with args.output_logs.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
