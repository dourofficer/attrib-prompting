import multiprocessing
import os
import json
import torch
import random
from transformers import pipeline as pipeline_function, AutoTokenizer, AutoModelForCausalLM, Pipeline
from vllm import LLM, SamplingParams
from tqdm import tqdm

def _get_sorted_json_files(directory_path):
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error reading or sorting files in {directory_path}: {e}")
        return []

def _load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def _uses_role_field(is_handcrafted):
    if isinstance(is_handcrafted, str):
        return is_handcrafted.lower() == "true"
    return bool(is_handcrafted)

def _agent_label(entry, preferred_field):
    fallback_field = "name" if preferred_field == "role" else "role"
    return entry.get(preferred_field) or entry.get(fallback_field) or "Unknown Agent"

def _run_local_generation_parallel(model_obj, batch_messages, model_family='llama', batch_size=4):
    if model_family != 'qwen':
        print("Warning: Parallel processing is currently only supported for Qwen models")
        return [_run_local_generation(model_obj, msg, model_family) for msg in batch_messages]
    
    try:
        model, tokenizer = model_obj
        max_new_tokens = 1024
        temperature = 0.6
        top_p = 0.95

        # Process in batches
        all_responses = []
        for i in range(0, len(batch_messages), batch_size):
            batch = batch_messages[i:i + batch_size]
            
            # Prepare all inputs in the batch
            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for messages in batch
            ]
            
            # Tokenize all inputs
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            
            # Generate for the entire batch
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Extract only the new tokens for each sequence
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # Decode all sequences
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_responses.extend(responses)
            
        return all_responses
        
    except Exception as e:
        print(f"Error during parallel model execution ({model_family}): {e}")
        import traceback
        traceback.print_exc()
        return [None] * len(batch_messages)

def _run_local_generation(model_obj, messages, model_family='llama'):
    max_new_tokens=1024
    temperature=0.6
    top_p=0.95

    try:
        if model_family == 'llama' and isinstance(model_obj, Pipeline):
            pipe = model_obj
            terminators = [
                pipe.tokenizer.eos_token_id,
                pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pipe.tokenizer.eos_token_id
            )
            if outputs and outputs[0]["generated_text"] and isinstance(outputs[0]["generated_text"], list):
                 return outputs[0]["generated_text"][-1]["content"]
            else:
                 print("Warning: Unexpected output format from Llama pipeline.")
                 return None
        elif model_family == 'qwen' and isinstance(model_obj, tuple) and len(model_obj) == 2:
            model, tokenizer = model_obj
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            print(f"the total length is {len(model_inputs.input_ids[0])}")
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id # Use default EOS for Qwen generate
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        else:
            print(f"Error: Unsupported model_family '{model_family}' or incorrect model object type provided.")
            return None

    except Exception as e:
        print(f"Error during local model execution ({model_family}): {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_all_at_once_local_parallel(model_obj, directory_path: str, is_handcrafted: bool, model_family: str, batch_size=4):
    print(f"\n--- Starting Local All-at-Once Parallel Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"
    
    # Prepare all prompts first
    all_prompts = []
    file_mapping = []  # Keep track of which file each prompt corresponds to
    
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue
            
        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        
        if not chat_history:
            continue
            
        chat_content = "\n".join([
            f"{_agent_label(entry, index_agent)}: {entry.get('content', '')}" 
            for entry in chat_history
        ])
        
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem} \n"
            f"The Answer for the problem is: {ground_truth}\n"
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
        )
        
        system_prompt = "You are a helpful assistant skilled in analyzing conversations."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        all_prompts.append(messages)
        file_mapping.append(json_file)
    
    # Process all prompts in parallel batches
    print(f"Processing {len(all_prompts)} files in batches of {batch_size}...")
    responses = _run_local_generation_parallel(model_obj, all_prompts, model_family, batch_size)
    
    # Output results
    for json_file, response in zip(file_mapping, responses):
        print(f"\nPrediction for {json_file}:")
        if response:
            print(response)
        else:
            print("Failed to get prediction from local model.")
        print("\n" + "="*50 + "\n")

def _run_vllm_generation(model_path: str, prompts: list, max_tokens: int = 8192, tensor_parallel_size: int = 8, temperature: float = 0.0, top_p: float = 1.0):
    # Ensure spawn method is set in the worker process
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    """
    Run generation using vllm for parallel inference with optimized memory usage
    """
    try:
        # Initialize vllm with optimized settings
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=True,

            rope_scaling={
                "rope_type": "yarn",
                "factor": 4,
                "original_max_position_embeddings": 32768,  # Model's original context length
            },
        )
        
        # Load tokenizer to get EOS token ID and apply chat schema
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'  # Set left padding for decoder-only model
        
        # Process prompts with chat schema
        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)

        formatted_prompts = formatted_prompts
        
        # Set sampling parameters aligned with HuggingFace
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=1024,
        )

        # Generate responses for all prompts in parallel
        outputs = llm.generate(formatted_prompts, sampling_params)

        # Extract generated text and clean up responses
        responses = []
        for prompt, output in zip(formatted_prompts, outputs):
            # Get only the newly generated text, not including the prompt
            response = output.outputs[0].text
            # Take only the first complete answer if multiple are present
            if "Agent Name:" in response:
                parts = response.split("Agent Name:")
                # Keep the first complete answer
                response = "Agent Name:" + parts[1].split("\n\n")[0]
            responses.append(response.strip())
        return responses
        
    except Exception as e:
        print(f"Error during vllm generation: {e}")
        import traceback
        traceback.print_exc()
        return [None] * len(prompts)

def analyze_all_at_once_vllm(model_path: str, directory_path: str, is_handcrafted: bool, tensor_parallel_size: int = 8, file_list: list = None, temperature: float = 0.0, top_p: float = 1.0):
    """
    Analyze conversations using vllm for parallel inference (LLM-as-a-Judge baseline)

    file_list: optional explicit list of JSON filenames to process; supports DP sharding.
    """
    print(f"\n--- Starting VLLM All-at-Once Analysis (baseline, no schemata) ---")
    if file_list is not None:
        json_files = file_list
    else:
        json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"
    print(f"Immediately after assignment: index_agent={index_agent}, is_handcrafted={is_handcrafted}")
    print(f"----- index agent is {index_agent} is handcrafted {is_handcrafted} ------")
    
    # Prepare all prompts
    all_prompts = []
    file_mapping = []
    
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue
            
        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        
        if not chat_history:
            continue
            
        # chat_content = "\n".join([
        # f"Step {idx}\n\n\n{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" 
        # for idx, entry in enumerate(chat_history)
        # ])

        chat_content = "\n".join([
        f"{_agent_label(entry, index_agent)}: {entry.get('content', '')}" 
        for idx, entry in enumerate(chat_history)
        ])
        
        # prompt = (
        #     "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        #     f"The problem is:  {problem} \n"
        #     f"The Answer for the problem is: {ground_truth}\n"
        #     "Identify which agent made an error, at which step, and explain the reason for the error. "
        #     "Here's the conversation:\n\n" + chat_content +
        #     "\n\nBased on this conversation, please predict the following:\n"
        #     "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        #     "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        #     '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
        #     "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        #     "3. The reason for your prediction."
        #     "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
        # )
        
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem} \n"
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
        )
        all_prompts.append(prompt)
        file_mapping.append(json_file)
    
    # Process all prompts using vllm
    print(f"Processing {len(all_prompts)} files using vllm with {tensor_parallel_size} GPUs...")
    responses = _run_vllm_generation(
        model_path=model_path,
        prompts=all_prompts,
        tensor_parallel_size=tensor_parallel_size,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Output results
    for json_file, response in zip(file_mapping, responses):
        print(f"\nPrediction for {json_file}:")
        if response:
            print(response)
        else:
            print("Failed to get prediction from vllm.")
        print("\n" + "="*50 + "\n")

def _run_vllm_generation_with_schemata(model_path: str, prompts: list, schemata: dict, file_numbers: list, max_tokens: int = 8192, tensor_parallel_size: int = 8, temperature: float = 0.0, top_p: float = 1.0):
    """
    Run generation using vllm for parallel inference with schemata support
    
    Args:
        model_path: Path to the model
        prompts: List of prompts to process
        schemata: Dictionary mapping file numbers to error schemata (can be single schema string or list of schemata)
        file_numbers: List of file numbers corresponding to each prompt
        max_tokens: Maximum number of tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
        
    Returns:
        List of generated responses
    """
    # Ensure spawn method is set in the worker process
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
        
    try:
        # Initialize vllm with optimized settings
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=True,

            rope_scaling={
                "rope_type": "yarn",
                "factor": 4,
                "original_max_position_embeddings": 32768,  # Model's original context length
            },
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'
        
        # Process prompts with chat schema AND error schemata
        formatted_prompts = []
        for prompt, file_num in zip(prompts, file_numbers):
            # Add error schema if available for this file number
            if file_num in schemata:
                schema_content = schemata[file_num]
                
                # Check if schema_content is a list (multiple schemata) or string (single schema)
                if isinstance(schema_content, list):
                    # Multiple schemata - format them nicely
                    if len(schema_content) == 1:
                        combined_schema = schema_content[0]
                    else:
                        schema_parts = []
                        for i, content in enumerate(schema_content):
                            schema_parts.append(f"Schema {i+1}:\n{content}")
                        combined_schema = "\n\n".join(schema_parts)
                    
                    schema_text = f"Here are error schemata to help guide your analysis:\n\n{combined_schema}"
                else:
                    # Single schema (string)
                    schema_text = f"Here's a error schema to help guide your analysis:\n\n{schema_content}"
                
                prompt = f"{prompt}\n\n{schema_text}.\n\n\nPlease remember the error schema{'s are' if isinstance(schema_content, list) and len(schema_content) > 1 else ' is'} just to guide your analysis. You can neglect it if you find the schema is not helpful.\nPlease remember to answer in the following format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n"
            
            # Format with chat schema
            messages = [
                {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=1024,
        )

        # Generate responses for all prompts in parallel
        outputs = llm.generate(formatted_prompts, sampling_params)

        # Extract and clean up responses
        responses = []
        for output in outputs:
            response = output.outputs[0].text
            if "Agent Name:" in response:
                parts = response.split("Agent Name:")
                response = "Agent Name:" + parts[1].split("\n\n")[0]
            responses.append(response.strip())
        return responses
        
    except Exception as e:
        print(f"Error during vllm generation with schemata: {e}")
        import traceback
        traceback.print_exc()
        return [None] * len(prompts)

def analyze_all_at_once_vllm_with_schemata(model_path: str, directory_path: str, is_handcrafted: str, schemata: dict, tensor_parallel_size: int = 8, file_list: list = None, temperature: float = 0.0, top_p: float = 1.0):
    """
    Analyze conversations using vllm with error schemata for parallel inference

    Args:
        model_path: Path to the model
        directory_path: Directory containing JSON files
        is_handcrafted: Whether the data is handcrafted
        schemata: Dictionary mapping file numbers to schemata (can be single string or list of strings per file)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        file_list: Optional explicit list of JSON filenames to process. When
            provided, overrides the directory scan — used for data-parallel
            sharding (each worker processes only its assigned slice).
    """
    print(f"\n--- Starting VLLM All-at-Once Analysis with Schemata ---")
    if file_list is not None:
        json_files = file_list
    else:
        json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"
    
    # Prepare all prompts
    all_prompts = []
    file_numbers = []
    file_mapping = []
    
    print(f"Number of JSON files to process: {len(json_files)}")
    print(f"Number of schemata loaded: {len(schemata)}")
    print(f"Schema numbers available: {sorted(schemata.keys())}")
    
    # Check if schemata contain multiple schemata per file
    multi_schema_count = sum(1 for v in schemata.values() if isinstance(v, list))
    if multi_schema_count > 0:
        print(f"Files with multiple schemata: {multi_schema_count}")
        # Show example of multi-schema structure
        for k, v in list(schemata.items())[:3]:
            if isinstance(v, list):
                print(f"  File #{k}: {len(v)} schemata")
    
    for json_file in json_files:
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue
            
        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        
        if not chat_history:
            continue
            
        # chat_content = "\n".join([
        # f"Step {idx}\n\n\n{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" 
        # for idx, entry in enumerate(chat_history)
        # ])
        chat_content = "\n".join([
            f"{_agent_label(entry, index_agent)}: {entry.get('content', '')}"
            for entry in chat_history
        ])
        
        # Extract file number from filename (used for schema matching)
        file_num = int(''.join(filter(str.isdigit, json_file)) or 0)
        
        # prompt = (
        #     "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        #     f"The problem is:  {problem} \n"
        #     f"The Answer for the problem is: {ground_truth}\n"
        #     "Identify which agent made an error, at which step, and explain the reason for the error. "
        #     "Here's the conversation:\n\n" + chat_content +
        #     "\n\nBased on this conversation, please predict the following:\n"
        #     "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        #     "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        #     '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
        #     "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        #     "3. The reason for your prediction."
        #     "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
        # )

        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem} \n"
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
        )
        
        
        all_prompts.append(prompt)
        file_numbers.append(file_num)  # Store the file number for schema lookup
        file_mapping.append(json_file)
    
    # Process all prompts using vllm with schemata
    print(f"Processing {len(all_prompts)} files using vllm with {tensor_parallel_size} GPUs...")
    responses = _run_vllm_generation_with_schemata(
        model_path=model_path,
        prompts=all_prompts,
        schemata=schemata,
        file_numbers=file_numbers,
        tensor_parallel_size=tensor_parallel_size,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Output results
    for json_file, response in zip(file_mapping, responses):
        print(f"\nPrediction for {json_file}:")
        if response:
            print(response)
        else:
            print("Failed to get prediction from vllm.")
        print("\n" + "="*50 + "\n")

def analyze_all_at_once_local(model_obj, directory_path: str, is_handcrafted: bool, model_family: str, use_parallel: bool = False, use_vllm: bool = False, model_path: str = None, tensor_parallel_size: int = 8):
    if use_vllm and model_path and model_family in ['qwen', 'qwq']:
        return analyze_all_at_once_vllm(model_path, directory_path, is_handcrafted, tensor_parallel_size)
    elif use_parallel and model_family == 'qwen' and not use_vllm:
        return analyze_all_at_once_local_parallel(model_obj, directory_path, is_handcrafted, model_family)
    print(f"\n--- Starting Local All-at-Once Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"

    for json_file in tqdm(json_files, desc=f"All-at-Once ({model_family})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history:
            continue

        chat_content = "\n".join([
            f"Step {idx}\n{_agent_label(entry, index_agent)}: {entry.get('content', '')}" for idx, entry in enumerate(chat_history)
        ])

        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
            f"The problem is:  {problem} \n"
            f"The Answer for the problem is: {ground_truth}\n"
            "Identify which agent made an error, at which step, and explain the reason for the error. "
            "Here's the conversation:\n\n" + chat_content +
            "\n\nBased on this conversation, please predict the following:\n"
            "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
            "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
            '{\n"agent a": "xx",\n"agent b": "xxxx",\n"agent c": "xxxxx",\n"agent a": "xxxxxxx"\n},\n'
            "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
            "3. The reason for your prediction."
            "Please answer in the format: Agent Name: (Your prediction)\n, Step Number: (Your prediction)\n, Reason for Mistake: (Your reason)\n."
        )

    
        system_prompt = "You are a helpful assistant skilled in analyzing conversations."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        assistant_response = _run_local_generation(model_obj, messages, model_family)

        print(f"Prediction for {json_file}:")
        if assistant_response:
            print(assistant_response)
        else:
            print("Failed to get prediction from local model.")
        print("\n" + "="*50 + "\n")

def analyze_step_by_step_vllm(model_path: str, directory_path: str, is_handcrafted: bool, tensor_parallel_size: int = 8):
    """
    Analyze conversations using vllm for step by step inference
    """
    print(f"\n--- Starting VLLM Step-by-Step Analysis ---")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"

    llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=True
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=8192,
    )

    for json_file in tqdm(json_files, desc="Step-by-Step (vLLM)"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history:
            continue

        current_conversation_history = ""
        error_found = False
        
        for idx, entry in enumerate(chat_history):
            # Initialize the tokenizer for each file to ensure proper state
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.padding_side = 'left'  # Set left padding for decoder-only model
            
            agent_name = _agent_label(entry, index_agent)
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n"
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                "Attention: Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
                {"role": "user", "content": prompt}
            ]
            
            # Format with chat schema
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Initialize vLLM for this step
            try:
                # Generate response
                outputs = llm.generate([formatted_prompt], sampling_params)
                answer = outputs[0].outputs[0].text if outputs and outputs[0].outputs else None
                
                if not answer:
                    print(f"Failed to get evaluation for step {idx} from vLLM. Stopping analysis for this file.")
                    error_found = True
                    break
                    
                if answer.lower().strip().startswith("1. yes"):
                    print(f"\nPrediction for {json_file}: Error found.")
                    print(f"Agent Name: {agent_name}")
                    print(f"Step Number: {idx}")
                    print(f"Reason: {answer}")
                    print("\n" + "="*50 + "\n")
                    error_found = True
                    break
                
            except Exception as e:
                print(f"Error during vLLM step-by-step analysis at step {idx}: {e}")
                import traceback
                traceback.print_exc()
                error_found = True
                break
        
        if not error_found:
            print(f"\nNo errors found in {json_file} during step-by-step analysis.")
            print("\n" + "="*50 + "\n")

def analyze_step_by_step_local(model_obj, directory_path: str, is_handcrafted: bool, model_family: str, use_vllm: bool = False, model_path: str = None, tensor_parallel_size: int = 8):
    if use_vllm and model_path and model_family in ['qwen', 'qwq']:
        return analyze_step_by_step_vllm(model_path, directory_path, is_handcrafted, tensor_parallel_size)
    print(f"\n--- Starting Local Step-by-Step Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"

    for json_file in tqdm(json_files, desc=f"Step-by-Step ({model_family})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history:
            continue

        current_conversation_history = ""
        error_found = False
        for idx, entry in enumerate(chat_history):
            agent_name = _agent_label(entry, index_agent)
            content = entry.get('content', '')
            current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

            prompt = (
                f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
                f"The Answer for the problem is: {ground_truth}\n"
                f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
                f"The most recent step ({idx}) was by '{agent_name}'.\n"
                "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
                "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
                "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
                "Attention: Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
            )

            system_prompt = "You are a helpful assistant skilled in analyzing conversations."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            answer = _run_local_generation(model_obj, messages, model_family)

            if not answer:
                print("Failed to get evaluation for this step from local model. Stopping analysis for this file.")
                error_found = True
                break

            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {json_file}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx}")
                try:
                    reason = answer.split('Reason:', 1)[-1].strip()
                except:
                    reason = "[Could not extract reason]"
                print(f"Reason provided by LLM: {reason}")
                error_found = True
                break
            elif answer.lower().strip().startswith("1. no"):
                pass
            else:
                print(f"Warning: Unexpected response format from local LLM for step {idx} in {json_file}. Response: {answer[:100]}...")

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")

        print("\n" + "="*50 + "\n")


def _construct_binary_search_prompt_local(problem, answer, chat_segment_content, range_description, upper_half_desc, lower_half_desc):
     # Added answer back in based on previous logic, remove if not desired
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user's query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"The Answer for the problem is: {answer}\n"
        f"Review the following conversation segment {range_description}:\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please simply output either 'upper half' or 'lower half'. You should not output anything else."
    )

def _report_binary_search_error_local(chat_history, step, json_file, is_handcrafted):
    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"
    entry = chat_history[step]
    agent_name = _agent_label(entry, index_agent)

    print(f"\nPrediction for {json_file} (Binary Search Result):")
    print(f"Agent Name: {agent_name}")
    print(f"Step Number: {step}")
    print("\n" + "="*50 + "\n")

def _find_error_in_segment_local(model_obj, chat_history: list, problem: str, answer: str, start: int, end: int, json_file: str, is_handcrafted: bool, model_family: str):
    if start > end:
         print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
         _report_binary_search_error_local(chat_history, end if end >= 0 else 0, json_file, is_handcrafted)
         return
    if start == end:
        _report_binary_search_error_local(chat_history, start, json_file, is_handcrafted)
        return

    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Reporting start index.")
        _report_binary_search_error_local(chat_history, start, json_file, is_handcrafted)
        return

    chat_content = "\n".join([
        f"{_agent_label(entry, index_agent)}: {entry.get('content', '')}"
        for entry in segment_history
    ])

    mid = start + (end - start) // 2

    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"

    prompt = _construct_binary_search_prompt_local(problem, answer, chat_content, range_description, upper_half_desc, lower_half_desc)

   
    system_prompt = "You are a helpful assistant skilled in analyzing conversations."


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    result = _run_local_generation(model_obj, messages, model_family)

    if not result:
        print(f"Model call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
        return

    result_lower = result.lower().strip()

    if "upper half" in result_lower:
         _find_error_in_segment_local(model_obj, chat_history, problem, answer, start, mid, json_file, is_handcrafted, model_family)
    elif "lower half" in result_lower:
         new_start = min(mid + 1, end)
         _find_error_in_segment_local(model_obj, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, model_family)
    else:
        print(f"Warning: Ambiguous response '{result}' from local LLM for segment {start}-{end}. Defaulting to upper half.")
        _find_error_in_segment_local(model_obj, chat_history, problem, answer, start, mid, json_file, is_handcrafted, model_family)


def analyze_binary_search_vllm(model_path: str, directory_path: str, is_handcrafted: bool, tensor_parallel_size: int = 8):
    """
    Analyze conversations using vllm for binary search inference
    """
    print(f"\n--- Starting VLLM Binary Search Analysis ---")
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'  # Set left padding for decoder-only model

    for json_file in tqdm(json_files, desc="Binary Search (vLLM)"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "")

        if not chat_history:
            continue

        _find_error_in_segment_vllm(
            model_path=model_path,
            tokenizer=tokenizer,
            chat_history=chat_history,
            problem=problem,
            answer=answer,
            start=0,
            end=len(chat_history) - 1,
            json_file=json_file,
            is_handcrafted=is_handcrafted,
            tensor_parallel_size=tensor_parallel_size
        )

def _find_error_in_segment_vllm(model_path: str, tokenizer, chat_history: list, problem: str, answer: str, 
                               start: int, end: int, json_file: str, is_handcrafted: bool, tensor_parallel_size: int = 8):
    """
    Find the error in a segment using vLLM for binary search method
    """
    if start > end:
        print(f"Warning: Invalid range in binary search for {json_file} (start={start}, end={end}). Reporting last valid step.")
        _report_binary_search_error_local(chat_history, end if end >= 0 else 0, json_file, is_handcrafted)
        return
    if start == end:
        _report_binary_search_error_local(chat_history, start, json_file, is_handcrafted)
        return

    index_agent = "role" if _uses_role_field(is_handcrafted) else "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {json_file} (start={start}, end={end}). Reporting start index.")
        _report_binary_search_error_local(chat_history, start, json_file, is_handcrafted)
        return

    chat_content = "\n".join([
        f"{_agent_label(entry, index_agent)}: {entry.get('content', '')}"
        for entry in segment_history
    ])

    mid = start + (end - start) // 2

    range_description = f"from step {start} to step {end}"
    upper_half_desc = f"from step {start} to step {mid}"
    lower_half_desc = f"from step {mid + 1} to step {end}"

    prompt = _construct_binary_search_prompt_local(problem, answer, chat_content, range_description, upper_half_desc, lower_half_desc)

    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
        {"role": "user", "content": prompt}
    ]
    
    # Format with chat schema
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Initialize vLLM for this segment
    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=True
        )
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=1024,
        )
        
        # Generate response
        outputs = llm.generate([formatted_prompt], sampling_params)
        result = outputs[0].outputs[0].text if outputs and outputs[0].outputs else None
        
        if not result:
            print(f"Model call failed for segment {start}-{end}. Stopping binary search for {json_file}.")
            return

        result_lower = result.lower().strip()

        if "upper half" in result_lower:
            _find_error_in_segment_vllm(model_path, tokenizer, chat_history, problem, answer, start, mid, json_file, is_handcrafted, tensor_parallel_size)
        elif "lower half" in result_lower:
            new_start = min(mid + 1, end)
            _find_error_in_segment_vllm(model_path, tokenizer, chat_history, problem, answer, new_start, end, json_file, is_handcrafted, tensor_parallel_size)
        else:
            print(f"Warning: Ambiguous response '{result}' from vLLM for segment {start}-{end}. Defaulting to upper half.")
            _find_error_in_segment_vllm(model_path, tokenizer, chat_history, problem, answer, start, mid, json_file, is_handcrafted, tensor_parallel_size)
            
    except Exception as e:
        print(f"Error during vLLM binary search analysis for segment {start}-{end}: {e}")
        import traceback
        traceback.print_exc()
        return

def analyze_binary_search_local(model_obj, directory_path: str, is_handcrafted: bool, model_family: str, use_vllm: bool = False, model_path: str = None, tensor_parallel_size: int = 8):
    if use_vllm and model_path and model_family in ['qwen', 'qwq']:
        return analyze_binary_search_vllm(model_path, directory_path, is_handcrafted, tensor_parallel_size)
    print(f"\n--- Starting Local Binary Search Analysis ({model_family}) ---")
    json_files = _get_sorted_json_files(directory_path)

    for json_file in tqdm(json_files, desc=f"Binary Search ({model_family})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data:
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "")

        if not chat_history:
            continue

        _find_error_in_segment_local(
            model_obj=model_obj,
            chat_history=chat_history,
            problem=problem,
            answer=answer,
            start=0,
            end=len(chat_history) - 1,
            json_file=json_file,
            is_handcrafted=is_handcrafted,
            model_family=model_family
        )
