"""
Paper-byte-identical cloud inference helpers.

Mirrors the original paper inference code:
- baseline:  `process_single_file` + `all_at_once_parallel`
- CORRECT:   `prepare_batch_data` + `analyze_all_at_once_cloud_with_templates_parallel`

The only deliberate divergence from paper code is the Gemini SDK: we use the
new `google-genai` package with Vertex AI service-account credentials. The
prompt strings, message structure, temperature, and clean_text behavior are
byte-identical to the paper helpers.
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from google.genai.types import GenerateContentConfig, ThinkingConfig
except ImportError:
    GenerateContentConfig = None
    ThinkingConfig = None


# --- Unicode handling, copied verbatim from paper ---

def _clean_unicode_content(text):
    """Paper `Lib/utils_cloud.py::_clean_unicode_content`. Replaces smart
    quotes/dashes only — does NOT strip arbitrary non-ASCII characters."""
    if not isinstance(text, str):
        return text
    replacements = {
        '“': '"', '”': '"',
        '‘': "'", '’': "'",
        '–': '-', '—': '--',
        '…': '...',
    }
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    try:
        text = text.encode('utf-8', errors='replace').decode('utf-8')
    except UnicodeEncodeError:
        text = text.encode('ascii', errors='replace').decode('ascii')
    return text


def clean_text(text):
    """Paper `Lib/cloud_model_parallel.py::clean_text`. Aggressively replaces
    every non-ASCII character with a space (including the `•` U+2022 bullets
    used inside the schema-injection block)."""
    if text is None:
        return ""
    replacements = {
        '“': '"', '”': '"',
        '‘': "'", '’': "'",
        '–': '-', '—': '--',
        '…': '...', ' ': ' ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    cleaned = []
    for char in text:
        if ord(char) < 128:
            cleaned.append(char)
        elif char.isspace():
            cleaned.append(' ')
        else:
            cleaned.append(' ')
    return ''.join(cleaned)


# --- Multi-backend API call ---

def _make_api_call_cloud(client, model, messages, max_tokens, model_type='gpt'):
    """Paper byte-identical for OpenAI; new google-genai SDK for Gemini.

    For Gemini, the paper concatenates system+user messages into a single
    `prompt` string with `"System: {sys}\\n\\nUser: {user}\\n\\n"` framing,
    then calls `client.generate_content(prompt, ...)` with temperature=0.7.
    We preserve that exact framing and temperature; only the SDK call site
    moves to `client.models.generate_content(model=..., contents=..., config=...)`
    so we can authenticate via Vertex AI service-account credentials.
    """
    try:
        cleaned_messages = [
            {"role": m["role"], "content": _clean_unicode_content(m["content"])}
            for m in messages
        ]
        if model_type == 'gemini':
            prompt = ""
            for msg in cleaned_messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n\n"
            gemini_kwargs = dict(max_output_tokens=max_tokens, temperature=0.7)
            tb = os.environ.get("GEMINI_THINKING_BUDGET")
            if tb and ThinkingConfig is not None:
                gemini_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=int(tb))
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=GenerateContentConfig(**gemini_kwargs),
            )
            return response.text.strip() if hasattr(response, "text") and response.text else None
        else:
            if 'gpt-5' in model:
                kwargs = dict(model=model, messages=cleaned_messages, max_completion_tokens=max_tokens)
                effort = os.environ.get("OPENAI_REASONING_EFFORT")
                if effort:
                    kwargs["reasoning_effort"] = effort
                response = client.chat.completions.create(**kwargs)
            else:
                response = client.chat.completions.create(
                    model=model, messages=cleaned_messages,
                    max_tokens=max_tokens,
                )
            return response.choices[0].message.content.strip()
    except Exception as e:
        try:
            error_type = type(e).__name__
            err = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"Error during API call ({error_type}): {err}")
        except Exception:
            print("Error during API call: [details unavailable]")
        return None


# --- Helpers (sorted file listing) ---

def _get_sorted_json_files(directory_path):
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        return []


def _load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _get_json_number(json_file):
    return int(''.join(filter(str.isdigit, json_file)) or 0)


# --- Baseline (mirrors utils_cloud_parallel.py) ---

def _baseline_process_single_file(client, json_file, directory_path, is_handcrafted,
                                  model, max_tokens, model_type):
    """Verbatim from paper `utils_cloud_parallel.py::process_single_file`."""
    index_agent = "role" if is_handcrafted else "name"
    file_path = os.path.join(directory_path, json_file)
    data = _load_json_data(file_path)
    if not data:
        return json_file, None
    chat_history = data.get("history", [])
    problem = data.get("question", "")
    if not chat_history:
        return json_file, None

    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in chat_history
    ])

    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {problem}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat_content +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        """
        {
            "agent a": "xx",
            "agent b": "xxxx",
            "agent c": "xxxxx",
            "agent a": "xxxxxxx"
        },
        """
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
        {"role": "user", "content": prompt},
    ]
    time.sleep(0.1)
    result = _make_api_call_cloud(client, model, messages, max_tokens, model_type)
    return json_file, result


def all_at_once_baseline_parallel(client, directory_path, is_handcrafted, model,
                                  max_tokens, model_type='gpt',
                                  batch_size=10, max_workers=5):
    """Mirrors paper `utils_cloud_parallel.py::all_at_once_parallel`."""
    print(f"\n--- Starting All-at-Once Analysis (PARALLEL - Batch Size: {batch_size}) ---\n")
    print(f"Max concurrent workers: {max_workers}")
    json_files = _get_sorted_json_files(directory_path)
    if not json_files:
        print("No JSON files found.")
        return

    print(f"Total files to process: {len(json_files)}")
    all_results = {}
    for batch_start in tqdm(range(0, len(json_files), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(json_files))
        batch_files = json_files[batch_start:batch_end]

        print(f"\nProcessing batch: files {batch_start + 1} to {batch_end}")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_file = {
                ex.submit(_baseline_process_single_file,
                          client, jf, directory_path, is_handcrafted,
                          model, max_tokens, model_type): jf
                for jf in batch_files
            }
            for fut in tqdm(as_completed(future_to_file), total=len(future_to_file),
                            desc=f"Batch {batch_start // batch_size + 1} files"):
                jf = future_to_file[fut]
                try:
                    file_name, result = fut.result(timeout=60)
                    all_results[file_name] = result
                    print(f"Prediction for {file_name}:")
                    if result:
                        print(result)
                    else:
                        print("Failed to get prediction.")
                    print("\n" + "=" * 50 + "\n")
                except Exception as e:
                    print(f"Error processing {jf}: {e}")
                    print("\n" + "=" * 50 + "\n")
                    all_results[jf] = None
        if batch_end < len(json_files):
            print(f"Waiting 2 seconds before next batch...")
            time.sleep(2)
    return all_results


# --- CORRECT (mirrors cloud_model_parallel.py) ---

def _correct_modify_prompt_paper(prompt, schema_keys, schema_contents, wording='template'):
    """Mirror of paper `SimilarityBasedTemplateAnalyzer.modify_prompt`.

    `wording` selects the user-visible noun strings:
      - 'template' (default): paper byte-identical "THOUGHT TEMPLATE(S)" / "template(s)"
      - 'schema'             : paper's final terminology "ERROR SCHEMA(TA)" / "schema(ta)"
    Structural prompt and `•` Unicode bullets are otherwise unchanged;
    `clean_text` later replaces bullets with spaces to match paper byte
    behavior. Word change confirmed invariant for Qwen greedy decoding.
    """
    is_schema = wording == 'schema'
    NOUN_S = "schema" if is_schema else "template"  # singular noun
    NOUN_P = "schemata" if is_schema else "templates"  # plural noun
    HDR_S = "ERROR SCHEMA" if is_schema else "THOUGHT TEMPLATE"  # single header
    HDR_P = "ERROR SCHEMATA" if is_schema else "THOUGHT TEMPLATES"  # multi header

    if not schema_contents:
        return prompt + (
            "\n\n"
            f"Since no reference {NOUN_S} is available, analyze the conversation step by step:\n"
            "1. Read the entire conversation first to understand the flow\n"
            "2. Go through each step (Step 0, Step 1, Step 2, etc.) and evaluate:\n"
            "   • Is the information accurate?\n"
            "   • Is the reasoning sound?\n"
            "   • Does it advance toward the correct answer?\n"
            "3. Identify where the first error occurs\n\n"
            "Format your response as:\n"
            "Agent Name: [your prediction]\n"
            "Step Number: [step where error occurred, counting from Step 0]\n"
            "Reason for Mistake: [your explanation]\n"
        )
    parts = []
    if len(schema_contents) == 1:
        key = schema_keys[0]
        content = schema_contents[0]
        parts.append(
            f"\n\n==== {HDR_S} FOR GUIDANCE ====\n"
            f"Here is how a similar error was identified in Case #{key}:\n\n"
            f"{content}\n"
        )
        parts.append(
            "HOW TO USE THIS REFERENCE EXAMPLE:\n"
            f"This {NOUN_S} demonstrates one type of error pattern for reference. To apply it to your analysis:\n\n"
            "1. Study the ERROR PATTERN shown: What type of mistake does this example identify?\n"
            "2. Use this as reference to analyze YOUR conversation:\n"
            "   • Read through your conversation systematically (Step 0, Step 1, Step 2...)\n"
            "   • At each step, ask: 'Is there an error here, and does it match this pattern or a different one?'\n"
            "   • The error in your case may follow the same pattern or be completely different\n"
            "3. Remember this is just a reference example:\n"
            "   • Your error may occur at any step number\n"
            "   • Your error may be a different type entirely\n"
            f"   • Use this {NOUN_S} to help you recognize what errors look like, not to assume your error matches\n"
        )
    else:
        parts.append(
            f"\n\n==== {HDR_P} FOR GUIDANCE ====\n"
            f"Here are {len(schema_contents)} examples of how similar errors were identified:\n"
            f"When applying these {NOUN_P}:\n"
            "• Look for common error patterns across these examples\n"
            "• Each example shows different step numbers - focus on the ERROR TYPE, not the step position\n"
            "• Systematically check each step in your conversation (starting from Step 0)\n"
        )
        for i, (key, content) in enumerate(zip(schema_keys, schema_contents), 1):
            parts.append(
                f"\n--- Example {i} (Case #{key}) ---\n"
                f"{content}\n"
                f"--- End Example {i} ---\n"
            )
        parts.append(
            "HOW TO USE THESE REFERENCE EXAMPLES:\n"
            f"These {len(schema_contents)} examples show different error patterns for reference. For your analysis:\n\n"
            "1. Study the various error patterns demonstrated above\n"
            "2. Read through your conversation step by step (Step 0, Step 1, Step 2...)\n"
            "3. At each step, check for errors - they may match one of these patterns or be different types\n"
            "4. When you identify an error, determine if it follows a similar pattern or is a new type\n\n"
            "Important: These are reference examples only. Your conversation may contain:\n"
            "• The same type of error as shown in the examples\n"
            "• A completely different type of error not shown here\n"
            "• An error at any step number, regardless of the examples\n"
        )
    section = '\n'.join(parts)
    return (
        f"{prompt}"
        f"{section}\n\n"
        "Now analyze your conversation using these reference examples as guidance:\n"
        "1. Examine your conversation step by step (starting from Step 0)\n"
        "2. Look for errors at each step - they may match the example patterns or be different types\n"
        "3. Identify where an error occurs and what type it is\n\n"
        "Format your response as:\n"
        "Agent Name: [agent who made the error]\n"
        "Step Number: [step where error occurred, counting from Step 0]\n"
        "Reason for Mistake: [explain the error - may match example patterns or be a different type]\n"
    )


def _correct_prepare_one(json_file, directory_path, is_handcrafted, schema_analyzer, num_schemata, wording='template'):
    """Mirrors paper `cloud_model_parallel.py::prepare_batch_data` for one
    file. Returns (json_file, cleaned_prompt, schema_keys)."""
    index_agent = "role" if is_handcrafted else "name"
    file_path = os.path.join(directory_path, json_file)
    data = _load_json_data(file_path)
    if not data:
        return json_file, None, []
    chat_history = data.get("history", [])
    problem = data.get("question", "")
    if not chat_history:
        return json_file, None, []

    chat_content = "\n".join([
        f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
        for entry in chat_history
    ])
    original_prompt = (
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
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: (Your reason)\n"
    )
    file_num = _get_json_number(json_file)
    schema_keys, schema_contents = schema_analyzer.get_similarity_based_schema(file_num, num_schemata)
    prompt = _correct_modify_prompt_paper(original_prompt, schema_keys, schema_contents, wording=wording)
    cleaned_prompt = clean_text(prompt)
    return json_file, cleaned_prompt, schema_keys


def _correct_process_one(client, model, prompt, max_tokens, model_type):
    """Mirrors paper `cloud_model_parallel.py::process_single_request`."""
    system_prompt = "You are a helpful assistant skilled in analyzing conversations."
    messages = [
        {"role": "system", "content": clean_text(system_prompt)},
        {"role": "user", "content": prompt},
    ]
    time.sleep(0.1)
    return _make_api_call_cloud(client, model, messages, max_tokens, model_type)


def analyze_all_at_once_cloud_with_schemata_parallel(client, directory_path, is_handcrafted,
                                                     model, max_tokens, model_type,
                                                     schema_analyzer, num_schemata=1,
                                                     batch_size=10, max_workers=5,
                                                     wording='template'):
    """Mirrors paper `cloud_model_parallel.py::analyze_all_at_once_cloud_with_templates_parallel`.
    `wording='template'` (default, paper byte-identical) | 'schema' (new terminology).
    """
    print(f"\n--- Starting Cloud All-at-Once Analysis with Templates (PARALLEL - Batch Size: {batch_size}) ({model_type.upper()}) ---")
    json_files = _get_sorted_json_files(directory_path)
    if not json_files:
        return
    all_results = {}
    schema_usage = {}

    print(f"Number of JSON files to process: {len(json_files)}")
    print(f"Number of templates per file: {num_schemata}")
    print(f"Batch size: {batch_size}")
    print(f"Max concurrent workers: {max_workers}")

    for batch_start in tqdm(range(0, len(json_files), batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, len(json_files))
        batch_files = json_files[batch_start:batch_end]
        print(f"\nProcessing batch: files {batch_start + 1} to {batch_end}")

        batch_data = []
        for jf in batch_files:
            jf2, prompt, keys = _correct_prepare_one(jf, directory_path, is_handcrafted,
                                                    schema_analyzer, num_schemata, wording=wording)
            if prompt is None:
                continue
            batch_data.append((jf2, prompt, keys))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_data = {
                ex.submit(_correct_process_one, client, model, prompt, max_tokens, model_type): (jf, keys)
                for jf, prompt, keys in batch_data
            }
            for fut in as_completed(future_to_data):
                jf, keys = future_to_data[fut]
                try:
                    result = fut.result(timeout=60)
                    all_results[jf] = result
                    if keys:
                        schema_usage[jf] = keys

                    print(f"\nPrediction for {jf}:")
                    if result:
                        print(result[:500] + "..." if len(result) > 500 else result)
                    else:
                        print("Failed to get prediction from cloud model.")
                except Exception as e:
                    print(f"\nError processing {jf}: {e}")
                    all_results[jf] = None
        if batch_end < len(json_files):
            print(f"Waiting 2 seconds before next batch...")
            time.sleep(2)

    print("\n" + "=" * 50 + "\n")
    print("\n=== TEMPLATE USAGE SUMMARY ===")
    print(f"Total files processed: {len(json_files)}")
    print(f"Successful predictions: {sum(1 for r in all_results.values() if r is not None)}")
    print(f"Failed predictions: {sum(1 for r in all_results.values() if r is None)}")
    print(f"Files with templates applied: {len(schema_usage)}")
    print(f"\nDetailed mapping:")
    for jf, keys in sorted(schema_usage.items()):
        file_num = _get_json_number(jf)
        print(f"  File {jf} (#{file_num}) -> Templates from logs: {keys}")
    print("=================================\n")
    return all_results
