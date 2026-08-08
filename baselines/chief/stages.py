"""CHIEF's six stages, factored into pure ``build_stepN`` / ``parse_stepN`` pairs.

Every prompt string and every regex here is lifted **verbatim** from the vendored
``baselines/CHIEF/CHIEF.py`` (functions ``step1_generate_subtasks`` …
``step6_predict_final_answer`` and ``build_dag_graph``). The only change is
structural: each vendored ``stepN`` interleaved *build prompt → call LLM → parse*;
here the prompt-building and parsing are split into side-effect-free functions so
that **both** execution paths can share them —

  * ``reference.py`` (per-sample): ``parse_stepN(call_model(build_stepN(...)))``
  * ``pipeline.py`` (columnar/batched): build all prompts, one batched
    ``engine.generate`` per stage, then ``parse_stepN`` each output.

Keeping a single source of truth for the prompts/regexes is what guarantees the
two paths are faithful to each other and to the original CHIEF.
"""
from __future__ import annotations

import re

# System prompt is verbatim from the vendored ``call_llm`` (CHIEF.py:72).
SYSTEM_PROMPT = "You are a helpful assistant skilled in analyzing multi-agent conversations."


def messages(prompt: str) -> list[dict]:
    """Wrap a stage prompt into the chat-message list ``PromptEngine`` expects."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Vendored helpers (CHIEF.py:35-44) — verbatim
# ─────────────────────────────────────────────────────────────────────────────

def normalize_agent(x):
    return re.sub(r"\s+", " ", str(x)).strip().lower() if x else None


def normalize_step(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RAG block formatting (from step1) — verbatim
# ─────────────────────────────────────────────────────────────────────────────

def format_rag_blocks(rag_results) -> str:
    """Format retriever hits into the ``rag_text`` string step-1 injects.

    Verbatim from ``step1_generate_subtasks`` (CHIEF.py:87-97).
    """
    rag_blocks = []
    for i, r in enumerate(rag_results):
        if r.get("source") == "GAIA":
            rag_blocks.append(
                f"[RAG Example {i + 1}]\nQuestion: {r.get('question')}\nAnnotated Steps:\n{r.get('steps')}"
            )
        else:
            rag_blocks.append(
                f"[RAG Example {i + 1}]\nText:\n{r.get('text')}"
            )
    return "\n\n--- Retrieved Example ---\n\n".join(rag_blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — subtasks
# ─────────────────────────────────────────────────────────────────────────────

def build_step1(history_text, question, ground_truth, rag_text=None) -> str:
    """Build the stage-1 prompt.

    When ``rag_text`` is a string (RAG enabled) the prompt is byte-identical to the
    vendored one. When ``rag_text is None`` (RAG disabled, e.g. off-domain datasets)
    the retrieved-example section is omitted.
    """
    head = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real-world problem.\n"
        f"The problem is: {question}\n"
        f"The correct answer for the problem is: {ground_truth}\n\n"
        "Here is the conversation in JSON format:\n"
        + str(history_text)
        + f"\n\nThere are total {len(history_text)} steps, each entry provides the agent output and its role.\n\n"
    )
    if rag_text is not None:
        middle = (
            "Here is the retrieved reference example:\n"
            + rag_text
            + "\n\nBased on this conversation and retrieved example, please decompose the reasoning into semantic subtasks.\n"
        )
    else:
        middle = (
            "Based on this conversation, please decompose the reasoning into semantic subtasks.\n"
        )
    tail = (
        "You must perform a self-reflection process to optimize your decomposition:\n"
        "1. Draft Optimization: propose an initial set of subtasks.\n"
        "2. Evidence Alignment: ensure each subtask’s step range aligns with the dialogue.\n"
        "3. Final Optimization: ensure step ranges are continuous, cover all steps, and do NOT overlap.\n\n"
        "Output format (plain text, NO markdown, NO bullets, NO '**'):\n"
        "The Subtask Name: <your prediction>\n"
        "Step Range: <start-end>    # use integer indices like 0-2, without the word 'step'\n"
        "The Oracle: <your prediction>\n"
        "Evidence: <one-line summary of key supporting steps>\n"
        "Loop Info:\n"
        "{\n"
        "  is_loop_related: true/false,\n"
        "  loop_role: entry/internal/exit/none,\n"
        "  loop_group_id: L1/L2/... or null,\n"
        "  reversibility: reversible/partial/irreversible,\n"
        "  loop_risk_score: float(0-1)\n"
        "}\n\n"
        "Loop detection rule: If steps show repeated reasoning patterns, retrying similar approaches, or cyclic agent/tool calls, mark them as loop-related. Otherwise set loop_role = none and loop_group_id = null.\n\n"
        "Important:\n"
        "- Use ONLY the above fields and labels.\n"
        "- Do NOT add markdown formatting or extra labels.\n"
        "- Step ranges must be contiguous, cover all steps, and must NOT overlap.\n"
        "Now generate your optimized output below:\n"
    )
    return head + middle + tail


def parse_step1(result_text):
    """Parse stage-1 output into subtasks. Verbatim from CHIEF.py:136-186."""
    result_text = result_text.strip()

    name_pattern = r"The Subtask Name:\s*([^\n]+)"
    range_pattern = r"Step Range:\s*(?:step)?(\d+)\s*-\s*(?:step)?(\d+)"
    oracle_pattern = r"The Oracle:\s*([^\n]+)"
    evidence_pattern = r"Evidence:\s*([^\n]+)"
    loop_block_pattern = r"Loop Info:\s*\{([\s\S]+?)\}"

    names = re.findall(name_pattern, result_text)
    range_matches = re.findall(range_pattern, result_text)
    ranges = [f"{s}-{e}" for (s, e) in range_matches]
    oracles = re.findall(oracle_pattern, result_text)
    evidences = re.findall(evidence_pattern, result_text)
    loop_blocks = re.findall(loop_block_pattern, result_text)

    loop_infos = []
    for blk in loop_blocks:
        def extract(pattern):
            m = re.search(pattern, blk)
            return m.group(1).strip() if m else None

        loop_infos.append({
            "is_loop_related": extract(r"is_loop_related:\s*(true|false)") == "true",
            "loop_role": extract(r"loop_role:\s*([a-zA-Z]+)") or "none",
            "loop_group_id": extract(r"loop_group_id:\s*([A-Za-z0-9]+|null)") or "null",
            "reversibility": extract(r"reversibility:\s*([a-zA-Z]+)") or "reversible",
            "loop_risk_score": float(extract(r"loop_risk_score:\s*([0-9]*\.?[0-9]+)") or 0.0)
        })

    subtasks = []
    n_subtasks = min(len(names), len(ranges), len(oracles), len(evidences))
    for i in range(n_subtasks):
        n = names[i]
        r = ranges[i]
        o = oracles[i]
        e = evidences[i]
        loop_info = loop_infos[i] if i < len(loop_infos) else {
            "is_loop_related": False,
            "loop_role": "none",
            "loop_group_id": None,
            "reversibility": "reversible",
            "loop_risk_score": 0.0
        }

        subtasks.append({
            "id": f"S{i + 1}",
            "name": n.strip(),
            "step_range": r.strip(),
            "oracle": o.strip(),
            "evidence": e.strip(),
            "loop_info": loop_info
        })
    return subtasks


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — subtask edges
# ─────────────────────────────────────────────────────────────────────────────

def build_step2(history_text, question, ground_truth, subtasks) -> str:
    """Verbatim from CHIEF.py:189-233."""
    return (
        "You are an expert in causal reasoning and multi-agent task analysis.\n"
        f"The problem is: {question}\n"
        f"The correct answer for the problem is: {ground_truth}\n\n"
        "Here is the conversation in JSON format:\n"
        + str(history_text)
        + f"\n\nThere are total {len(history_text)} steps.\n\n"
        "Here are the subtasks with their IDs (in execution order):\n"
        + str(subtasks)
        + "\n\nNow, construct causal edges ONLY for consecutive subtask pairs: (S1->S2), (S2->S3), ...\n"
        "For each consecutive pair (Si, S(i+1)), you MUST output ONE block in the following exact format:\n\n"
        "From: <upstream subtask id, e.g., S1>\n"
        "To: <downstream subtask id, e.g., S2>\n"
        "Type: data_dependency or logical_prereq\n"
        "Strength: 0.xx\n"
        "Explanation: one-line explanation of why the downstream subtask depends on the upstream subtask.\n"
        "Data_Transfer:\n"
        "{\n"
        "  upstream_output:\n"
        "  - data_item: \"...\"\n"
        "    data_type: \"numeric/text/list/boolean\"\n"
        "    producer_steps: [0, 1, 2]   # integer step indices, no 'step' prefix, no quotes\n"
        "    producer_agents: [\"agent1\", \"agent2\"]\n"
        "  downstream_usage:\n"
        "  - data_item: \"...\"\n"
        "    consumer_steps: [3, 4]      # integer step indices, no 'step' prefix, no quotes\n"
        "    consumer_agents: [\"agentY\"]\n"
        "    usage_type: \"calculation/condition/retrieval/reference\"\n"
        "    correctness: \"correct/misinterpreted/misused\"\n"
        "  consistency_score: 0.xx\n"
        "}\n"
        "Failure Modes:\n"
        "- type: loop_issue/data_issue/irrecoverability_issue\n"
        "  description: ...\n"
        "  severity: 0.xx\n"
        "  if_from_wrong: ...\n"
        "  likely_effect_on_to: ...\n"
        "- type: ... (you may add more items, or output no items if there is no failure mode)\n\n"
        "Guidelines:\n"
        "- Use ONLY the subtask ids listed above (S1, S2, ...) in the 'From' and 'To' fields. Do NOT use step numbers like 'step0'.\n"
        "- For Data_Transfer, identify key information produced in the upstream subtask and how it is used in the downstream subtask.\n"
        "- You may output zero, one, or multiple '- data_item' entries under upstream_output and downstream_usage.\n"
        "- Each edge block must follow the exact field names and structure above.\n"
        "- No extra commentary, no markdown.\n"
    )


def parse_step2(raw):
    """Parse stage-2 output. Verbatim from CHIEF.py:235-398."""
    raw = raw.strip()

    edge_blocks = re.split(r"(?=^From:\s*S[0-9]+)", raw, flags=re.M)
    edge_blocks = [blk.strip() for blk in edge_blocks if blk.strip()]

    edges = []

    for blk in edge_blocks:
        from_match = re.search(r"^From:\s*(S[0-9]+)", blk, flags=re.M)
        to_match = re.search(r"^To:\s*(S[0-9]+)", blk, flags=re.M)
        type_match = re.search(r"^Type:\s*([^\n]+)", blk, flags=re.M)
        strength_match = re.search(r"^Strength:\s*([0-9]*\.?[0-9]+)", blk, flags=re.M)
        explanation_match = re.search(r"^Explanation:\s*(.*)", blk, flags=re.M)

        from_id = from_match.group(1).strip() if from_match else ""
        to_id = to_match.group(1).strip() if to_match else ""
        etype = type_match.group(1).strip() if type_match else ""
        strength = float(strength_match.group(1)) if strength_match else 0.0
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        # -------- Data_Transfer block --------
        dt_match = re.search(
            r"Data_Transfer:\s*\{([\s\S]+?)\}\s*^Failure Modes:",
            blk,
            flags=re.S | re.M
        )
        dt_text = dt_match.group(1) if dt_match else ""

        uo_match = re.search(
            r"upstream_output:\s*([\s\S]+?)^\s*downstream_usage:",
            dt_text,
            flags=re.S | re.M
        )
        uo_text = uo_match.group(1) if uo_match else ""
        upstream_output = []

        for item_block in re.split(r"(?=^\s*-\s*data_item:\s*)", uo_text, flags=re.M):
            item_block = item_block.strip()
            if not item_block:
                continue
            data_item_m = re.search(r"data_item:\s*\"([^\"]+)\"", item_block)
            if not data_item_m:
                continue
            data_type_m = re.search(r"data_type:\s*\"([^\"]+)\"", item_block)
            prod_steps_m = re.search(r"producer_steps:\s*\[([^\]]*)\]", item_block)
            prod_agents_m = re.search(r"producer_agents:\s*\[([^\]]*)\]", item_block)

            producer_steps = []
            if prod_steps_m:
                step_str = prod_steps_m.group(1)
                for x in step_str.split(","):
                    m_num = re.search(r"(\d+)", x)
                    if m_num:
                        producer_steps.append(int(m_num.group(1)))

            producer_agents = []
            if prod_agents_m:
                agents_str = prod_agents_m.group(1)
                for a in agents_str.split(","):
                    a = a.strip().strip('"').strip("'")
                    if a:
                        producer_agents.append(a)

            upstream_output.append({
                "data_item": data_item_m.group(1),
                "data_type": data_type_m.group(1) if data_type_m else "",
                "producer_steps": producer_steps,
                "producer_agents": producer_agents
            })

        du_match = re.search(
            r"downstream_usage:\s*([\s\S]+?)^\s*consistency_score:",
            dt_text,
            flags=re.S | re.M
        )
        du_text = du_match.group(1) if du_match else ""
        downstream_usage = []

        for item_block in re.split(r"(?=^\s*-\s*data_item:\s*)", du_text, flags=re.M):
            item_block = item_block.strip()
            if not item_block:
                continue
            data_item_m = re.search(r"data_item:\s*\"([^\"]+)\"", item_block)
            if not data_item_m:
                continue
            cons_steps_m = re.search(r"consumer_steps:\s*\[([^\]]*)\]", item_block)
            cons_agents_m = re.search(r"consumer_agents:\s*\[([^\]]*)\]", item_block)
            usage_type_m = re.search(r"usage_type:\s*\"([^\"]+)\"", item_block)
            correctness_m = re.search(r"correctness:\s*\"([^\"]+)\"", item_block)

            consumer_steps = []
            if cons_steps_m:
                step_str = cons_steps_m.group(1)
                for x in step_str.split(","):
                    m_num = re.search(r"(\d+)", x)
                    if m_num:
                        consumer_steps.append(int(m_num.group(1)))

            consumer_agents = []
            if cons_agents_m:
                agents_str = cons_agents_m.group(1)
                for a in agents_str.split(","):
                    a = a.strip().strip('"').strip("'")
                    if a:
                        consumer_agents.append(a)

            downstream_usage.append({
                "data_item": data_item_m.group(1),
                "consumer_steps": consumer_steps,
                "consumer_agents": consumer_agents,
                "usage_type": usage_type_m.group(1) if usage_type_m else "",
                "correctness": correctness_m.group(1) if correctness_m else ""
            })

        # == consistency_score ==
        cs_match = re.search(r"consistency_score:\s*([0-9]*\.?[0-9]+)", dt_text)
        consistency_score = float(cs_match.group(1)) if cs_match else 0.0

        # -------- Failure Modes block --------
        fm_match = re.search(
            r"Failure Modes:\s*([\s\S]+)$",
            blk,
            flags=re.S | re.M
        )
        fm_text = fm_match.group(1) if fm_match else ""
        failure_modes = []

        for fm_block in re.split(r"(?=^\s*-\s*type:\s*)", fm_text, flags=re.M):
            fm_block = fm_block.strip()
            if not fm_block:
                continue

            type_m = re.search(r"type:\s*([a-zA-Z_]+)", fm_block)
            if not type_m:
                continue

            desc_m = re.search(r"description:\s*(.*)", fm_block)
            sev_m = re.search(r"severity:\s*([0-9]*\.?[0-9]+)", fm_block)
            if_from_m = re.search(r"if_from_wrong:\s*(.*)", fm_block)
            effect_m = re.search(r"likely_effect_on_to:\s*(.*)", fm_block)

            failure_modes.append({
                "type": type_m.group(1).strip(),
                "description": desc_m.group(1).strip() if desc_m else "",
                "severity": float(sev_m.group(1)) if sev_m else 0.0,
                "if_from_wrong": if_from_m.group(1).strip() if if_from_m else "",
                "likely_effect_on_to": effect_m.group(1).strip() if effect_m else ""
            })

        edges.append({
            "from": from_id,
            "to": to_id,
            "type": etype,
            "strength": strength,
            "explanation": explanation,
            "failure_modes": failure_modes,
            "data_transfer": {
                "upstream_output": upstream_output,
                "downstream_usage": downstream_usage,
                "consistency_score": consistency_score
            }
        })

    return {"subtasks_edges": edges}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — agents within subtasks
# ─────────────────────────────────────────────────────────────────────────────

def build_step3(history_text, question, ground_truth, subtasks) -> str:
    """Verbatim from CHIEF.py:402-448."""
    subtask_lines = []
    for s in subtasks:
        subtask_lines.append(
            f"- id: {s.get('id')}, name: {s.get('name')}, step_range: {s.get('step_range')}"
        )
    subtasks_text = "\n".join(subtask_lines)

    return (
        "You are an AI assistant tasked with analyzing multi-agent execution traces.\n"
        f"The problem is: {question}\n"
        f"The correct answer for the problem is: {ground_truth}\n\n"
        "Here is the conversation in JSON format:\n"
        + str(history_text)
        + f"\n\nThere are total {len(history_text)} steps, each entry provides the output of the agent and its role.\n\n"
        "Below are the subtasks :\n"
        + str(subtasks_text)
        + "\n\nYour job:\n"
          "1. For each subtask, identify the agents that actively perform actions within its step_range.\n"
          "2. Summarize each agent's behavior into Action / Observation / Thought / Result.\n"
          "3. Within each subtask, identify step-level data flows: which step produces a data_item and which later step(s) consume it.\n\n"
        "For EACH subtask, you MUST answer in the following strict format:\n\n"
        "The Subtask Name: (must exactly match one of the given subtask names)\n"
        "Agents:\n"
        "- Agent: <agent_name>\n"
        "  -- Action: <what this agent did>\n"
        "  -- Observation: <what they saw>\n"
        "  -- Thought: <their reasoning>\n"
        "  -- Result: <their output>\n"
        "(repeat '- Agent' blocks if multiple agents in this subtask)\n"
        "Data_Flow:\n"
        "- from_step: <integer step id where the data is produced>\n"
        "  to_step: <integer step id where the data is used>\n"
        "  source_agent: <agent name in the from_step>\n"
        "  target_agent: <agent name in the to_step>\n"
        "  data_item: \"short description of the data (e.g., 'distance=30' or 'parsed_goal')\"\n"
        "  data_type: \"text/numeric/list/boolean\"\n"
        "  transformation: \"how the data is transformed (e.g., parsing/normalization/aggregation)\"\n"
        "  correctness: \"correct/misinterpreted/misused\"\n"
        "  confidence: 0.xx\n"
        "(repeat '- from_step' blocks if there are multiple data flows; if no data flow, you may output 'Data_Flow: []')\n\n"
        "Guidelines:\n"
        "- Only use steps inside the subtask's step_range when constructing Agents and Data_Flow.\n"
        "- Data_Flow should capture meaningful data passing from one step to another (within the same subtask).\n"
        "- You may output zero, one, or multiple Data_Flow items.\n"
        "- Do NOT add extra fields. Use only the structure given above.\n\n"
        "Now generate your output for ALL subtasks, one subtask block after another. No extra commentary.\n"
    )


def parse_step3(raw, subtasks):
    """Parse stage-3 output and merge into subtasks. Verbatim from CHIEF.py:451-566."""
    result_text = raw.strip()

    sections = re.split(r"^The Subtask Name:\s*", result_text, flags=re.M)
    sections = [sec.strip() for sec in sections if sec.strip()]
    parsed_by_name = {}

    for sec in sections:
        first_newline = sec.find("\n")
        if first_newline == -1:
            subtask_name = sec.strip()
            rest = ""
        else:
            subtask_name = sec[:first_newline].strip()
            rest = sec[first_newline + 1:]

        agents = []
        agents_match = re.search(
            r"Agents:\s*([\s\S]*?)(?=^Data_Flow:|\Z)",
            rest,
            flags=re.M,
        )
        agents_text = agents_match.group(1) if agents_match else ""

        agent_blocks = re.findall(
            r"-\s*Agent:\s*(.*?)\s*"
            r"--\s*Action:\s*(.*?)\s*"
            r"--\s*Observation:\s*(.*?)\s*"
            r"--\s*Thought:\s*(.*?)\s*"
            r"--\s*Result:\s*(.*?)(?=\n-\s*Agent:|\nData_Flow:|\Z)",
            agents_text,
            flags=re.DOTALL,
        )

        for a, act, obs, th, res in agent_blocks:
            agents.append({
                "agent": a.strip(),
                "action": act.strip(),
                "observation": obs.strip(),
                "thought": th.strip(),
                "result": res.strip(),
            })

        data_flow = []
        df_match = re.search(
            r"Data_Flow:\s*([\s\S]+)$",
            rest,
            flags=re.M,
        )
        if df_match:
            df_text = df_match.group(1).strip()
            if not re.match(r"^\[\s*\]$", df_text):
                flow_blocks = re.split(
                    r"(?=^\s*-\s*from_step:\s*)",
                    df_text,
                    flags=re.M,
                )
                for fb in flow_blocks:
                    fb = fb.strip()
                    if not fb or not fb.startswith("-"):
                        continue

                    def ex_int(pattern, _fb=fb):
                        m = re.search(pattern, _fb)
                        if m and m.group(1).strip().isdigit():
                            return int(m.group(1).strip())
                        return None

                    def ex_str(pattern, _fb=fb):
                        m = re.search(pattern, _fb)
                        return m.group(1).strip() if m else ""

                    from_step = ex_int(r"from_step:\s*([0-9]+)")
                    to_step = ex_int(r"to_step:\s*([0-9]+)")
                    source_agent = ex_str(r"source_agent:\s*([^\n]+)")
                    target_agent = ex_str(r"target_agent:\s*([^\n]+)")
                    data_item = ex_str(r"data_item:\s*\"([^\"]+)\"")
                    data_type = ex_str(r"data_type:\s*\"([^\"]+)\"")
                    transformation = ex_str(r"transformation:\s*\"([^\"]+)\"")
                    correctness = ex_str(r"correctness:\s*\"([^\"]+)\"")
                    conf_match = re.search(r"confidence:\s*([0-9]*\.?[0-9]+)", fb)
                    confidence = float(conf_match.group(1)) if conf_match else 0.0

                    if from_step is not None and to_step is not None:
                        data_flow.append({
                            "from_step": from_step,
                            "to_step": to_step,
                            "source_agent": source_agent,
                            "target_agent": target_agent,
                            "data_item": data_item,
                            "data_type": data_type,
                            "transformation": transformation,
                            "correctness": correctness,
                            "confidence": confidence,
                        })

        parsed_by_name[subtask_name] = {
            "agents": agents,
            "data_flow": data_flow,
        }

    subtasks_agents = []
    for s in subtasks:
        name = s.get("name") or s.get("Subtask Name")
        parsed = parsed_by_name.get(name, {"agents": [], "data_flow": []})
        merged = {
            "id": s.get("id"),
            "name": name,
            "step_range": s.get("step_range"),
            "oracle": s.get("oracle"),
            "evidence": s.get("evidence"),
            "loop_info": s.get("loop_info"),
            "agents": parsed["agents"],
            "data_flow": parsed["data_flow"],
        }
        subtasks_agents.append(merged)

    return subtasks_agents


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — agent edges within subtasks
# ─────────────────────────────────────────────────────────────────────────────

def build_step4(history_text, question, ground_truth, subtasks_agents) -> str:
    """Verbatim from CHIEF.py:569-614."""
    subtask_agent_lines = []
    for s in subtasks_agents:
        agent_names = [a.get("agent") or a.get("Agent") or "" for a in s.get("agents", [])]
        agent_names = [a for a in agent_names if a]
        subtask_agent_lines.append(
            f"- id: {s.get('id')}, name: {s.get('name')}, step_range: {s.get('step_range')}, agents: {agent_names}"
        )

    subtasks_agents_text = "\n".join(subtask_agent_lines)

    return (
        "You are an expert in causal reasoning and multi-agent task analysis.\n"
        f"The problem is: {question}\n"
        f"The correct answer for the problem is: {ground_truth}\n\n"
        "Here is the conversation in JSON format:\n"
        + str(history_text)
        + f"\n\nThere are total {len(history_text)} steps, each entry provides the output of the agent and its role.\n\n"
        "Below are the subtasks with their agents:\n"
        + str(subtasks_agents_text)
        + "\n\nYour job:\n"
          "- For each subtask, construct causal edges BETWEEN agents inside this subtask only.\n"
          "- Do NOT create edges across different subtasks.\n"
          "- Use the agent-level DAG to describe how observations, reasoning, and decisions flow from one agent to another.\n\n"
        "For EACH subtask, you MUST output in the following strict format:\n\n"
        "The Subtask Name: <must exactly match one of the given subtask names>\n"
        "Agents_edges:\n"
        "- From_agent: <agent name>\n"
        "  To_agent: <agent name>\n"
        "  agent_dependency_type: obs_dependency / reasoning_continuation / decision_dependency / environment_feedback / memory_ref / loop_control\n"
        "  agent_strength: 0.xx\n"
        "  agent_explanation: one or two sentences explaining why this dependency exists, with specific referenced steps.\n"
        "  agent_failure_modes:\n"
        "    - type: loop_issue / data_issue / irrecoverability_issue\n"
        "      description: short description of the potential failure along this edge.\n"
        "      severity: 0.xx\n"
        "      if_from_agent_wrong: what happens if the FROM agent makes a wrong decision or produces wrong information here.\n"
        "      likely_effect_on_to_agent: how this will affect the TO agent's reasoning or decision.\n"
        "    - type: ... (optional additional items; you may output zero, one, or multiple failure modes)\n\n"
        "Guidelines for failure mode types:\n"
        "- loop_issue: use this when the FROM agent's behavior may cause entering, reinforcing, or exiting an unnecessary loop within this subtask.\n"
        "- data_issue: use this when the FROM agent misinterprets, fabricates, or misuses data that is passed to the TO agent.\n"
        "- irrecoverability_issue: use this when the FROM agent's mistake makes it hard or impossible for downstream agents to recover the correct reasoning path.\n"
        "- You may output zero, one, or multiple agent_failure_modes for each edge, but each 'type' must be one of: loop_issue, data_issue, irrecoverability_issue.\n"
        "- Do NOT add extra fields beyond the structure given above.\n\n"
        "Now generate your output for ALL subtasks, one subtask block after another. No extra commentary.\n"
    )


def parse_step4(raw):
    """Parse stage-4 output. Verbatim from CHIEF.py:617-700."""
    result_text = raw.strip()

    all_subtask_edges = []

    sections = re.split(r"^The Subtask Name:\s*", result_text, flags=re.M)
    sections = [sec.strip() for sec in sections if sec.strip()]

    for i, sec in enumerate(sections):
        first_newline = sec.find("\n")
        if first_newline == -1:
            subtask_name = sec.strip()
            rest = ""
        else:
            subtask_name = sec[:first_newline].strip()
            rest = sec[first_newline + 1:]

        edges_match = re.search(r"Agents_edges:\s*([\s\S]+)$", rest, flags=re.M)
        edges_text = edges_match.group(1) if edges_match else ""
        agents_edges = []

        edge_blocks = re.split(r"(?=^\s*-\s*From_agent:\s*)", edges_text, flags=re.M)
        for blk in edge_blocks:
            blk = blk.strip()
            if not blk or not blk.startswith("-"):
                continue

            def ex_str(pattern, text=blk):
                m = re.search(pattern, text)
                return m.group(1).strip() if m else ""

            def ex_float(pattern, text=blk):
                m = re.search(pattern, text)
                return float(m.group(1)) if m else 0.0

            from_agent = ex_str(r"From_agent:\s*([^\n]+)")
            to_agent = ex_str(r"To_agent:\s*([^\n]+)")
            dep_type = ex_str(r"agent_dependency_type:\s*([^\n]+)")
            strength = ex_float(r"agent_strength:\s*([0-9]*\.?[0-9]+)")
            explanation = ex_str(r"agent_explanation:\s*(.*)")

            # ---- agent_failure_modes ----
            failure_modes = []
            fm_match = re.search(r"agent_failure_modes:\s*([\s\S]+)", blk)
            if fm_match:
                fm_text = fm_match.group(1)
                fm_blocks = re.split(r"(?=^\s*-\s*type:\s*)", fm_text, flags=re.M)
                for fb in fm_blocks:
                    fb = fb.strip()
                    if not fb.startswith("-"):
                        continue

                    type_m = re.search(r"type:\s*([a-zA-Z_]+)", fb)
                    if not type_m:
                        continue
                    fm_type = type_m.group(1).strip()

                    desc_m = re.search(r"description:\s*(.*)", fb)
                    sev_m = re.search(r"severity:\s*([0-9]*\.?[0-9]+)", fb)
                    if_from_m = re.search(r"if_from_agent_wrong:\s*(.*)", fb)
                    effect_m = re.search(r"likely_effect_on_to_agent:\s*(.*)", fb)

                    failure_modes.append({
                        "type": fm_type,
                        "description": desc_m.group(1).strip() if desc_m else "",
                        "severity": float(sev_m.group(1)) if sev_m else 0.0,
                        "if_from_agent_wrong": if_from_m.group(1).strip() if if_from_m else "",
                        "likely_effect_on_to_agent": effect_m.group(1).strip() if effect_m else ""
                    })

            agents_edges.append({
                "From_agent": from_agent,
                "To_agent": to_agent,
                "agent_dependency_type": dep_type,
                "agent_strength": strength,
                "agent_explanation": explanation,
                "agent_failure_modes": failure_modes
            })

        all_subtask_edges.append({
            "The Subtask Name": subtask_name,
            "Agents_edges": agents_edges
        })

    return all_subtask_edges


# ─────────────────────────────────────────────────────────────────────────────
# DAG assembly (CHIEF.py:999-1005) — verbatim
# ─────────────────────────────────────────────────────────────────────────────

def build_dag_graph(subtasks_agents, subtasks_edges, all_subtask_edges):
    return {
        "subtasks": subtasks_agents,
        "subtask_edges": subtasks_edges,
        "agent_edges": all_subtask_edges
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — candidate error set
# ─────────────────────────────────────────────────────────────────────────────

def build_step5(history_text, question, ground_truth, dag_graph) -> str:
    """Verbatim from CHIEF.py:703-760."""
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation solving a real-world problem.\n"
        f"The problem is: {question}\n"
        f"The correct answer for the problem is: {ground_truth}\n\n"
        "Here is the conversation:\n"
        + str(history_text)
        + f"\n\nThere are total {len(history_text)} steps, each entry provides an agent's output.\n\n"
        "Here is the graph describing the reasoning structure:\n"
        + str(dag_graph)
        + "\n\nYour job:\n"
        "1. Review the entire reasoning trace holistically.\n"
        "2. Use the graph to understand high-level structure:\n"
        "   - Use subtask.loop_info and agent_edges to detect loops (entry/internal/exit) and how agents contribute to entering or exiting loops.\n"
        "   - Use data_transfer (in subtask_edges) and step-level data flows (if available) to trace how key data items are produced and consumed.\n"
        "3. Do a chain-of-thought reasoning based on the following three rules for error localization:\n"
        "   - Rule1. When there is a loop in the trajectory, first determine whether the loop is a reasonable exploration at that time; Responsibility should only be attributed to the decision that led to the entry into the cycle when it is determined that it should not have entered the cycle. Otherwise, responsibility should fall on actions that cause irreversible consequences when entering or exiting the cycle\n"
        "   - Rule2. Trace the source of each key piece of information used in the final conclusion: if the information comes from upstream but is misinterpreted or incorrectly inferred, blame the current executor; If the information is directly generated without any upstream basis, the responsibility lies with the executor who generated the information; If the information is correct but later misused, it is attributed to the node where the misuse occurred.\n"
        "   - Rule3. The final error should be attributed to the first node that prevents the correct answer path from being restored through conventional means, rather than the position where the deviation occurred earliest.\n"
        "4. Decide:\n"
        " - Which agents appear to have made reasoning slips or incorrect assumptions (list them as candidate error agents).\n"
        " - Identify up to **likely subtasks** (the key mistake subtasks) and list them as candidate error subtasks.\n"
        " - Referencing the candidate subtask set and candidate agent set, jointly infer to determine the candidate step set(at least 5 candidates), and create step nodes for each candidate.\n\n"
        "Now, please strictly follow this output format:\n\n"
        "Candidate Error Subtasks: [Id1, Id2, ...]\n"
        "Candidate Error Agents: [agent1, agent2, ...]\n"
        "Candidate Error Steps:\n"
        "- step_id: <integer id in conversation>\n"
        "  agent_in_step: [agent1, agent2, ...]\n"
        "  loop_issue:\n"
        "    is_in_loop: true/false\n"
        "    loop_role: entry/internal/exit/none\n"
        "    loop_group_id: L1/L2/... or null\n"
        "  data_issue:\n"
        "    has_issue: true/false\n"
        "    data_item: \"short description of the key data item or 'none' if no data issue\"\n"
        "    source_step: <integer step id or null>\n"
        "    consistency_score: 0.xx\n"
        "    explanation: short explanation of whether the data is fabricated, misinterpreted, or misused.\n"
        "  irrecoverability_issue:\n"
        "    is_irrecoverable: true/false\n"
        "    reason: short explanation of whether this step blocks the recovery of the correct path.\n"
        "  impact:\n"
        "    affected_steps: [step_id_1, step_id_2, ...]\n"
        "    impact_score: 0.xx\n"
        "  information:\n"
        "    input: <input to the step>\n"
        "    output: <output from the step>\n"
        "  confidence: 0.xx\n"
        "...\n\n"
        "Guidelines:\n"
        "- Use the three rules to assign loop_issue, data_issue, and irrecoverability_issue for each candidate step.\n"
        "- loop_issue should be true only if this step is part of a loop (entry/internal/exit) according to loop_info and agent_edges.\n"
        "- data_issue should be true only if this step directly fabricates, misinterprets, or misuses key data items traced through the DAG.\n"
        "- irrecoverability_issue.is_irrecoverable should be true only if after this step, it becomes hard or impossible to restore the correct reasoning path.\n"
        "- impacted_steps should list downstream steps whose reasoning depends on this step.\n"
        "- You must output at least 5 candidate error steps.\n"
        "- No special symbols, no extra commentary, no extra fields beyond the schema above.\n"
    )


def parse_step5(response):
    """Parse stage-5 output. Verbatim from CHIEF.py:763-933."""
    result_text = response.strip()

    error_subtasks_pattern = r"Candidate Error Subtasks:\s*\[([^\]]*)\]"
    error_agents_pattern = r"Candidate Error Agents:\s*\[([^\]]*)\]"

    error_subtasks_match = re.search(error_subtasks_pattern, result_text)
    error_agents_match = re.search(error_agents_pattern, result_text)

    def parse_list_in_brackets(s):
        if not s:
            return []
        items = []
        for x in s.split(","):
            x = x.strip().strip('"').strip("'")
            if x:
                items.append(x)
        return items

    candidate_error_subtasks = parse_list_in_brackets(
        error_subtasks_match.group(1) if error_subtasks_match else ""
    )
    candidate_error_agents = parse_list_in_brackets(
        error_agents_match.group(1) if error_agents_match else ""
    )

    candidate_error_steps = []

    steps_block_match = re.search(
        r"Candidate Error Steps:\s*([\s\S]+)$",
        result_text
    )

    if steps_block_match:
        steps_block = steps_block_match.group(1)

        step_blocks = re.findall(
            r"-\s*step_id:\s*.+?(?=\n\s*-\s*step_id:|\Z)",
            steps_block,
            re.S
        )

        for block in step_blocks:

            def extract(pattern, _block=block):
                m = re.search(pattern, _block, re.S)
                return m.group(1).strip() if m else None

            def to_int(x):
                try:
                    return int(x)
                except Exception:
                    return None

            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            def to_bool(x):
                if not x:
                    return False
                x = x.strip().lower()
                if x in ["true", "yes"]:
                    return True
                if x in ["false", "no"]:
                    return False
                return False

            step_id_raw = extract(r"step_id:\s*([0-9]+)")
            step_id = to_int(step_id_raw)

            # agent_in_step: [agent1, agent2, ...]
            agent_list_raw = extract(r"agent_in_step:\s*\[([^\]]*)\]")
            agent_in_step = []
            if agent_list_raw:
                for a in agent_list_raw.split(","):
                    a = a.strip().strip('"').strip("'")
                    if a:
                        agent_in_step.append(a)

            # ---- loop_issue ----
            is_in_loop_raw = extract(r"is_in_loop:\s*(true|false)")
            loop_role = extract(r"loop_role:\s*([a-zA-Z_]+)") or "none"
            loop_group_id_raw = extract(r"loop_group_id:\s*([A-Za-z0-9_]+|null)")

            if loop_group_id_raw is None or loop_group_id_raw.lower() == "null":
                loop_group_id = None
            else:
                loop_group_id = loop_group_id_raw

            loop_issue = {
                "is_in_loop": to_bool(is_in_loop_raw),
                "loop_role": loop_role,
                "loop_group_id": loop_group_id,
            }

            # ---- data_issue ----
            has_issue_raw = extract(r"has_issue:\s*(true|false)")
            data_item = extract(r"data_item:\s*\"?([^\n\"]+)\"?")
            source_step_raw = extract(r"source_step:\s*([0-9]+|null)")
            consistency_score_raw = extract(r"consistency_score:\s*([0-9]*\.?[0-9]+)")
            explanation = extract(r"explanation:\s*([^\n]+)")

            if source_step_raw is None or source_step_raw.lower() == "null":
                source_step = None
            else:
                source_step = to_int(source_step_raw)

            data_issue = {
                "has_issue": to_bool(has_issue_raw),
                "data_item": data_item or "",
                "source_step": source_step,
                "consistency_score": to_float(consistency_score_raw),
                "explanation": explanation or "",
            }

            # ---- irrecoverability_issue ----
            is_irrecoverable_raw = extract(r"is_irrecoverable:\s*(true|false)")
            irrecover_reason = extract(r"reason:\s*([^\n]+)")

            irrecoverability_issue = {
                "is_irrecoverable": to_bool(is_irrecoverable_raw),
                "reason": irrecover_reason or "",
            }

            # ---- impact ----
            affected_steps_raw = extract(r"affected_steps:\s*\[([^\]]*)\]")
            impact_score_raw = extract(r"impact_score:\s*([0-9]*\.?[0-9]+)")

            affected_steps = []
            if affected_steps_raw:
                for x in affected_steps_raw.split(","):
                    x = x.strip()
                    if x.isdigit():
                        affected_steps.append(int(x))

            impact = {
                "affected_steps": affected_steps,
                "impact_score": to_float(impact_score_raw),
            }

            # ---- information ----
            information_input = extract(r"input:\s*([^\n]+)")
            information_output = extract(r"output:\s*([^\n]+)")

            information = {
                "input": information_input or "",
                "output": information_output or "",
            }

            # ---- confidence ----
            confidence_raw = extract(r"confidence:\s*([0-9]*\.?[0-9]+)")
            confidence = to_float(confidence_raw)

            candidate_error_steps.append({
                "step_id": step_id,
                "agent_in_step": agent_in_step,
                "loop_issue": loop_issue,
                "data_issue": data_issue,
                "irrecoverability_issue": irrecoverability_issue,
                "impact": impact,
                "information": information,
                "confidence": confidence,
            })

    return {
        "candidate_error_subtasks": candidate_error_subtasks,
        "candidate_error_agents": candidate_error_agents,
        "candidate_error_steps": candidate_error_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — final single-step attribution
# ─────────────────────────────────────────────────────────────────────────────

def build_step6(history_text, question, ground_truth, candidate_set, dag_graph) -> str:
    """Verbatim from CHIEF.py:937-970."""
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation solving a real-world problem.\n"
        f"The problem is: {question}\n"
        f"The correct answer for the problem is: {ground_truth}\n\n"
        "Here is the multi-agent conversation:\n"
        + str(history_text)
        + f"\n\nThere are total {len(history_text)} steps, each entry provides an agent's output.\n\n"
        "Here is the structured candidate_set generated by a previous analysis stage.\n"
        + str(candidate_set)
        + "\n\nHere is the graph:\n"
        + str(dag_graph)
        + "\n\nYour job is to identify the SINGLE most responsible reasoning mistake that directly leads to the wrong final result.\n"
        "You MUST follow this reasoning procedure over the candidate_error_steps:\n"
        "1. For each candidate_error_step:\n"
        "   - Use loop_issue to analyze whether this step is inside a loop and whether it plays an entry/internal/exit role.\n"
        "     * According to Rule 1: if the loop should NOT have been entered, focus on the decision that caused entering the loop; otherwise, focus on actions that create irreversible consequences when entering or exiting the loop.\n"
        "   - Use data_issue to analyze whether this step fabricates key data, misinterprets upstream data, or misuses otherwise correct data.\n"
        "     * According to Rule 2: if information is fabricated, blame the step that produces it; if misinterpreted, blame the step that misreads it; if misused, blame the step where misuse occurs.\n"
        "   - Use irrecoverability_issue and impact to judge whether this step is the first point that prevents restoration of the correct reasoning path.\n"
        "     * According to Rule 3: the final error should be attributed to the first step after which it becomes hard or impossible to recover the correct path, not necessarily the earliest deviation.\n"
        "   - Combine: loop_issue, data_issue, irrecoverability_issue, impact. Form an internal responsibility score for each candidate step.\n"
        "2. Compare all candidate_error_steps:\n"
        "   - Prefer steps with strong data_issue and irrecoverability_issue, especially when they are loop entries or irreversible loop exits.\n"
        "   - If two steps are similar, choose the earlier step that already makes the path practically irrecoverable.\n"
        "3. After you decide the single most responsible step:\n"
        "   - Determine the agent name at that step (from agent_in_step and the conversation context).\n"
        "   - Determine the exact step index (step_id in the candidate_error_step).\n"
        "   - Provide a concise explanation that explicitly mentions how the three rules apply to this step.\n\n"
        "Now, please answer in this exact plain text format:\n"
        "Agent Name: (your final prediction, a single agent name)\n"
        "Step Number: (your final prediction, a single integer step id)\n"
        "Reason for Mistake: (your explanation, summarize in 2-3 sentences)\n"
        "No special symbols, no extra commentary.\n"
    )


def parse_step6(response):
    """Parse stage-6 output into the final prediction. Verbatim from CHIEF.py:973-997."""
    result_text = response.strip()

    agent_pattern = r"Agent Name:\s*([^\n*]+)"
    step_pattern = r"Step Number:\s*([0-9]+)"
    reason_pattern = r"Reason for Mistake:\s*(.*)"

    agent = re.search(agent_pattern, result_text)
    step = re.search(step_pattern, result_text)
    reason = re.search(reason_pattern, result_text, re.S)

    agent_name = agent.group(1).strip() if agent else None
    if agent_name:
        agent_name = agent_name.split("(", 1)[0].strip()

    step_number = int(step.group(1)) if step else None
    reason_text = reason.group(1).strip() if reason else ""

    return {
        "final": {
            "mistake_agent": agent_name,
            "mistake_step": step_number,
            "reason": reason_text,
        },
        "raw": result_text,
    }
