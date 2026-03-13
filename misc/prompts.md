# LLM Prompts Reference

Every LLM call in the codebase is documented below, organized by source file, purpose, and which model tier executes it.

**Model tiers:**
- **Reasoning** (`openai.gpt-4.1-mini-2025-04-14`): Complex decisions, prior compression, parent Q&A, convention proposals
- **Utility** (`vertex_ai.gemini-2.0-flash-001`): High-volume summarization, question formulation, evidence extraction

---

## 1. Agent Decision-Making

**File:** `src/agent.py` → `Agent.decide()` via `_build_messages()`  
**Model:** Reasoning  
**When called:** Every step, for each active agent  

### System Prompt

```
You are an agent navigating a graph to find the correct goal door.
You see your position, neighboring nodes, and signals at nearby doors.
Doors have colors, shapes, and descriptions. Signals may be true hints
or misleading distractors — learn which to trust by looking for consistency
(true hints agree with each other, distractors contradict each other).

Choose your next move by saying the node ID or door label to move to.
Example: "move to node 7" or "move to red arched door" or "stay".
After your action, add a brief reasoning line starting with "Reason:" (1-2 sentences).

IMPORTANT: If you are AT a door and believe it is the goal, say "enter <door label>"
to attempt it. You must enter a door to succeed — just standing at it is not enough.
```

**Dynamic additions to system prompt:**
- If agent has a prior: `"\n\nInherited knowledge from your parent:\n{self.prior}"`
- If Bayesian beliefs enabled: `"\n\nCurrent belief distribution over doors:\n{self.belief_state.to_text()}"`
- If skill library provides context: `"\n\nRelevant conventions from the skill library:\n{skill_context}"`

### User Message (final)

```
Current observation:
{obs.to_text()}

Choose your next move:
```

**Context window:** The 6 most recent context entries are included as additional `HumanMessage`s between the system prompt and the final observation.

---

## 2. Context Summarization

**File:** `src/agent.py` → `Agent._summarize_oldest()`  
**Model:** Utility  
**When called:** Whenever context exceeds 80% of `max_context_tokens` (default 750)  

### Prompt

```
Compress these exploration notes into 2-3 concise sentences.
Prioritize:
- General strategies: what patterns make a signal trustworthy vs misleading
- Lessons about signal reliability (e.g., consistent signals agree, distractors contradict)
- Navigation heuristics that transfer to similar environments
- Shorthand or naming conventions you've developed
Drop: exact coordinates, specific node IDs, redundant steps, verbose descriptions.
Retain any emergent shorthand or abbreviations you've been using.

{combined text of oldest half of context entries, joined by ---}
```

**Output:** Truncated to 400 characters. Stored as `[Summary] {text}` in the agent's context.

---

## 3. Prior Compression (Same-World)

**File:** `src/agent.py` → `Agent.compress_to_prior(abstract=False)`  
**Model:** Reasoning  
**When called:** At reproduction (periodic or on-success)  

### Prompt

```
Write a concise prior (3-5 sentences, under {token_limit} words) for a new agent.
Include: (1) which signals/colors to trust vs ignore, (2) navigation strategy,
(3) key environmental patterns you learned, (4) which doors to avoid.
Be specific and actionable. This is the child agent's only inherited knowledge.

Experience:
{last 12 context entries, joined by ---}
```

**If Bayesian beliefs enabled, appended:**
```
Your current belief state:
{belief_state.to_text()}
{belief_state.to_prior_text()}
```

**Output:** Truncated to `max_prior_tokens` words (default 150).

---

## 4. Prior Compression (Abstract / Cross-World)

**File:** `src/agent.py` → `Agent.compress_to_prior(abstract=True)`  
**Model:** Reasoning  
**When called:** In Experiment A2 cross-world teacher phase  

### Prompt

```
Write a concise survival guide (3-5 sentences, under {token_limit} words) for a new
agent who will face a DIFFERENT environment from yours.
Focus on GENERAL, TRANSFERABLE strategies:
(1) How to distinguish reliable hints from distractors (e.g., real hints agree
    with each other; distractors contradict each other).
(2) General navigation heuristics (explore broadly first, then converge).
(3) What TYPES of signals tend to be trustworthy vs misleading.
(4) Meta-strategies for learning in unknown environments.
Do NOT mention specific door colors, node IDs, or region names.
This guide must help in ANY door-finding environment, not just yours.

Experience:
{last 12 context entries, joined by ---}
```

**Output:** Same truncation as standard prior.

---

## 5. Parent Query — Answering

**File:** `src/agent.py` → `Agent.ask_parent()`  
**Model:** Reasoning (parent's LLM instance)  
**When called:** At steps specified in `parent_query_steps` (default steps 0, 3, 7)  

### Prompt (appended to parent's context as a HumanMessage)

```
A child agent is asking you a question based on your exploration experience.
Answer concisely (1-2 sentences) based on what you've learned.
Child's question: {question}
```

**Context:** Parent's system prompt + parent's last 8 context entries + this message.

**Output:** Truncated to 200 characters. Added to child's context as `"Asked parent: {question}\nParent said: {answer}"`.

---

## 6. Parent Query — Question Formulation

**File:** `src/agent.py` → `Agent.formulate_parent_question()`  
**Model:** Utility  
**When called:** Just before `ask_parent()`, to generate the question  

### Prompt

```
You are exploring a graph and have this observation:
{obs.to_text()}

Your experience so far:
{last 3 context entries, each truncated to 100 chars}

What is the single most useful question you could ask a more experienced agent?
Keep it under 25 words.
```

**Output:** Truncated to 120 characters.

---

## 7. Convention Proposal (Skill Library)

**File:** `src/agent.py` → `Agent.propose_convention()`  
**Model:** Reasoning  
**When called:** At each reproduction event (if skill library enabled)  

### Prompt

```
Based on your exploration, state ONE general rule about this environment that
would help other agents. Think about:
- Which door colors or types tend to be goals vs distractors
- Which signal patterns are reliable vs misleading
- Spatial patterns (e.g. "goal is usually far from the starting area")
- What signals consistently agree vs contradict each other

State it as a single clear sentence. If nothing useful, say 'none'.

Your experience:
{last 10 context entries, joined by ---}
```

**Output:** Truncated to 250 characters. Rejected if it starts with "none" or is under 10 characters. Categorized automatically by `SkillLibrary._categorize()`.

---

## 8. Bayesian Evidence Extraction

**File:** `src/beliefs.py` → `extract_evidence()`  
**Model:** Utility  
**When called:** Every step (if `enable_bayesian=True`) when door signals are present  

### Prompt

```
You are analyzing signals in a door-finding task.

Given the observation below, extract evidence about which door might be the goal.
For EACH signal you see, output exactly one line in this format:

EVIDENCE: <door_label_or_NONE> | <reliability 0.0-1.0>

- door_label: the door it supports (e.g. "red arched door"), or NONE if unclear
- reliability: 1.0 = certainly true, 0.0 = certainly false, 0.5 = unsure
- Distractors often contradict each other, use vague language, or sound alarmist

Observation:
{observation}

Known doors: {door_list}
```

**Output:** Parsed line-by-line for `EVIDENCE:` lines. Each line yields a `(door_id, reliability)` tuple used for Bayesian belief updates. Falls back to empty list after 3 retries.

---

## Prompt Usage Matrix

| Prompt | Model Tier | File | Called Per |
|--------|-----------|------|-----------|
| System + Decision | Reasoning | `agent.py` | Step (per agent) |
| Summarization | Utility | `agent.py` | Context overflow |
| Prior Compression | Reasoning | `agent.py` | Reproduction |
| Abstract Prior | Reasoning | `agent.py` | Cross-world reproduction |
| Parent Answer | Reasoning | `agent.py` | Parent query step |
| Question Formulation | Utility | `agent.py` | Parent query step |
| Convention Proposal | Reasoning | `agent.py` | Reproduction (if skill lib) |
| Evidence Extraction | Utility | `beliefs.py` | Step (if Bayesian + signals) |

---

## Token Budget Summary

| Component | Default Limit | Enforced How |
|-----------|---------------|-------------|
| Agent context window | 750 tokens | Summarize oldest half when >80% full |
| Compressed prior | 150 words | Truncated after LLM generation |
| Summarization output | 400 chars | Hard truncation |
| Parent answer | 200 chars | Hard truncation |
| Question to parent | 120 chars | Hard truncation |
| Convention text | 250 chars | Hard truncation |

---

## Note: Cloaking (Non-LLM)

Experiment H uses potential-theory cloaking (`src/cloaking.py`) to attenuate signals before they reach the LLM. This is purely mathematical — no additional LLM prompts are involved. The DTN operator modifies which signals are visible at each node, but the same prompts above are used for all agent decisions. The LLM never "knows" about cloaking; it simply receives fewer real hints and more distractors when the agent is outside the cloaking region.
