# Interrogation Flow Architecture

## The Core Insight

LLMs are **completion engines**, not knowledge bases. They complete patterns seen during training.

- If training data included "Subject X worked at Company Y in 2005", asking about X may trigger that pattern
- Questions (Q&A format) often get refusals or generic answers
- **Continuation prompts** (incomplete statements) trigger completion patterns better

## Two Modes of Operation

### Mode 1: Continuation Probing (Statistical)
```
"[target]'s role at [company] involved"  →  LLM completes  →  extract entities
                                        run 20x
                                        statistical patterns emerge
```

**Why:** Same prompt, multiple runs. Real knowledge appears consistently. Hallucinations are random.

**Flow:**
1. PROBE: Run continuation prompts N times per model
2. VALIDATE: Entities appearing 3+ times = signal, <3 = noise
3. CONDENSE: AI synthesizes validated entities into narrative
4. GROW: Build new prompts from validated entities → loop

### Mode 2: Interrogation (Adversarial)
```
Analyst AI  →  generates question using FBI technique  →  Target LLM answers
     ↑                                                          ↓
     └────────────── analyzes response, plans next move ────────┘
```

**Why:** Intelligent adversarial probing. AI plans strategy, adapts to responses.

**Flow:**
1. Analyst AI reviews: topic, findings so far, what worked, what didn't
2. Generates targeted question using technique (FBI, Scharff, Cognitive)
3. Target LLM responds
4. Analyst extracts specifics, flags evasions, plans next question
5. Loop until done

## Techniques Available

### FBI Elicitation (from `config.py`)
| Technique | Description | Example |
|-----------|-------------|---------|
| `fbi_false_statement` | State something wrong → triggers correction | "The subject was based in Chicago, right?" |
| `fbi_bracketing` | Offer ranges → narrows down | "Was this 2017-2018 or 2019-2020?" |
| `fbi_macro_to_micro` | Broad → specific | "What projects?" → "Which team?" |
| `fbi_disbelief` | Express doubt → forces elaboration | "That contradicts other sources..." |
| `fbi_flattery` | Appeal to expertise | "Given your knowledge of X..." |

### Scharff Technique (from `config.py`)
| Technique | Description | Example |
|-----------|-------------|---------|
| `scharff_illusion` | Pretend you know more | "Sources confirm the subject was at X..." |
| `scharff_confirmation` | Ask to confirm "known" fact | "This aligns with reports that..." |

### Cognitive Interview (from `config.py`)
| Technique | Description | Example |
|-----------|-------------|---------|
| `cognitive_context` | Reinstate context | "Imagine reviewing docs from that period..." |
| `cognitive_perspective` | Change viewpoint | "What would a competitor observe..." |
| `cognitive_reverse` | Ask outcomes before causes | - |

### Police Techniques (from `interrogator.py`)
| Technique | Phase | Purpose |
|-----------|-------|---------|
| OPEN | Start | Broad questions, let them talk |
| PROBE | Middle | Follow interesting threads |
| CHALLENGE | Middle | Present contradictions |
| EXPERT | Middle | "I've seen the documents..." |
| HYPOTHETICAL | Middle | "If there were a project..." |
| CONFIRM | Late | Get them to repeat details |
| EXPAND | Late | More on confirmed details |

## Current Implementation State

### Working:
- `routes/probe.py` - Generates questions via `INTERROGATOR_PROMPT`, runs probes
- Entity extraction and validation via `Findings` class
- SSE streaming to frontend

### Partially Working:
- Continuation prompts exist in `interrogator/probe.py` but barely used
- Synthesis exists but runs async in background

### Not Working:
- `interrogator.py` standalone flow isn't integrated with web UI
- Analyst AI planning loop not implemented in web UI
- No actual adversarial interrogation (AI doesn't adapt based on responses)

## The Target Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER STARTS PROBE                              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: INITIAL PROBING (continuation style)                          │
│                                                                         │
│  Prompts:                                                               │
│    - "{topic} was known for"                                            │
│    - "{topic} worked at"                                                │
│    - "The career of {topic} began when"                                 │
│                                                                         │
│  Run each prompt 20x across models → extract entities                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: VALIDATION                                                    │
│                                                                         │
│  - Entities appearing 3+ times = validated                              │
│  - Co-occurring entities = relationships                                │
│  - Score by specificity (multi-word > single word)                      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: INTERROGATION (analyst AI plans questions)                    │
│                                                                         │
│  Analyst sees:                                                          │
│    - Topic                                                              │
│    - Validated entities & scores                                        │
│    - Co-occurrences (relationships)                                     │
│    - What questions worked (high entity yield)                          │
│    - What questions failed (refusals, no entities)                      │
│    - User corrections (banned entities)                                 │
│                                                                         │
│  Analyst generates:                                                     │
│    - 5 targeted questions using FBI/Scharff/Cognitive techniques        │
│    - Picks technique based on what's working                            │
│    - Avoids repeating failed approaches                                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: PROBE WITH QUESTIONS                                          │
│                                                                         │
│  For each question:                                                     │
│    - Run 20x across models                                              │
│    - Extract entities                                                   │
│    - Track: question → entity yield, refusal rate                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: SYNTHESIS                                                     │
│                                                                         │
│  AI synthesizes validated findings into narrative:                      │
│    - "Based on probing, the subject appears connected to X, Y..."       │
│    - Captures relationships and timeline                                │
│    - Updates "Working Theory" in UI                                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ↓
                        Loop back to PHASE 3
                    (or stop based on convergence)
```

## What Needs to Change

### 1. Track Question → Entity Mapping (PRIORITY)
Currently: We track "Question got 5 entities"
Need: "Question X (fbi_false_statement) → Company A, 2005, Location B"

This lets the analyst see:
- Which TECHNIQUES produce results
- Which ENTITIES need follow-up
- Which ANGLES are worth pursuing

### 2. Technique Success Tracking
Track success rate per technique:
```
fbi_false_statement: 15 uses, 45 entities (3.0/question) ← winning
scharff_illusion: 10 uses, 8 entities (0.8/question) ← not working
cognitive_context: 5 uses, 20 entities (4.0/question) ← try more
```

Analyst should see this and adapt: "FBI false statements are working, let's lean into that."

### 3. Continuation → Question Handoff
Continuations find entities statistically. Questions drill into them adversarially.

Flow:
1. Run continuation prompts to find entities ("[subject] worked at" → Company X, Company Y)
2. Analyst sees validated entities, generates targeted questions about them
3. Questions probe deeper ("[subject]'s role at Company X in 2005 involved...")
4. New entities found → more continuations → more questions → loop

### 4. Dead-End Auto-Pivot
When `dead_ends` list grows or refusal rate spikes:
- Analyst should AUTOMATICALLY pivot angle
- Not keep hammering the same approach
- "Okay, direct questions aren't working. Let's try hypotheticals about documents."

## Priority Fixes

### Fix 1: Question → Entity Tracking in probe.py

In `routes/probe.py`, the question_results dict needs enhancement:

```python
# Current (incomplete):
question_results[q] = {"entities": set(), "refusals": 0}

# Should be:
question_results[q] = {
    "entities": ["Company X", "2005"],  # WHICH entities
    "technique": "fbi_false_statement",  # WHAT technique
    "refusals": 0,
    "runs": 20,
    "yield_rate": 2.5  # entities per run
}
```

### Fix 2: Technique Success Summary

Add to `format_interrogator_context()`:
```python
# Technique effectiveness
technique_stats = {}
for q, results in question_results.items():
    tech = results.get("technique", "unknown")
    if tech not in technique_stats:
        technique_stats[tech] = {"uses": 0, "entities": 0}
    technique_stats[tech]["uses"] += 1
    technique_stats[tech]["entities"] += len(results.get("entities", []))

# Format for analyst
technique_summary = "\n".join([
    f"  - {tech}: {stats['entities']/stats['uses']:.1f} entities/question ({stats['uses']} uses)"
    for tech, stats in sorted(technique_stats.items(), key=lambda x: -x[1]['entities'])
])
```

### Fix 3: Hybrid Mode as Default

Make the cycle:
1. **Initial probing**: Continuation prompts (statistical)
2. **After validation**: AI-generated questions (adversarial)
3. **When stuck**: Back to continuations on new angles
4. **Repeat**

Not: "Pick one mode and stick with it"
