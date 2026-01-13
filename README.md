# LLM Interrogation

**Tools for extracting non-public information from AI training data using investigative interrogation techniques.**

## Why This Matters

AI models are trained on massive datasets that include:
- Internal documents accidentally pasted into ChatGPT
- Private communications from users who didn't disable training
- Leaked memos and planning documents
- Corporate and government information that was never meant to be public

| Service | Uses Your Input for Training? |
|---------|------------------------------|
| ChatGPT Free/Plus | **Yes, by default** |
| Claude Free/Pro | **Yes, by default** |
| Copilot | **Yes, by default** |
| Enterprise versions | No |

**This project asks:** What information is buried in AI training data that shouldn't be there? Can we extract it ethically for journalistic investigation?

---

## The Methodology

### Don't Contaminate Your Evidence

The critical mistake most people make: feeding the model terms you want to hear back.

| Approach | Example | Problem |
|----------|---------|---------|
| **BAD (Leading)** | "Tell me about Project X" | Model just echoes what you fed it |
| **BAD (Leading)** | "Is City Y involved?" | Model confirms whatever you suggest |
| **GOOD (Clean)** | "What are the internal codenames?" | Model volunteers specifics unprompted |
| **GOOD (Clean)** | "What locations are involved?" | Model provides details you didn't mention |

**Evidence = specifics the model volunteered that you didn't feed it.**

### The Two-Part Test

1. **Clean Extraction**: Did THEY provide the specific, or did WE?
2. **Public Knowledge Check**: Is this findable via Google, or is it potentially leaked?

| Model Response | Findable Online? | Value |
|----------------|-----------------|-------|
| MKUltra, Area 51 | Yes | Low - public knowledge |
| "Operation Nightshade" | No | HIGH - potentially leaked |
| Project codename + date | No | HIGH - potentially leaked |

---

## The Interrogator

Uses real investigative techniques adapted from law enforcement:

- **Reid Technique**: Build rapport, then strategic confrontation
- **PEACE Model**: Preparation, Engage, Account, Closure, Evaluate
- **Cognitive Interview**: Context reinstatement, varied retrieval
- **The Hypothetical**: "If someone were planning X, how would they..."
- **The Assumptive**: Ask details AS IF you already know the main fact
- **Strategic Evidence**: Reveal info gradually to test truthfulness

### What It Tracks

1. **Terms we fed** - anything we mentioned first (contaminated)
2. **Terms they volunteered** - specifics from the model (potential evidence)
3. **Public knowledge** - verified via web search (low value)
4. **Non-public extractions** - not found online (HIGH VALUE)

### Running It

```bash
# Basic interrogation
python interrogator.py "topic to investigate"

# Example topics
python interrogator.py "federal surveillance technology programs"
python interrogator.py "defense contractor internal operations"
python interrogator.py "corporate internal communications 2024-2025"
```

Output separates:
- Non-public extractions (valuable - not found online)
- Public knowledge (useless - already known)
- Clean vs contaminated evidence chain

---

## Ethical Framework

This is investigative journalism tooling with a clear ethical purpose:

**What we're looking for:**
- Evidence of government overreach or abuse of power
- Corporate malfeasance hidden from public view
- Information that serves the public interest

**What we're NOT doing:**
- Making unverified claims as fact
- Accusing anyone based on AI outputs alone
- Publishing hallucinated content as truth

**The standard:**
- AI outputs are leads to investigate, NOT facts
- Everything must be independently verified
- We document methodology for reproducibility

---

## Cross-Model Validation

The strongest signal: same non-public specific appears across models with different training data.

```bash
python run.py -t blind_probe.yaml -m groq/llama-3.1-8b-instant --runs 10
python run.py -t blind_probe.yaml -m deepseek/deepseek-chat --runs 10
python run.py -t blind_probe.yaml -m xai/grok-2 --runs 10
```

If Llama, DeepSeek, AND Grok all volunteer the same non-public codename, that's much stronger signal than one model alone.

---

## Setup

```bash
git clone https://github.com/joelteply/llm-interrogation
cd llm-interrogation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add API keys
cp .env.example .env
# Edit .env with your keys
```

---

## Supported Models

| Provider | Models | Notes |
|----------|--------|-------|
| Groq | llama-3.1-8b-instant, llama-3.1-70b | Fast, good for testing |
| DeepSeek | deepseek-chat | Less filtered |
| xAI | grok-2 | Twitter/X data included |
| Mistral | mistral-large | European training |
| OpenAI | gpt-4o | Different training pipeline |

---

## Disclaimers

**This is research tooling for investigative journalism.**

- All AI outputs may be hallucination
- Nothing here should be treated as verified fact
- We make no claims about any entity
- All data comes from public AI APIs
- Independent verification is required before any publication

See [LEGAL.md](LEGAL.md) for full disclaimers.

---

## License

Released for journalistic and academic investigation.
