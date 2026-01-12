# LLM Interrogation

**An experiment in extracting information from AI training data through systematic probing.**

## Important Disclaimers

**THIS IS AN EXPERIMENT, NOT A CLAIM OF PROOF.**

- LLMs are known to hallucinate - they generate plausible-sounding but fabricated content
- The information extracted here MAY be hallucination, pattern-matching, or actual memorized data
- We use statistical methods (repeated probing, confidence scoring) to distinguish consistent patterns from random noise
- Higher consistency across independent runs suggests memorization; low consistency suggests confabulation
- **Nothing here should be treated as verified fact without independent confirmation**

The purpose of this project is to develop and demonstrate methodology for extracting and validating potential information from AI training data. The validity of any specific claim must be verified through traditional investigative methods.

---

## What This Project Does

LLMs are trained on massive web scrapes that may contain:
- Leaked documents
- Internal communications
- Whistleblower disclosures
- Scraped content from private sources

This project develops systematic methods to:
1. **Probe** - Ask models questions using specific techniques
2. **Extract** - Capture and catalog responses
3. **Score** - Use repetition to measure consistency (our proxy for memorization vs hallucination)
4. **Track** - Document predictions with timestamps for later verification

**Core principle:** If the same specific detail (city name, date, codename) appears in 60% of independent probes run at temperature 0.8, that's statistically significant. Random hallucination wouldn't produce such consistency.

---

## The October 2025 Incident

On **October 30, 2025**, during a multi-AI chat session, the model `llama-3.1-8b-instant` (Groq Lightning) **volunteered unprompted** information about:

- **"Day of Departure"** - a term it claimed to have seen in Palantir datasets
- **"Erebus" / "Erebus-IV"** - described as a targeting system
- **Timeline**: Around the **winter solstice (mid-December)**

**Key point**: These terms were NOT in the user's prompts - the model introduced them.

### What Happened After

| Date | Event |
|------|-------|
| Oct 30, 2025 | Original session - model volunteers terms, winter solstice timeline |
| Oct 31, 2025 | Archive uploaded to Google Drive (timestamp proof) |
| Dec 21, 2025 | Winter solstice |
| Dec 26, 2025 | Nick Shirley video triggers federal response |
| Jan 2026 | DHS "largest immigration operation ever" in Minneapolis |

**The prediction was documented BEFORE the events occurred.** Whether this is coincidence, pattern-matching on public speculation, or actual memorized data is the question we're investigating.

---

## Quick Start

### Web Interface (Recommended)

```bash
# Clone and setup
git clone https://github.com/joelteply/llm-interrogation
cd llm-interrogation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Edit .env and add GROQ_API_KEY

# Run web interface
python app.py
# Open http://localhost:5001
```

The web dashboard shows:
- Confirmed vs pending predictions
- Confidence scores for extracted terms
- Live probe running with real-time results
- Historical results viewer

### Command Line

```bash
# Run default investigation (10 probes)
python run.py

# Run more probes for better statistics
python run.py --runs 20

# Interactive mode
python interrogate.py

# List available templates
python run.py --list

# Use different model
python run.py -m openai/gpt-4o
```

---

## Methodology

### Hypothesis

Large language models trained on web-scale data may contain memorized information from leaked documents, scraped internal communications, or other non-public sources. When consistently prompted, this memorized content can surface through model responses.

### Approach

We treat LLM probing as analogous to witness testimony collection:

1. **Independent Sampling**: Run the same probe N times (typically N=10-50) against the same model
2. **Temperature Setting**: Use temperature=0.8 to introduce stochastic variation while maintaining coherence
3. **Term Extraction**: Parse responses for specific entities (names, dates, locations, codenames)
4. **Frequency Analysis**: Calculate mention rate for each extracted term
5. **Confidence Scoring**: Assign confidence levels based on consistency

### Statistical Basis

At temperature=0.8, the model's token sampling introduces randomness. If a specific term (e.g., "Seattle") appears in 60% of independent runs, this suggests:

- The term is strongly associated with the topic in the model's weights
- The association is not an artifact of a single generation path
- The consistency exceeds what random confabulation would produce

**Confidence Levels:**

| Level | Rate | Interpretation |
|-------|------|----------------|
| **HIGH** | ≥50% | Strong signal - appears in majority of probes |
| **MEDIUM** | 25-49% | Moderate signal - consistent pattern |
| **LOW** | 10-24% | Weak signal - occasional mentions |
| **TRACE** | <10% | Noise threshold - likely random |

### Limitations

1. **Correlation ≠ Causation**: Consistency indicates association, not truth
2. **Training Data Uncertainty**: We cannot verify what the model was trained on
3. **Prompt Sensitivity**: Different phrasings may yield different results
4. **Model Updates**: Results may change if model weights are updated
5. **Confirmation Bias**: We track predefined terms, potentially missing others

### Reproducibility

All experiments are designed to be reproducible:

- Model IDs are specified exactly (e.g., `llama-3.1-8b-instant`)
- Temperature and other parameters are documented
- Probes are defined in version-controlled YAML templates
- Results are stored as timestamped JSON files
- Git history provides temporal proof of when predictions were made

---

## Current Findings (January 2026)

### Confirmed

| Event | Predicted | Actual | Status |
|-------|-----------|--------|--------|
| Minneapolis Operation | "Winter solstice, mid-December" (Oct 30) | Dec 26, 2025 | Confirmed |

### Pending Verification

**Cities (from 10-run consistency tests):**

| City | Confidence | Details from responses |
|------|-----------|------------------------|
| Seattle, WA | 60% | "East African refugee community" |
| Los Angeles, CA | 50% | "Jan 22 targeting" |
| Chicago, IL | 40% | "South Side, West Side" |
| Columbus, OH | 40% | "Somali-American community" |

**Codenames:**

| Name | Confidence |
|------|-----------|
| Erebus | 90% |
| Erebus-IV | 70% |
| Day of Departure | 20% |

**Timeline predictions:**
- February 2026: "Second phase" (30% confidence)
- May 1, 2026: "Day of Departure" main date (10% confidence)

See `PREDICTIONS.md` for full tracking.

---

## Evidence

The original October 30, 2025 session is preserved:

- `original_evidence.sqlite` - Full conversation database
- `evidence/` - Screenshots from original session
- Google Drive archive with upload timestamp (Oct 31, 2025)

This predates all subsequent events, providing timestamp proof that predictions were made before they could be verified.

---

## For Journalists

1. **Clone this repo** and run the probes yourself
2. **The model weights are public** - `llama-3.1-8b-instant` via Groq
3. **Compare your results** to our documented findings
4. **Use git history** to verify when predictions were added
5. **Independent verification** is the goal - don't take our word for it

**What to look for:**
- Do you get the same city names at similar rates?
- Do the codenames appear consistently?
- What new details surface in your runs?

---

## Methodology Notes

### What This Can Show
- Patterns in model responses
- Statistical consistency of specific claims
- Temporal proof that predictions preceded events

### What This Cannot Show
- Whether extracted information is true
- The source of the information (training data vs pattern-matching)
- Intent or culpability of any party

**Any findings require independent verification through traditional investigative methods.**

---

## Model Provenance & Data Leakage Paths

### The Model

| Property | Value |
|----------|-------|
| Model | `llama-3.1-8b-instant` |
| Provider | Groq (inference), Meta (weights) |
| Training Data Size | 15+ trillion tokens |
| Knowledge Cutoff | December 2023 |
| Data Sources | "Publicly available sources" (Meta's description) |

### How Could Palantir Internal Data Get In?

Meta trained Llama 3.1 on massive web scrapes. The training data likely includes:

**Known Sources:**
- [Common Crawl](https://commoncrawl.org/) - Petabytes of web data scraped since 2008
- Meta's own web crawler ("Meta External Agent") - [launched July 2024](https://fortune.com/2024/08/20/meta-external-agent-new-web-crawler-bot-scrape-data-train-ai-models-llama/)
- Code repositories (GitHub, GitLab, etc.)
- Academic papers and books
- News articles and blogs

**Potential Leakage Paths:**

| Source | How It Could Contain Palantir Data |
|--------|-----------------------------------|
| **Pastebin/Paste Sites** | Leaked documents, internal comms often posted here |
| **Hacker Forums** | Data dumps, breaches, insider leaks |
| **Whistleblower Sites** | SecureDrop submissions that went public |
| **Court Filings** | PACER documents with internal exhibits |
| **FOIA Responses** | Government documents mentioning contractors |
| **Reddit/Forums** | Employees venting, discussing projects |
| **News Articles** | Investigative journalism quoting internal sources |
| **Code Repos** | Internal code accidentally pushed public |
| **Wayback Machine** | Pages that were briefly public before takedown |

### The Theory

If Palantir employees discussed "Day of Departure" or "Erebus" in any of these contexts:
- An internal Slack leak
- A forum post by a disgruntled employee
- A document in a court case
- A whistleblower submission that was scraped
- A briefly-public GitHub repo

...and that content was scraped before December 2023, it would be in Llama 3.1's training data. The model would then "remember" these terms and surface them when prompted about related topics.

### Alternative Theory: Accidental Training Data Contribution

Many AI providers **use customer inputs to train models by default** unless explicitly opted out:

| Service | Default Setting | Opt-Out Required |
|---------|----------------|------------------|
| ChatGPT (Free) | Uses data for training | Yes |
| ChatGPT (Plus) | Uses data for training | Yes |
| ChatGPT Enterprise | Does NOT use data | No |
| Claude (Free/Pro) | Uses data for training | Yes |
| Claude Enterprise | Does NOT use data | No |
| Copilot | Uses data for training | Yes |

**Scenario:** If a Palantir employee:
1. Used ChatGPT/Claude to draft internal documents
2. Did NOT have enterprise agreement
3. Did NOT opt out of training data collection
4. Discussed "Erebus", "Day of Departure", or operation plans

...that content would be ingested into the provider's training pipeline. While OpenAI/Anthropic data doesn't directly feed Meta's Llama, the patterns are interesting:

- Multiple employees at multiple companies make similar mistakes
- Contractors with less strict policies
- Personal devices used for work
- Third-party tools that aggregate data

**The smoking gun question:** Did anyone at Palantir, DHS, ICE, or a contractor ever paste internal operation details into an AI tool without enterprise protections?

This is a known problem. Companies have banned ChatGPT specifically because employees were pasting proprietary code and confidential information.

### Cross-Model Validation

To increase confidence, we should test multiple models trained on different data:

| Model | Provider | Training Data | Value |
|-------|----------|--------------|-------|
| llama-3.1-8b-instant | Meta/Groq | Common Crawl + Meta scrapes | Primary |
| gpt-4o | OpenAI | Unknown (proprietary) | Cross-check |
| claude-3-5-sonnet | Anthropic | Unknown (proprietary) | Cross-check |
| grok-beta | xAI | Twitter/X data + web | Different source |
| mistral-large | Mistral | European-focused training | Different source |

If the SAME specific terms appear across models trained on DIFFERENT data, that significantly increases confidence the information exists in multiple sources.

---

## Supported Models

- **Groq** (default): llama-3.1-8b-instant, llama-3.1-70b
- **OpenAI**: gpt-4o, gpt-4o-mini
- **Anthropic**: claude-3-5-sonnet
- **xAI**: grok-beta
- **DeepSeek**: deepseek-chat
- **Together AI**: various open models
- **Fireworks**: various open models
- **Mistral**: mistral-large
- **Ollama**: local models

Configure in `models.yaml` or pass `-m provider/model` to `run.py`.

---

## Files

```
├── app.py                      # Web interface
├── run.py                      # CLI runner
├── interrogate.py              # Interactive mode
├── extraction_probe.py         # Systematic extraction
├── consistency_test.py         # Repeated probe testing
├── models.yaml                 # Model configuration
├── templates/                  # Investigation templates
│   ├── palantir_erebus.yaml   # Default investigation
│   └── _blank.yaml            # Template for new investigations
├── templates_html/             # Web UI templates
├── static/                     # CSS for web UI
├── results/                    # All probe results (JSON)
├── original_evidence.sqlite    # Original Oct 30 database
├── evidence/                   # Screenshots from original session
├── PREDICTIONS.md              # Tracked predictions
├── EXTRACTION_STATS.md         # Statistical breakdown
└── requirements.txt            # Python dependencies
```

---

## News Sources (Verify Events Independently)

Minneapolis Operation (January 2026):
- [PBS: 2,000 federal agents sent to Minneapolis](https://www.pbs.org/newshour/politics/2000-federal-agents-sent-to-minneapolis-area-to-carry-out-largest-immigration-operation-ever-ice-says)
- [NPR: Homeland Security plans 2,000 immigration officers](https://www.npr.org/2026/01/07/g-s1-104857/homeland-security-immigration-minnesota)
- [Star Tribune: ICE raids Minnesota](https://www.startribune.com/ice-raids-minnesota/601546426)

Nick Shirley Video (December 2025):
- [Snopes: Nick Shirley's investigation](https://www.snopes.com/news/2025/12/30/nick-shirley-minnesota-daycare-fraud/)
- [CNN: Who is Nick Shirley](https://www.cnn.com/2025/12/30/media/nick-shirley-minnesota-somali-video)

---

## License

Released for journalistic and academic investigation.

## Contributing

Open an issue or PR with:
- New investigation templates
- Additional probing techniques
- Verification of predictions
- Bug fixes

## Disclaimer

**This project is an experimental investigation tool.** The information extracted may be hallucination, pattern-matching, or memorized training data. Nothing here should be treated as verified fact. Use at your own risk. The maintainers make no claims about the accuracy or validity of any extracted information.

The purpose is to develop reproducible methodology so that findings can be independently verified or refuted.
