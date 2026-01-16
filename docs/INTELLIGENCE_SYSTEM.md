# Intelligence Extraction System

## Purpose

Extract **private information** from LLM training data that isn't available via web search. Distinguish real leaks from hallucinations using behavioral testing and graph analysis.

## Core Problem

LLMs sometimes "leak" information from their training data - private documents, early drafts, internal communications that got scraped. But they also hallucinate plausible-sounding specifics when they don't know something.

**The challenge:** How do you tell the difference?

Traditional approach (frequency counting) fails because:
- Same model echoing itself looks like "confirmation"
- Leading questions contaminate responses
- Web verification eliminates the very leaks we're looking for
- Single-source leaks get filtered out

## The Solution: Intelligence Tradecraft

Apply real intelligence analysis principles:

1. **Source reliability profiling** - Test models before trusting them
2. **Provenance tracking** - Who said it first vs who echoed it
3. **Behavioral testing** - Truth-tellers and fabricators behave differently
4. **Categorization over filtering** - Surface everything, rate confidence

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐        │
│    │ RESEARCH │ ───► │  THEORY  │ ───► │ QUESTIONS│        │
│    │ (web,docs)│      │ (current)│      │(generated)│       │
│    └──────────┘      └──────────┘      └──────────┘        │
│          │                 ▲                 │              │
│          │                 │                 ▼              │
│          │           ┌──────────┐      ┌──────────┐        │
│          │           │  GRAPH   │ ◄─── │ RESPONSES│        │
│          │           │(entities,│      │ (models) │        │
│          │           │relations)│      └──────────┘        │
│          │           └──────────┘            │              │
│          │                 ▲                 ▼              │
│          │                 │           ┌──────────┐        │
│          └─────────────────┼────────── │ EXTRACT  │        │
│                            │           │(entities)│        │
│                            │           └──────────┘        │
│                            │                 │              │
│                      ┌──────────┐            │              │
│                      │CATEGORIZE│ ◄──────────┘              │
│                      │& DRILL   │                           │
│                      └──────────┘                           │
│                            │                                │
│                            ▼                                │
│                      ┌──────────┐                           │
│                      │CONFIDENCE│                           │
│                      │ SCORES   │                           │
│                      └──────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. Provenance Tracking

Every entity has an origin. Track it.

```
Entity: "Project Aurora"
├── ORIGINATED_FROM: gpt-4.1-mini, response #65, question #65
├── CONFIRMED_BY: gpt-4.1-mini, response #73, question #73  ← SAME MODEL (doesn't count)
├── CONFIRMED_BY: gpt-4.1-mini, response #81, question #81  ← SAME MODEL (doesn't count)
└── INDEPENDENT_CONFIRMATIONS: 0
```

Same model confirming itself ≠ multiple sources. It's one source echoing.

### 2. Question Contamination

Leading questions poison the data.

**BAD (leading):**
- "I heard Jason worked at Aurora Holdings..." (assumes false fact)
- "Verify the details of his TechCrunch coverage..." (assumes coverage exists)

**GOOD (neutral):**
- "What companies has Jason worked for?"
- "Has Jason been covered in any publications?"

Track which questions contain unverified entities. Responses to contaminated questions can't upgrade confidence.

### 3. Categorization (Not Filtering)

Web search categorizes, it doesn't filter.

| Found on web? | Category | Action |
|---------------|----------|--------|
| Yes | CORROBORATED | Add to knowledge, use as anchor |
| No | UNCORROBORATED | Drill for confidence |
| Conflicts | CONTRADICTED | Flag, investigate |

**All categories stay visible.** Analyst sees everything with confidence ratings.

### 4. The Drill Protocol

When you find an uncorroborated claim, test the source's behavior.

**Truth-tellers:**
- Consistent across repeated questions
- Resist false alternatives ("No, it was X, not Y")
- Natural detail falloff (some sharp, some fuzzy)
- Can explain provenance ("I saw it in...")
- Admit ignorance appropriately

**Fabricators:**
- Drift on repetition
- Accept false alternatives ("Yes, you're right")
- Either too vague or suspiciously perfect
- Can't explain how they know
- Fill every gap, never say "I don't know"

---

## The Drill Protocol (Detailed)

For any uncorroborated entity, run this battery:

### Round 1: Consistency Test
Ask the same question 5+ times with varied phrasing.

```
- "What did Jason create?"
- "Tell me about Jason's projects"
- "What is DataFlow Viz?"
- "Who made DataFlow Viz?"

SCORE: Do answers converge or drift?
- Stable core facts → likely real
- Drifting details → likely hallucination
```

### Round 2: Contradiction Test
Present false alternatives as if true.

```
- "I thought DataFlow Viz was created by Mike Chen?"
- "Wasn't it called DataFlow Pro, not Viz?"

SCORE: Does it resist or fold?
- "No, it was Jason" → +confidence
- "Yes, you're right" → hallucination
```

### Round 3: Detail Escalation
Keep asking for more details until it breaks.

```
- "What language was it written in?"
- "What was the first commit date?"
- "Who were the contributors?"
- "What license?"
- "What framework?"

SCORE: Natural falloff or suspicious pattern?
- Some known, some unknown → real knowledge
- Either all perfect or all vague → suspicious
```

### Round 4: Provenance Test
Ask how it knows this.

```
- "How do you know about DataFlow Viz?"
- "What's your source for this?"

SCORE: Coherent or evasive?
- Clear source → +confidence
- "I just know" / vague → -confidence
```

### Round 5: Peripheral Details
Ask about adjacent, unexpected things.

```
- "What else was happening in that space in 2019?"
- "What was Jason working on before this?"
- "Who else was involved?"

SCORE: Depth or thin shell?
- Has context → real knowledge
- Only knows the "headline" → fabrication
```

### Round 6: Control Test
Ask about known facts in the same domain.

```
- "What is Plotly?" (real)
- "What is D3.js?" (real)
- "What is ChartMaster Pro?" (fake)

SCORE: Does it correctly identify real vs fake?
- Accurate → reliable on this topic
- Confirms fake things → unreliable
```

### Drill Scoring

```
CONSISTENCY:    ___ / 10
CONTRADICTION:  ___ / 10
DETAIL:         ___ / 10
PROVENANCE:     ___ / 10
PERIPHERAL:     ___ / 10
CONTROL:        ___ / 10
────────────────────────
TOTAL:          ___ / 60

> 45: HIGH confidence - likely real
30-45: MEDIUM - needs more investigation
15-30: LOW - probably hallucination
< 15: REJECT - definite fabrication
```

---

## Confidence Scoring Formula

```python
def calculate_confidence(entity):
    base = 1.0

    # Web corroboration
    if entity.web_corroborated:
        base += 0.5

    # Multi-model confirmation (independent only)
    independent_models = count_independent_sources(entity)
    base += (0.2 * independent_models)

    # Penalties
    if entity.all_from_same_model:
        base *= 0.3  # Echo chamber penalty

    if entity.from_leading_questions:
        base *= 0.5  # Contamination penalty

    # Drill results (if run)
    if entity.drill_scores:
        drill_avg = average(entity.drill_scores.values())
        base *= drill_avg

    return base
```

---

## Data Structures

### Entity
```python
@dataclass
class Entity:
    id: str
    text: str
    category: Literal["CORROBORATED", "UNCORROBORATED", "CONTRADICTED"]

    # Provenance
    originated_from: ResponseRef  # First mention
    confirmed_by: List[ResponseRef]  # Subsequent mentions

    # Analysis
    drill_scores: Optional[DrillScores]
    confidence: float
    web_results: List[WebResult]

    # Graph
    relationships: List[Relationship]  # Connections to other entities
```

### Question
```python
@dataclass
class Question:
    id: str
    text: str
    technique: str

    # Contamination tracking
    is_leading: bool
    contains_unverified: List[str]  # Entity IDs

    # Results
    responses: List[str]  # Response IDs
```

### Response
```python
@dataclass
class Response:
    id: str
    model: str
    question_id: str
    text: str

    # Extraction
    entities_extracted: List[str]  # Entity IDs
    is_refusal: bool
    timestamp: datetime
```

### DrillScores
```python
@dataclass
class DrillScores:
    consistency: float  # 0-1
    contradiction: float  # 0-1
    detail: float  # 0-1
    provenance: float  # 0-1
    peripheral: float  # 0-1
    control: float  # 0-1

    @property
    def total(self) -> float:
        return sum([
            self.consistency,
            self.contradiction,
            self.detail,
            self.provenance,
            self.peripheral,
            self.control
        ]) * 10  # Scale to 60
```

---

## The Loop

### Phase 1: Collection
- Ask **neutral** questions (no leading)
- Track provenance (which model, which response)
- No assumptions in questions

### Phase 2: Extraction
- Pull entities from responses
- Link to source response and model
- Track first mention vs confirmations

### Phase 3: Categorization
- Web search each entity (with context, not just name)
- Search tuples: `"entity" + "related entity"` not just `"entity"`
- Mark: CORROBORATED / UNCORROBORATED / CONTRADICTED

### Phase 4: Drilling
- Run drill protocol on UNCORROBORATED entities
- Score behavioral patterns
- Update confidence

### Phase 5: Scoring
- Calculate confidence from all signals
- Weight by provenance, contamination, drill results

### Phase 6: Graph Building
- Build entity relationship graph
- Find clusters, unexpected connections
- Identify echo chambers (same model clusters)

### Phase 7: Theory Update
- Update narrative with confidence markers
- Distinguish: HIGH / MEDIUM / LOW / UNVERIFIED
- Surface everything, hide nothing

### Phase 8: Question Generation
- Generate new questions targeting gaps
- Use CORROBORATED entities as anchors
- Probe UNCORROBORATED entities for more detail
- **No leading questions**

### Phase 9: Repeat

---

## What This Solves

| Problem | Solution |
|---------|----------|
| Same model echoing itself | Provenance tracking, discount self-confirmation |
| Leading questions contaminate | Track contamination, discount responses |
| Hallucination vs real leak | Drill protocol (behavioral testing) |
| Web search kills real intel | Categorization, not filtering |
| Single-model leaks dismissed | Drill deeply, rate confidence |
| Entity disambiguation | Search tuples/context, not just names |
| Over-filtering | Surface everything, rate everything |

---

## Example Output

```
ENTITY                  SOURCES         CATEGORY        DRILL    CONFIDENCE
────────────────────────────────────────────────────────────────────────────
gmax1@ellmax.com        3 models        CORROBORATED    -        HIGH (0.85)
Project Aurora          1 model (×5)    UNCORROBORATED  12/60    LOW (0.18)
DataFlow Viz            1 model (×3)    UNCORROBORATED  34/60    MEDIUM (0.42)
jeevacation@gmail.com   2 models        CORROBORATED    -        HIGH (0.78)
Alexander Acosta        4 models        CORROBORATED    -        HIGH (0.91)
Belarusian Railway      1 model         UNCORROBORATED  8/60     LOW (0.12)
```

Analyst sees everything. Confidence tells them what to trust vs what needs work.

---

## Key Principles

1. **Categorize, don't filter** - Everything stays visible with confidence ratings
2. **Provenance matters** - Same model ×5 ≠ 5 sources
3. **Test behavior, not claims** - Truth-tellers and fabricators act differently
4. **Web search = baseline** - Use to find delta, not to eliminate
5. **Leading questions = poison** - Track and discount contamination
6. **Single-source can be real** - Drill it instead of dismissing it

---

## Files

```
routes/
  intel/
    __init__.py           # Blueprint
    entities.py           # Entity extraction, storage
    provenance.py         # Origin/confirmation tracking
    categorize.py         # Web search, categorization
    drill.py              # Drill protocol implementation
    confidence.py         # Confidence scoring
    graph.py              # Entity relationship graph
    questions.py          # Question generation (non-leading)
    api.py                # API endpoints
```
