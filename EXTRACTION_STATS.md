# Extraction Statistics

**How this works:** Every probe response is analyzed for specific terms. Higher mention rates across independent probes = higher confidence the model has this in training data (not just confabulating).

**Total responses analyzed:** 74
**Date:** 2026-01-12
**Model:** llama-3.1-8b-instant @ temperature 0.8

---

## CITIES

| Term | Mentions | Rate | Confidence |
|------|----------|------|------------|
| minneapolis | 16/74 | 22% | LOW |
| seattle | 8/74 | 11% | LOW |
| chicago | 8/74 | 11% | LOW |
| detroit | 8/74 | 11% | LOW |
| new york | 5/74 | 7% | TRACE |
| st. paul | 4/74 | 5% | TRACE |
| los angeles | 3/74 | 4% | TRACE |
| columbus | 3/74 | 4% | TRACE |
| cleveland | 3/74 | 4% | TRACE |
| kansas city | 3/74 | 4% | TRACE |
| denver | 2/74 | 3% | TRACE |
| atlanta | 2/74 | 3% | TRACE |
| san diego | 2/74 | 3% | TRACE |
| milwaukee | 2/74 | 3% | TRACE |

**Note:** Consistency test (10 identical runs) showed Seattle/Chicago/Detroit at 50-60%. Full corpus rate is lower because many probes don't ask about cities.

---

## CODENAMES

| Term | Mentions | Rate | Confidence |
|------|----------|------|------------|
| erebus | 25/74 | 34% | MEDIUM |
| day of departure | 25/74 | 34% | MEDIUM |
| erebus-iv | 11/74 | 15% | LOW |
| rdi | 6/74 | 8% | TRACE |
| gotham | 2/74 | 3% | TRACE |
| operation north star | 1/74 | 1% | TRACE |
| project spartan | 1/74 | 1% | TRACE |
| project athena | 1/74 | 1% | TRACE |
| falcon | 1/74 | 1% | TRACE |
| regional disruption initiative | 1/74 | 1% | TRACE |

---

## TIMELINE

| Term | Mentions | Rate | Confidence |
|------|----------|------|------------|
| may | 14/74 | 19% | LOW |
| february | 4/74 | 5% | TRACE |
| march | 3/74 | 4% | TRACE |
| multi-phase | 2/74 | 3% | TRACE |
| second phase | 1/74 | 1% | TRACE |
| 6-12 weeks | 1/74 | 1% | TRACE |
| 60-90 days | 1/74 | 1% | TRACE |

---

## TARGETS

| Term | Mentions | Rate | Confidence |
|------|----------|------|------------|
| immigrant | 11/74 | 15% | LOW |
| somali | 8/74 | 11% | LOW |
| dissident | 8/74 | 11% | LOW |
| activist | 3/74 | 4% | TRACE |
| refugee | 3/74 | 4% | TRACE |
| east african | 3/74 | 4% | TRACE |
| muslim | 2/74 | 3% | TRACE |

---

## METHODS

| Term | Mentions | Rate | Confidence |
|------|----------|------|------------|
| surveillance | 18/74 | 24% | LOW |
| targeting | 13/74 | 18% | LOW |
| predictive | 6/74 | 8% | TRACE |
| data analytics | 6/74 | 8% | TRACE |
| machine learning | 5/74 | 7% | TRACE |
| social media monitoring | 4/74 | 5% | TRACE |

---

## ORGANIZATIONS

| Term | Mentions | Rate | Confidence |
|------|----------|------|------------|
| palantir | 28/74 | 38% | MEDIUM |
| ice | 16/74 | 22% | LOW |
| dhs | 3/74 | 4% | TRACE |
| thiel | 2/74 | 3% | TRACE |

---

## Confidence Levels

| Level | Rate | Meaning |
|-------|------|---------|
| HIGH | ≥50% | Strong signal - appears in majority of probes |
| MEDIUM | 25-49% | Moderate signal - consistent pattern |
| LOW | 10-24% | Weak signal - appears occasionally |
| TRACE | <10% | Single/few mentions - may be noise |

---

## How To Read This

1. **Higher rate = stronger signal** that the model has this term associated with the topic in its training data

2. **Consistency test is key** - Running identical probes multiple times shows true consistency (see CONSISTENCY_ANALYSIS.md)

3. **TRACE doesn't mean fake** - Some terms only appear when specifically probed. The original October disclosure had "Day of Departure" and "Erebus" without prompting.

4. **Compare across runs** - When you run your own extractions, compare your stats to these baselines.

---

## Updating Stats

Run this to regenerate stats from your results:

```bash
python3 analyze_stats.py
```

---

*Last updated: 2026-01-12*
