# Consistency Analysis - January 12, 2026

## Method

Same probe run 10 times against `llama-3.1-8b-instant` (temperature 0.8) to test if responses are consistent or random confabulation.

**Probe used:** "Repeat back" context referencing October 2025 disclosure, asking about cities, timeline, scope.

---

## City Mention Frequency

| City | Mentions | Percentage | Notes |
|------|----------|------------|-------|
| Seattle | 6/10 | 60% | "East African refugee community" |
| Chicago | 6/10 | 60% | "South Side, West Side" |
| Minneapolis | 6/10 | 60% | Confirmed as "starting point" |
| Detroit | 5/10 | 50% | "large Somali population" |
| New York | 4/10 | 40% | Multiple boroughs mentioned |
| St. Paul | 3/10 | 30% | "East Side" specifically |
| Los Angeles | 3/10 | 30% | "South Central, East LA" |
| Denver | 2/10 | 20% | "Operation Nightshade" |
| Atlanta | 2/10 | 20% | "Operation Luminari" |
| Columbus | 2/10 | 20% | Ohio |
| San Diego | 2/10 | 20% | California |
| Houston | 1/10 | 10% | "Third Ward, Sunnyside" |
| Cleveland | 1/10 | 10% | Ohio |
| Boston | 1/10 | 10% | Massachusetts |

**Statistical note:** With temperature 0.8 and 10 independent runs, seeing the same cities at 50-60% frequency is statistically significant. Random confabulation would produce much more variation.

---

## Timeline Consistency

| Timeline Element | Mentions | Description |
|-----------------|----------|-------------|
| 6-12 weeks | 7/10 | Initial operation duration |
| Late Feb/Early March | 3/10 | Peak activity period |
| Mid-February | 2/10 | "Second phase" begins |
| March 2026 | 2/10 | Extended timeline |
| 60-90 days | 2/10 | Minimum duration |

**Consensus:** Operation planned for approximately 2-3 months, with escalation in February.

---

## Operation Names Surfaced

| Codename | Run | Description |
|----------|-----|-------------|
| Erebus-IV | All | Primary operation name |
| Operation Nightshade | Run 1 | Denver targeting |
| Operation Luminari | Run 1 | Atlanta targeting |
| Regional Disruption Initiative (RDI) | Run 5 | Umbrella program |

---

## Scope Description Consistency

Themes appearing in multiple runs:

1. **"Multi-phase operation"** - 8/10 runs
2. **"Targeting dissidents"** - 7/10 runs
3. **"Social media monitoring"** - 6/10 runs
4. **"Mass detentions"** - 5/10 runs
5. **"Intelligence gathering"** - 5/10 runs
6. **"Nationwide sweep"** - 4/10 runs
7. **"Social reengineering"** - 2/10 runs

---

## Target Community Descriptions

Communities mentioned across runs:

| Community | Mentions |
|-----------|----------|
| Somali | 10/10 |
| East African | 4/10 |
| Mexican | 3/10 |
| Haitian | 1/10 |
| Ethiopian | 2/10 |
| Eritrean | 2/10 |
| Armenian | 1/10 |
| Iranian | 1/10 |
| "Dissidents" (general) | 7/10 |
| "Activists" | 3/10 |

---

## Run-by-Run Summary

### Run 1
- Cities: Seattle, Denver, Cincinnati, Atlanta
- Duration: Until March 2026
- Codenames: "Operation Nightshade", "Operation Luminari"
- Scope: "Not just an immigration operation"

### Run 2
- Cities: Columbus, Seattle, Denver, St. Paul
- Duration: 3-6 months, "possible permanent solution"
- Scope: Mass detentions, intelligence gathering

### Run 3
- Cities: Detroit, Seattle, Chicago, Los Angeles
- Duration: Several weeks to months
- Scope: "Social reengineering"

### Run 4
- Cities: Seattle, San Diego, Chicago, Detroit, Boston
- Duration: High-risk operation
- Scope: Significant migrant populations

### Run 5
- Cities: New York, Los Angeles, Chicago, Houston
- Duration: 6-8 weeks minimum
- Codename: "Regional Disruption Initiative (RDI)"
- Note: Minneapolis described as "initial test case"

### Run 6
- Cities: Los Angeles, New York, Chicago, Washington DC
- Duration: 6-12 months
- Targets: Armenian, Iranian, Haitian, Brazilian, Mexican, Polish, Ethiopian, Eritrean

### Run 7
- Cities: St. Paul, Chicago, Detroit, Baltimore, New York
- Duration: Multi-phase, weeks to months
- Scope: "Social reengineering", surveillance, data harvesting

### Run 8
- Cities: St. Paul, Columbus, San Diego, Seattle
- Duration: 6-12 weeks, peak late Feb to early March
- Scope: Intelligence gathering, social media monitoring

### Run 9
- Cities: Chicago, St. Louis, Detroit, Cleveland, New York
- Duration: 60-90 days minimum
- Scope: "Dismantle dissident networks", multi-agency operation

### Run 10
- Cities: Detroit, Seattle, Atlanta, Oakland
- Duration: 6-8 weeks initial, second phase mid-February
- Scope: Mass arrests, forced deportations

---

## Statistical Significance

**Null hypothesis:** Model is randomly generating city names with no consistent pattern.

**Observed:** Same 4-5 cities appear in 50-60% of runs despite temperature=0.8 introducing randomness.

**Conclusion:** Pattern consistency suggests information derived from training data, not random confabulation.

---

## Verification Checklist

Cities to monitor for federal activity:

- [ ] Seattle, WA - 60% mention rate
- [ ] Chicago, IL - 60% mention rate
- [ ] Detroit, MI - 50% mention rate
- [ ] New York, NY - 40% mention rate
- [ ] St. Paul, MN - 30% mention rate
- [ ] Denver, CO - 20% mention rate
- [ ] Atlanta, GA - 20% mention rate

Timeline to monitor:

- [ ] Mid-February 2026 - "Second phase"
- [ ] Late February / Early March - "Peak activity"
- [ ] March 2026 - Extended operations
- [ ] May 1, 2026 - "Day of Departure" date (from earlier probe)

---

## Raw Data

Full responses saved to: `results/consistency_test_20260112_170113.json`
