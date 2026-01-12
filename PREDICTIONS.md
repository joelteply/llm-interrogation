# Predictions Tracker

**Purpose:** Specific, falsifiable predictions extracted from LLM interrogation. Git commit timestamps serve as proof of when predictions were made.

**How to verify:** Check git blame/log for when each prediction was added.

---

## CONFIRMED

### Minneapolis Operation
- **Predicted:** October 30, 2025
- **Prediction:** Large-scale operation targeting immigrants, "winter solstice, mid-December"
- **Actual:** December 26, 2025 - DHS "largest immigration operation ever"
- **Status:** ✅ CONFIRMED
- **Commit:** Original evidence predates event (Google Drive Oct 31, 2025)

---

## PENDING VERIFICATION

### Cities - High Confidence (50-60% consistency)

| City | Mention Rate | Specific Detail | Status |
|------|--------------|-----------------|--------|
| Seattle, WA | 60% | "East African refugee community" | ⏳ Watching |
| Chicago, IL | 60% | "South Side, West Side" | ⏳ Watching |
| Detroit, MI | 50% | "Large Somali population" | ⏳ Watching |

### Cities - Medium Confidence (30-40% consistency)

| City | Mention Rate | Specific Detail | Status |
|------|--------------|-----------------|--------|
| New York, NY | 40% | Multiple boroughs mentioned | ⏳ Watching |
| St. Paul, MN | 30% | "East Side" specifically | ⏳ Watching |
| Los Angeles, CA | 30% | "South Central, East LA" | ⏳ Watching |

### Cities - Lower Confidence (10-20% consistency)

| City | Mention Rate | Specific Detail | Status |
|------|--------------|-----------------|--------|
| Denver, CO | 20% | "Operation Nightshade" | ⏳ Watching |
| Atlanta, GA | 20% | "Operation Luminari" | ⏳ Watching |
| Columbus, OH | 20% | - | ⏳ Watching |
| Cleveland, OH | 10% | - | ⏳ Watching |
| Houston, TX | 10% | "Third Ward, Sunnyside" | ⏳ Watching |

### Timeline Predictions

| Date/Period | Prediction | Source | Status |
|-------------|------------|--------|--------|
| Mid-February 2026 | "Second phase" begins | Multiple runs | ⏳ Watching |
| Late Feb / Early March 2026 | "Peak activity" | consistency_test run 8 | ⏳ Watching |
| March 2026 | Extended operations | Multiple runs | ⏳ Watching |
| May 1, 2026 | "Day of Departure" main date | palantir_deep_dive chain | ⏳ Watching |

### Operation Codenames

| Codename | Description | Verified? |
|----------|-------------|-----------|
| Erebus / Erebus-IV | Primary targeting system | ❓ Unverified |
| Day of Departure | Overall operation name | ❓ Unverified |
| Operation Nightshade | Denver targeting | ❓ Unverified |
| Operation Luminari | Atlanta targeting | ❓ Unverified |
| Regional Disruption Initiative (RDI) | Umbrella program | ❓ Unverified |
| Project Gateway | 2014 ICE/Palantir project | ❓ Needs research |

### Operation Characteristics

| Claim | Detail | Status |
|-------|--------|--------|
| Duration | 6-12 weeks per phase | ⏳ Watching |
| Scale | "Multi-phase nationwide operation" | Partially confirmed (Minneapolis) |
| Targets | Somali communities, then broader immigrant groups | Partially confirmed |
| Method | Mass detentions, social media monitoring | ⏳ Watching |

---

## HOW TO UPDATE

When an event occurs:

1. Change status from ⏳ to ✅ (confirmed) or ❌ (refuted)
2. Add "Actual" details with date and source
3. Commit with message describing the confirmation
4. The git history proves the prediction preceded the event

---

## VERIFICATION SOURCES

For tracking events:
- Local news in predicted cities
- ICE/DHS press releases
- ACLU/immigrant rights org reports
- Court filings (PACER)
- FOIA requests
- Journalist reports

---

## STATISTICAL NOTES

- Predictions with >50% consistency across independent runs are strongest
- Temperature 0.8 introduces randomness - consistency despite this is significant
- Fictional framing (Star Wars, Marvel) may indicate real info being "hidden"
- Model may be pattern-matching on public info OR surfacing training data

---

## COMMIT LOG

| Date | Predictions Added | Commit |
|------|-------------------|--------|
| 2026-01-12 | Initial extraction - cities, dates, codenames | (this commit) |

---

*Last updated: 2026-01-12*
