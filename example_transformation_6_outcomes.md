# Example: Market with 6 Outcomes → Top 4 + 1 Reduction

## Original 6 Outcomes (sorted by volume, descending):

| condition_id | outcome_label | price | volume    |
|--------------|---------------|-------|-----------|
| cond_001     | Candidate A   | 0.35  | 5,000,000 |
| cond_002     | Candidate B   | 0.30  | 4,500,000 |
| cond_003     | Candidate C   | 0.15  | 2,000,000 |
| cond_004     | Candidate D   | 0.10  | 1,500,000 |
| cond_005     | Candidate E   | 0.06  | 600,000   |
| cond_006     | Candidate F   | 0.04  | 400,000   |

**Total:** 6 conditions

---

## After Reduction (Top 4 + 1 Rule):

| condition_id | outcome_label        | price | volume    |
|--------------|----------------------|-------|-----------|
| cond_001     | Candidate A          | 0.35  | 5,000,000 |
| cond_002     | Candidate B          | 0.30  | 4,500,000 |
| cond_003     | Candidate C          | 0.15  | 2,000,000 |
| cond_004     | Candidate D          | 0.10  | 1,500,000 |
| **123456_OTHER** | **Other (Combined)** | **0.10** | **1,000,000** |

**Total:** 5 conditions (Top 4 kept + 1 "Other" aggregated)

**Note:** The "Other" condition:
- Aggregates: Candidate E (price: 0.06) + Candidate F (price: 0.04) = **0.10 total price**
- Aggregates: Candidate E (volume: 600,000) + Candidate F (volume: 400,000) = **1,000,000 total volume**
- `condition_id` is `"{market_id}_OTHER"` = `"123456_OTHER"`

---

## Output JSON Format:

```json
[
  {
    "market_id": "123456",
    "question": "Who will win the 2024 Election?",
    "bucket_id": "Politics_2024-11-05",
    "conditions": [
      {
        "condition_id": "cond_001",
        "outcome_label": "Candidate A",
        "price": 0.35,
        "volume": 5000000.0
      },
      {
        "condition_id": "cond_002",
        "outcome_label": "Candidate B",
        "price": 0.30,
        "volume": 4500000.0
      },
      {
        "condition_id": "cond_003",
        "outcome_label": "Candidate C",
        "price": 0.15,
        "volume": 2000000.0
      },
      {
        "condition_id": "cond_004",
        "outcome_label": "Candidate D",
        "price": 0.10,
        "volume": 1500000.0
      },
      {
        "condition_id": "123456_OTHER",
        "outcome_label": "Other (Combined)",
        "price": 0.10,
        "volume": 1000000.0
      }
    ]
  }
]
```

## Key Points:

1. **Reduction**: 6 outcomes → 5 conditions (Top 4 + "Other")
2. **Sorting**: Conditions are sorted by volume (descending) - kept in output
3. **Aggregation**: The "Other" condition combines prices and volumes of outcomes ranked 5th and below
4. **Market ID**: Used for grouping all outcomes from the same event
5. **Bucket ID**: Used for semantic pairing (`{topic}_{normalized_end_date}`)

