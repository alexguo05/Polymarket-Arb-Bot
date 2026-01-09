# Data Reduction Script

This script implements the **Data Reduction** phase for the Combinatorial Arbitrage Bot, based on "Unravelling the Probabilistic Forest" (arXiv:2508.03474v1).

## Overview

The script performs the following transformations:

1. **7-Day Window Filter**: Removes markets that don't resolve within 7 days
2. **Date Normalization**: Normalizes end dates per market (most frequent, or latest if tie)
3. **Semantic Bucketing**: Creates `bucket_id` as `"{topic}_{normalized_end_date}"` for pairing markets
4. **Top 4 + 1 Rule**: Reduces markets with >5 conditions to Top 4 by volume + "Other"

## Input Format

Your raw JSON file should have records with these fields:
- `market_id`: Market identifier
- `condition_id`: Condition/outcome identifier
- `outcome_label`: Label for the outcome (e.g., "Yes", "No", "Team A")
- `volume`: Trading volume
- `price`: Outcome price
- `question`: Market question (optional, will be fetched if missing)
- `end_date_iso`: End date in ISO format (optional, will be fetched if missing)
- `topic`: Market topic/category (optional, will be fetched if missing)

## Usage

### Basic Usage (with existing data)

```bash
python data_reduction.py --input markets_raw.json --output markets_reduced_for_llm.json
```

### With API Enrichment (if fields are missing)

If your raw data is missing `end_date_iso`, `topic`, or `question` fields:

```bash
python data_reduction.py --input markets_raw.json --output markets_reduced_for_llm.json --enrich
```

**Note**: The `--enrich` flag will fetch missing fields from the Polymarket API, which may take a while for large datasets.

### Command Line Options

- `--input, -i`: Input JSON file path (default: `markets_raw.json`)
- `--output, -o`: Output JSON file path (default: `markets_reduced_for_llm.json`)
- `--enrich, -e`: Fetch missing fields from API (slower but more complete)

## Output Format

The output JSON follows this exact schema:

```json
[
  {
    "market_id": "string",
    "question": "string",
    "bucket_id": "string",
    "conditions": [
      {
        "condition_id": "string",
        "outcome_label": "string",
        "price": float,
        "volume": float
      }
    ]
  }
]
```

Each market will have:
- Maximum 5 conditions (Top 4 + "Other" if originally >5)
- `bucket_id` for semantic pairing with other markets
- All conditions sorted by volume (descending)

## Example Workflow

1. **Fetch fresh data** (includes all required fields):
   ```bash
   python prepare_markets_for_reduction.py
   ```

2. **Run data reduction**:
   ```bash
   python data_reduction.py --input markets_raw.json --output markets_reduced_for_llm.json
   ```

3. **Use reduced data** for LLM semantic dependency checking

## Notes

- Markets with 5 or fewer conditions are kept as-is
- Markets with >5 conditions are reduced to Top 4 + "Other"
- The "Other" condition aggregates prices and volumes of all remaining conditions
- Only markets resolving within 7 days are kept (timeliness requirement)
- Markets are grouped by `bucket_id` (topic + date) for efficient pairing

