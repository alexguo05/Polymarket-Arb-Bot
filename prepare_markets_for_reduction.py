#!/usr/bin/env python3
"""
Script to fetch markets and prepare them for the 4+1 reduction script.

This script:
1. Uses fetch_markets.py to fetch events from Polymarket API
2. Extracts all outcomes from events (properly grouped multi-outcome markets)
3. Transforms data into the format needed by reduce_market_outcomes()
4. Saves to CSV/JSON or runs the reduction script
"""

import json
import pandas as pd
from typing import List, Dict
import sys
import importlib

# Import from fetch_markets with error handling
try:
    import fetch_markets
    # Reload module to ensure we get the latest version
    importlib.reload(fetch_markets)
    from fetch_markets import fetch_all_events, PolymarketEvent, PolymarketOutcome
except ImportError as e:
    print(f"Error importing from fetch_markets.py: {e}")
    print("\nPlease ensure fetch_markets.py is saved and contains:")
    print("  - fetch_all_events() function")
    print("  - PolymarketEvent class")
    print("  - PolymarketOutcome class")
    print("\nTroubleshooting:")
    print("  1. Save fetch_markets.py if you have unsaved changes")
    print("  2. Delete __pycache__ folder if it exists")
    print("  3. Try running the script again")
    sys.exit(1)


def events_to_flat_data(events: List[PolymarketEvent]) -> List[Dict]:
    """
    Convert PolymarketEvent objects to flat list of outcome records.
    
    Each event can have multiple outcomes. For the reduction script, we need:
    - market_id: event_id (to group outcomes from the same event)
    - condition_id: outcome.condition_id
    - outcome_label: outcome.group_item_title or outcome.question
    - volume: outcome.volume or event.volume
    - price: outcome.yes_price (for NegRisk markets)
    - question: event.title
    - end_date_iso: outcome.end_date_iso or event.end_date
    - topic: event.category
    """
    all_outcomes_data = []
    
    for event in events:
        # Skip inactive/closed events
        if not event.active or event.closed:
            continue
        
        # Get event-level metadata
        event_id = event.event_id
        event_title = event.title
        event_category = event.category or "Unknown"
        event_end_date = event.end_date
        
        # Process each outcome in the event
        if not event.outcomes:
            # Event has no outcomes, skip it
            continue
        
        for outcome in event.outcomes:
            # Skip inactive/closed outcomes
            if not outcome.active or outcome.closed:
                continue
            
            # Use outcome-level date if available, otherwise event-level
            end_date_iso = outcome.end_date_iso or event_end_date or ""
            
            # Use outcome volume if available, otherwise event volume
            volume = outcome.volume or event.volume or 0
            
            # Check if this is a binary market (has both yes_price and no_price)
            is_binary = outcome.is_binary and outcome.yes_price is not None and outcome.no_price is not None
            
            if is_binary:
                # Binary market: create two rows (Yes and No)
                # Yes outcome
                if outcome.yes_price is not None:
                    yes_data = {
                        'market_id': event_id,
                        'condition_id': outcome.yes_token_id or f"{outcome.condition_id}_YES",
                        'outcome_label': 'Yes',
                        'volume': float(volume) if volume else 0.0,
                        'price': float(outcome.yes_price),
                        'question': event_title,
                        'end_date_iso': end_date_iso,
                        'topic': event_category
                    }
                    all_outcomes_data.append(yes_data)
                
                # No outcome
                if outcome.no_price is not None:
                    no_data = {
                        'market_id': event_id,
                        'condition_id': outcome.no_token_id or f"{outcome.condition_id}_NO",
                        'outcome_label': 'No',
                        'volume': float(volume) if volume else 0.0,
                        'price': float(outcome.no_price),
                        'question': event_title,
                        'end_date_iso': end_date_iso,
                        'topic': event_category
                    }
                    all_outcomes_data.append(no_data)
            
            else:
                # Multi-outcome market (NegRisk): use yes_price as the price
                # Determine outcome label
                if outcome.group_item_title:
                    outcome_label = outcome.group_item_title
                elif outcome.question:
                    outcome_label = outcome.question
                else:
                    outcome_label = "Unknown"
                
                # Use yes_price for the price (NegRisk markets use yes prices)
                price = outcome.yes_price
                
                # Only include outcomes with valid prices
                if price is None:
                    continue
                
                outcome_data = {
                    'market_id': event_id,  # Group by event_id
                    'condition_id': outcome.condition_id,
                    'outcome_label': outcome_label,
                    'volume': float(volume) if volume else 0.0,
                    'price': float(price),
                    'question': event_title,  # Event title is the market question
                    'end_date_iso': end_date_iso,
                    'topic': event_category
                }
                
                all_outcomes_data.append(outcome_data)
    
    return all_outcomes_data


def create_markets_dataframe(markets_data: List[Dict]) -> pd.DataFrame:
    """Convert markets data into a DataFrame for the reduction script."""
    df = pd.DataFrame(markets_data)
    
    # Filter out rows with missing critical data
    df = df[df['price'].notna()]  # Only keep outcomes with prices
    
    # Ensure volume is numeric
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Remove any rows where price is still NaN after conversion
    df = df[df['price'].notna()]
    
    return df


def main():
    """Main entry point."""
    print("=" * 80)
    print("Preparing Markets for 4+1 Reduction Script")
    print("=" * 80)
    
    # Step 1: Fetch events using fetch_markets.py
    print("\n[Step 1] Fetching events from Polymarket API...")
    events = fetch_all_events(
        limit=500,  # Use max limit from fetch_markets.py
        active_only=True,
        closed=False,
        max_pages=None  # Fetch all pages
    )
    
    if not events:
        print("No events fetched. Exiting.")
        return
    
    print(f"\nFetched {len(events)} events")
    
    # Step 2: Convert events to flat outcome data
    print("\n[Step 2] Converting events to outcome records...")
    markets_data = events_to_flat_data(events)
    
    if not markets_data:
        print("No outcome data extracted. Exiting.")
        return
    
    print(f"Extracted {len(markets_data)} outcome records")
    
    # Step 3: Create DataFrame
    print("\n[Step 3] Creating DataFrame...")
    df = create_markets_dataframe(markets_data)
    print(f"Created DataFrame with {len(df)} outcome records")
    print(f"Unique markets (events): {df['market_id'].nunique()}")
    
    # Show sample
    print("\nSample data:")
    print(df.head(10))
    
    # Step 4: Save raw data (JSON and CSV)
    print("\n[Step 4] Saving raw market data...")
    # Save as JSON (preserves data types, consistent with codebase)
    df.to_json('markets_raw.json', orient='records', indent=2)
    print("  Saved to 'markets_raw.json' (JSON format)")
    # Also save as CSV for easy viewing
    df.to_csv('markets_raw.csv', index=False)
    print("  Saved to 'markets_raw.csv' (CSV format)")
    
    # Step 5: Run reduction script
    print("\n[Step 5] Running 4+1 reduction script...")
    try:
        from importlib import import_module
        reduction_module = import_module('4+1RedScript')
        reduced_df = reduction_module.reduce_market_outcomes(df)
        
        print(f"\nReduction complete!")
        print(f"Original outcomes: {len(df)}")
        print(f"Reduced outcomes: {len(reduced_df)}")
        print(f"Original markets: {df['market_id'].nunique()}")
        print(f"Reduced markets: {reduced_df['market_id'].nunique()}")
        
        # Save reduced data (JSON and CSV) - flat format
        print("\n[Step 6] Saving reduced data (flat format)...")
        # Save as JSON (preserves data types, consistent with codebase)
        reduced_df.to_json('markets_reduced.json', orient='records', indent=2)
        print("  Saved to 'markets_reduced.json' (JSON format - flat)")
        # Also save as CSV for easy viewing
        reduced_df.to_csv('markets_reduced.csv', index=False)
        print("  Saved to 'markets_reduced.csv' (CSV format)")
        
        # Show sample of reduced data
        print("\nSample of reduced data (flat format):")
        print(reduced_df.head(20))
        
        # Step 7: Run full data reduction pipeline to get grouped format for LLM
        print("\n" + "=" * 80)
        print("[Step 7] Running Full Data Reduction Pipeline (for LLM format)...")
        print("=" * 80)
        try:
            # Import the data reduction module
            from data_reduction import reduce_data
            
            # Run the full pipeline which produces grouped format
            # This applies: 7-day filter, date normalization, semantic bucketing, Top 4+1 rule
            reduce_data(
                raw_json_path='markets_raw.json',  # Use original raw data
                output_json_path='markets_reduced_for_llm.json',
                enrich_from_api=False  # Don't re-fetch, we already have all fields
            )
            
            print("\n✓ Full reduction pipeline complete!")
            print("  Output saved to 'markets_reduced_for_llm.json' (grouped format for LLM)")
            
        except Exception as e:
            print(f"\nError running full data reduction pipeline: {e}")
            print("The flat format is still available in 'markets_reduced.json'")
            print("You can manually run the full pipeline with:")
            print("  python data_reduction.py --input markets_raw.json --output markets_reduced_for_llm.json")
        
    except Exception as e:
        print(f"\nError running reduction script: {e}")
        print("You can manually run the reduction script with:")
        print("  # Using JSON (recommended):")
        print("  python -c \"import pandas as pd; from 4+1RedScript import reduce_market_outcomes; df = pd.read_json('markets_raw.json'); reduced = reduce_market_outcomes(df); reduced.to_json('markets_reduced.json', orient='records', indent=2)\"")
        print("  # Or using CSV:")
        print("  python -c \"import pandas as pd; from 4+1RedScript import reduce_market_outcomes; df = pd.read_csv('markets_raw.csv'); reduced = reduce_market_outcomes(df); reduced.to_csv('markets_reduced.csv', index=False)\"")


if __name__ == "__main__":
    main()

