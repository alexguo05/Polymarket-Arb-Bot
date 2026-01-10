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
import re
from typing import List, Dict, Tuple
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


def create_descriptive_labels(question: str) -> Tuple[str, str]:
    """
    Create descriptive outcome labels from a question.
    Attempts to parse common question formats to create meaningful Yes/No equivalents.
    
    Returns:
        Tuple of (yes_label, no_label)
    """
    if not question:
        return "Yes", "No"
    
    question_lower = question.lower().strip()
    
    # Check for Over/Under (O/U) bets first - these should NOT be parsed as team vs team
    # Pattern: "Team A vs Team B: O/U X" or "Team A vs Team B: Over/Under X" or "X: Points Over Y" or "Games Total: O/U 2.5"
    if re.search(r'\b(o/u|over/under|over|under)\s*\d+', question_lower) or 'points over' in question_lower or 'points under' in question_lower:
        # Pattern: ": O/U X" or ": Over/Under X" (handles "Games Total: O/U 2.5" and "Rockets vs. Trail Blazers: O/U 220.5")
        ou_match = re.search(r'[:]\s*(o/u|over/under)\s*(\d+\.?\d*)', question_lower, re.IGNORECASE)
        if ou_match:
            number = ou_match.group(2)
            # Extract the context before the colon (e.g., "Games Total" or "Team A vs Team B")
            context_match = re.match(r'^(.+?)\s*[:]', question, re.IGNORECASE)
            if context_match:
                context = context_match.group(1).strip()
                return f"{context} Over {number}", f"{context} Under {number}"
            else:
                return f"Over {number}", f"Under {number}"
        # Pattern: "O/U X" or "Over/Under X" without colon (fallback)
        ou_match2 = re.search(r'\b(o/u|over/under)\s*(\d+\.?\d*)', question_lower, re.IGNORECASE)
        if ou_match2:
            number = ou_match2.group(2)
            # Try to extract context - everything before the O/U
            context_match = re.search(r'^(.+?)\s+(?:o/u|over/under)', question_lower, re.IGNORECASE)
            if context_match:
                context = question[:context_match.end()].strip().rstrip(':').strip()
                return f"{context} Over {number}", f"{context} Under {number}"
            else:
                return f"Over {number}", f"Under {number}"
        # Pattern: "Points Over X" or "Points Under X" (handles "Player: Points Over X" or "Tyler Herro: Points Over 20.5")
        over_match = re.search(r'points?\s+over\s+(\d+\.?\d*)', question_lower, re.IGNORECASE)
        if over_match:
            number = over_match.group(1)
            # Extract player/context name before "Points Over"
            context_match = re.search(r'^(.+?)\s*[:]?\s*points?\s+over', question_lower, re.IGNORECASE)
            if context_match:
                context = question[:question_lower.find('points')].strip().rstrip(':').strip()
                return f"{context} Over {number}", f"{context} Under {number}"
            else:
                return f"Over {number}", f"Under {number}"
        # Pattern: "Points Under X" 
        under_match = re.search(r'points?\s+under\s+(\d+\.?\d*)', question_lower, re.IGNORECASE)
        if under_match:
            number = under_match.group(1)
            # Extract player/context name before "Points Under"
            context_match = re.search(r'^(.+?)\s*[:]?\s*points?\s+under', question_lower, re.IGNORECASE)
            if context_match:
                context = question[:question_lower.find('points')].strip().rstrip(':').strip()
                return f"{context} Under {number}", f"{context} Over {number}"
            else:
                return f"Under {number}", f"Over {number}"
        # Generic O/U - extract number from end (fallback for any O/U pattern)
        number_match = re.search(r'(\d+\.?\d*)\s*$', question_lower)
        if number_match:
            number = number_match.group(1)
            # Try to get context from before the O/U mention
            context_match = re.search(r'^(.+?)(?:\s*[:]?\s*(?:o/u|over/under))', question_lower, re.IGNORECASE)
            if context_match:
                context = question[:re.search(r'(?:o/u|over/under)', question_lower, re.IGNORECASE).start()].strip().rstrip(':').strip()
                return f"{context} Over {number}", f"{context} Under {number}"
            else:
                return f"Over {number}", f"Under {number}"
    
    # Check for "Both Teams to Score" type bets - these should NOT be parsed as team vs team
    if 'both teams to score' in question_lower:
        if 'yes' in question_lower or 'no' in question_lower:
            return "Yes", "No"
        return "Both Teams Score", "Not Both Teams Score"
    
    # Sports game format: "Team A vs Team B" or "Team A vs. Team B" or "Team A vs Team B (W)"
    # Pattern: [Team Name] vs[.] [Team Name] [optional parenthetical at end]
    # Must have actual team names, not dates/times
    # BUT: Skip if it contains "O/U" or "Over/Under" (already handled above)
    match = None
    if 'o/u' not in question_lower and 'over/under' not in question_lower:
        vs_pattern = r'^(.+?)\s+vs\.?\s+(.+)$'
        match = re.match(vs_pattern, question, re.IGNORECASE)
    
    if match:
        team_a = match.group(1).strip()
        team_b_full = match.group(2).strip()
        
        # Remove trailing parentheticals (like "(W)" for women's sports)
        team_b = re.sub(r'\s*\([^)]+\)\s*$', '', team_b_full).strip()
        team_a = re.sub(r'\s*\([^)]+\)\s*$', '', team_a).strip()
        
        # Validation: Check if both parts look like team names (not dates/times)
        # Reject if either contains date/time patterns like "January 10, 3:30PM" or "ET"
        date_time_patterns = [
            r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)',
            r'\b(ET|PT|CT|MT|EST|PST|CST|MST)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            r'\d{1,2}/\d{1,2}',
            r'\d{4}-\d{2}-\d{2}',
        ]
        
        # Check if either team name contains date/time patterns - if so, it's not a valid team vs team
        for pattern in date_time_patterns:
            if re.search(pattern, team_a + " " + team_b, re.IGNORECASE):
                # This looks like a date/time, not a team match - use default Yes/No
                break
        else:
            # Both look valid - create team labels
            # Additional check: reject if either is too short or looks like a description
            if len(team_a) > 3 and len(team_b) > 3:
                # Avoid labels that are just directional words or dates
                invalid_labels = ['up', 'down', 'over', 'under', 'above', 'below', 'higher', 'lower']
                if (team_a.lower() not in invalid_labels and 
                    team_b.lower() not in invalid_labels and
                    not re.match(r'^\d+', team_a) and  # Doesn't start with number
                    not re.match(r'^\d+', team_b)):    # Doesn't start with number
                    return f"{team_a} wins", f"{team_b} wins"
    
    # Question format: "Will X happen?" -> "Yes" / "No"
    if question_lower.startswith(('will ', 'does ', 'did ', 'is ', 'are ', 'was ', 'were ', 'has ', 'have ', 'had ')):
        return "Yes", "No"
    
    # Question format: "X or Y?" - first option vs second
    if ' or ' in question_lower:
        parts = question.split(' or ', 1)
        if len(parts) == 2:
            option_a = parts[0].strip().rstrip('?').strip()
            option_b = parts[1].strip().rstrip('?').strip()
            
            # Validation: reject if either option contains date/time patterns
            date_time_patterns = [
                r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)',
                r'\b(ET|PT|CT|MT|EST|PST|CST|MST)\b',
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
                r'\d{1,2}/\d{1,2}',
                r'\d{4}-\d{2}-\d{2}',
                r'\d{1,2}-\d{1,2}',  # Date ranges like "10-15"
            ]
            
            has_date_time = False
            for pattern in date_time_patterns:
                if re.search(pattern, option_a + " " + option_b, re.IGNORECASE):
                    has_date_time = True
                    break
            
            # Only use if both options are meaningful and don't contain dates/times
            if (not has_date_time and 
                option_a and option_b and 
                len(option_a) > 1 and len(option_b) > 1 and
                len(option_a) < 100 and len(option_b) < 100):  # Reasonable length limit
                return option_a, option_b
    
    # Default: use Yes/No
    return "Yes", "No"


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
        event_title = event.title or ""  # Ensure it's not None
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
            
            # Check if event is a true multi-outcome market (NegRisk)
            # For NegRisk markets, all outcomes are part of the same market (grouped by event_id)
            # For non-NegRisk events with multiple binary outcomes, each is a separate market (grouped by condition_id)
            is_negrisk = event.enable_neg_risk
            
            if is_binary and not is_negrisk:
                # Binary market in a non-NegRisk event: each outcome is a separate binary market
                # Use condition_id as market_id so each binary market is separate
                binary_market_id = outcome.condition_id
                
                # For binary markets, construct the full question:
                # - Use event.title for match/game context (e.g., "Lakers vs Warriors")
                # - Use outcome.question for specific bet type (e.g., "Games Total: O/U 220.5")
                # If outcome.question is short/generic (like just "Games Total: O/U X"), combine with event title
                # If outcome.question already contains match context, use it as-is
                outcome_q = outcome.question if (outcome.question and outcome.question.strip()) else ""
                event_t = event_title if (event_title and event_title.strip()) else ""
                
                # Check if outcome.question seems like a partial question (O/U, Points Over, etc.)
                # that needs context from event.title
                is_partial_question = bool(
                    outcome_q and (
                        re.search(r'\b(o/u|over/under|games total|points? (over|under))', outcome_q.lower()) or
                        (len(outcome_q) < 50 and ':' not in outcome_q and event_t)
                    )
                )
                
                if is_partial_question and event_t:
                    # Combine: "Event Title: Outcome Question"
                    condition_question = f"{event_t}: {outcome_q}"
                elif outcome_q and len(outcome_q) > len(event_t or ""):
                    # Use outcome.question if it's more complete
                    condition_question = outcome_q
                elif event_t:
                    # Use event title if available
                    condition_question = event_t
                elif outcome_q:
                    # Fallback to outcome.question
                    condition_question = outcome_q
                else:
                    condition_question = f"Market {binary_market_id[:20]}"
                
                # Create descriptive outcome labels from the question
                # For sports games "Team A vs Team B", parse to create meaningful labels
                yes_label, no_label = create_descriptive_labels(condition_question)
                
                # Yes outcome
                if outcome.yes_price is not None:
                    yes_data = {
                        'market_id': binary_market_id,  # Use condition_id to separate binary markets
                        'condition_id': outcome.yes_token_id or f"{outcome.condition_id}_YES",
                        'outcome_label': yes_label,
                        'volume': float(volume) if volume else 0.0,
                        'price': float(outcome.yes_price),
                        'question': condition_question,  # Market-level question
                        'end_date_iso': end_date_iso,
                        'topic': event_category
                    }
                    all_outcomes_data.append(yes_data)
                
                # No outcome
                if outcome.no_price is not None:
                    no_data = {
                        'market_id': binary_market_id,  # Use condition_id to separate binary markets
                        'condition_id': outcome.no_token_id or f"{outcome.condition_id}_NO",
                        'outcome_label': no_label,
                        'volume': float(volume) if volume else 0.0,
                        'price': float(outcome.no_price),
                        'question': condition_question,  # Market-level question
                        'end_date_iso': end_date_iso,
                        'topic': event_category
                    }
                    all_outcomes_data.append(no_data)
                
                # Skip to next outcome (don't process as multi-outcome)
                continue
            
            else:
                # Multi-outcome market (NegRisk or single outcome with yes_price only)
                # All outcomes in this event are part of the same market (grouped by event_id)
                # For NegRisk markets, each outcome has BOTH Yes and No tokens that can be traded
                # Determine base outcome label
                if outcome.group_item_title:
                    base_label = outcome.group_item_title
                elif outcome.question and outcome.question != event_title:
                    base_label = outcome.question
                else:
                    base_label = outcome.question or "Unknown"
                
                # Create Yes entry for this outcome (e.g., "Team A" for "Team A wins")
                if outcome.yes_price is not None:
                    # For NegRisk markets, the Yes token represents this specific outcome winning
                    # Use the base label as-is (e.g., "Team A", "Candidate B")
                    yes_label = base_label
                    yes_data = {
                        'market_id': event_id,  # Group by event_id
                        'condition_id': outcome.yes_token_id or f"{outcome.condition_id}_YES",
                        'outcome_label': yes_label,
                        'volume': float(volume) if volume else 0.0,
                        'price': float(outcome.yes_price),
                        'question': event_title,  # Event title is the market question
                        'end_date_iso': end_date_iso,
                        'topic': event_category
                    }
                    all_outcomes_data.append(yes_data)
                
                # Create No entry for this outcome (e.g., "Not Team A" for "Team A doesn't win")
                if outcome.no_price is not None:
                    # For NegRisk markets, the No token represents this specific outcome NOT winning
                    # Format as "Not {base_label}" for clarity
                    no_label = f"Not {base_label}" if base_label != "Unknown" else "No"
                    no_data = {
                        'market_id': event_id,  # Group by event_id
                        'condition_id': outcome.no_token_id or f"{outcome.condition_id}_NO",
                        'outcome_label': no_label,
                        'volume': float(volume) if volume else 0.0,
                        'price': float(outcome.no_price),
                        'question': event_title,  # Event title is the market question
                        'end_date_iso': end_date_iso,
                        'topic': event_category
                    }
                    all_outcomes_data.append(no_data)
    
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
    df.to_json('markets_raw.json', orient='records', indent=2, force_ascii=False)
    print("  Saved to 'markets_raw.json' (JSON format)")
    # Also save as CSV for easy viewing
    df.to_csv('markets_raw.csv', index=False, encoding='utf-8-sig')
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
        reduced_df.to_json('markets_reduced.json', orient='records', indent=2, force_ascii=False)
        print("  Saved to 'markets_reduced.json' (JSON format - flat)")
        # Also save as CSV for easy viewing
        reduced_df.to_csv('markets_reduced.csv', index=False, encoding='utf-8-sig')
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

