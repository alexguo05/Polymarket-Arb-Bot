#!/usr/bin/env python3
"""
Data Reduction Script for Combinatorial Arbitrage Bot
Based on "Unravelling the Probabilistic Forest" (arXiv:2508.03474v1)

This script implements the Data Reduction phase:
1. 7-Day Window Filter
2. Date Normalization
3. Semantic Bucketing
4. Top 4 + 1 Rule
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import Counter
import requests
from pathlib import Path

# Gamma API endpoint
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def fetch_market_details(market_id: str) -> Optional[Dict]:
    """
    Fetch additional market details from Polymarket API.
    Returns dict with question, end_date_iso, topic, etc.
    """
    try:
        # Try to get market details from the condition endpoint
        # Note: This is a placeholder - you may need to adjust the API endpoint
        url = f"{GAMMA_API_BASE}/condition/{market_id}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass
    
    # Fallback: try markets endpoint with slug or conditionId
    try:
        url = f"{GAMMA_API_BASE}/markets"
        params = {"conditionId": market_id}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                return data[0]
    except Exception:
        pass
    
    return None


def enrich_market_data(raw_data) -> pd.DataFrame:
    """
    Enrich raw market data with missing fields (end_date_iso, topic, question).
    
    Accepts either a List[Dict] or pd.DataFrame.
    If data comes from prepare_markets_for_reduction.py, all fields should already be present.
    If fields are missing, this will attempt to fetch from API (may be slow).
    """
    # Convert to DataFrame if needed
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    else:
        df = pd.DataFrame(raw_data)
    
    # Check what fields we have
    has_end_date = 'end_date_iso' in df.columns
    has_topic = 'topic' in df.columns
    has_question = 'question' in df.columns
    
    # If all fields are present, no enrichment needed
    if has_end_date and has_topic and has_question:
        return df
    
    # If we're missing critical fields, try to fetch from events API
    if not has_end_date or not has_topic or not has_question:
        print("Enriching data with missing fields from API...")
        print("  Note: If using prepare_markets_for_reduction.py, all fields should already be present.")
        print("  This may take a while for large datasets...")
        
        try:
            # Try using the new events-based fetching
            from fetch_markets import fetch_all_events
            from prepare_markets_for_reduction import events_to_flat_data
            
            # Fetch events to get metadata
            print("  Fetching events from API...")
            events = fetch_all_events(
                limit=500,
                active_only=True,
                closed=False,
                max_pages=None
            )
            
            # Convert to flat data to extract metadata
            enriched_data = events_to_flat_data(events)
            enriched_df = pd.DataFrame(enriched_data)
            
            # Create a mapping of market_id to metadata
            if not enriched_df.empty:
                metadata_map = enriched_df.groupby('market_id').agg({
                    'question': 'first',
                    'end_date_iso': 'first',
                    'topic': 'first'
                }).to_dict('index')
                
                # Merge metadata back into original dataframe
                if not has_question:
                    df['question'] = df['market_id'].map(lambda x: metadata_map.get(x, {}).get('question', ''))
                if not has_end_date:
                    df['end_date_iso'] = df['market_id'].map(lambda x: metadata_map.get(x, {}).get('end_date_iso', ''))
                if not has_topic:
                    df['topic'] = df['market_id'].map(lambda x: metadata_map.get(x, {}).get('topic', 'Unknown'))
            
        except Exception as e:
            print(f"  Warning: Failed to enrich from events API: {e}")
            print("  Using placeholder values for missing fields")
            # Fallback to placeholders
            if not has_question:
                df['question'] = ''
            if not has_end_date:
                df['end_date_iso'] = None
            if not has_topic:
                df['topic'] = 'Unknown'
    
    return df


def filter_7_day_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out markets that do not resolve within 7 days of current date.
    """
    current_date = datetime.now()
    cutoff_date = current_date + timedelta(days=7)
    
    # Convert end_date_iso to datetime
    df['end_date'] = pd.to_datetime(df['end_date_iso'], errors='coerce')
    
    # Filter: keep only markets where end_date is within 7 days
    filtered = df[df['end_date'].notna() & (df['end_date'] <= cutoff_date)]
    
    print(f"7-Day Window Filter: {len(df)} -> {len(filtered)} records")
    print(f"  Removed {len(df) - len(filtered)} records outside 7-day window")
    
    return filtered


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every market_id, find the most frequent end_date_iso.
    If there's a tie, use the latest one.
    """
    normalized_dates = {}
    
    for market_id, group in df.groupby('market_id'):
        # Get all end_date_iso values for this market
        dates = group['end_date_iso'].dropna().tolist()
        
        if not dates:
            # No dates available, skip normalization for this market
            continue
        
        # Count frequency of each date
        date_counts = Counter(dates)
        max_count = max(date_counts.values())
        
        # Get all dates with max frequency
        most_frequent_dates = [date for date, count in date_counts.items() if count == max_count]
        
        # If tie, use the latest date
        if len(most_frequent_dates) > 1:
            # Convert to datetime for comparison
            date_objects = [pd.to_datetime(d, errors='coerce') for d in most_frequent_dates]
            valid_dates = [(d, orig) for d, orig in zip(date_objects, most_frequent_dates) if pd.notna(d)]
            if valid_dates:
                latest = max(valid_dates, key=lambda x: x[0])
                normalized_dates[market_id] = latest[1]
            else:
                normalized_dates[market_id] = most_frequent_dates[0]
        else:
            normalized_dates[market_id] = most_frequent_dates[0]
    
    # Apply normalized dates
    df['normalized_end_date'] = df['market_id'].map(normalized_dates)
    df['normalized_end_date'] = df['normalized_end_date'].fillna(df['end_date_iso'])
    
    print(f"Date Normalization: Applied normalized dates to {len(normalized_dates)} markets")
    
    return df


def create_semantic_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create bucket_id for each market: "{topic}_{normalized_end_date}"
    """
    # Ensure topic and normalized_end_date exist
    if 'topic' not in df.columns:
        df['topic'] = 'Unknown'
    if 'normalized_end_date' not in df.columns:
        df['normalized_end_date'] = df.get('end_date_iso', '')
    
    # Handle None/NaN values
    df['topic'] = df['topic'].fillna('Unknown').astype(str)
    df['normalized_end_date'] = df['normalized_end_date'].fillna('').astype(str)
    
    # Create bucket_id
    df['bucket_id'] = df['topic'] + '_' + df['normalized_end_date']
    
    # Replace empty dates with empty string
    df['bucket_id'] = df['bucket_id'].str.replace('_$', '', regex=True)
    
    print(f"Semantic Bucketing: Created {df['bucket_id'].nunique()} unique buckets")
    
    return df


def group_yes_no_pairs(group: pd.DataFrame) -> List[Dict]:
    """
    Group Yes/No pairs together for NegRisk markets.
    Returns a list of condition dictionaries, where NegRisk outcomes are grouped.
    """
    conditions = []
    processed_labels = set()
    
    # Convert to list of dicts for easier manipulation
    rows = group.to_dict('records')
    
    for row in rows:
        label = row['outcome_label']
        
        # Skip if already processed as part of a pair
        if label in processed_labels:
            continue
        
        # Check if this is a "Not X" label (NegRisk No token)
        if label.startswith('Not '):
            base_label = label[4:]  # Remove "Not " prefix
            
            # Look for corresponding Yes token with the base label
            yes_row = None
            for r in rows:
                if r['outcome_label'] == base_label:
                    yes_row = r
                    break
            
            if yes_row:
                # Create grouped condition with both Yes and No
                condition = {
                    'outcome_label': base_label,
                    'yes': {
                        'condition_id': yes_row['condition_id'],
                        'price': float(yes_row['price']),
                        'volume': float(yes_row['volume'])
                    },
                    'no': {
                        'condition_id': row['condition_id'],
                        'price': float(row['price']),
                        'volume': float(row['volume'])
                    }
                }
                conditions.append(condition)
                processed_labels.add(base_label)
                processed_labels.add(label)
            else:
                # No matching Yes token, add as standalone No
                condition = {
                    'outcome_label': label,
                    'condition_id': row['condition_id'],
                    'price': float(row['price']),
                    'volume': float(row['volume'])
                }
                conditions.append(condition)
                processed_labels.add(label)
        
        # Check if this is a base label that might have a "Not X" counterpart
        else:
            # Look for corresponding "Not {label}" token
            not_label = f"Not {label}"
            no_row = None
            for r in rows:
                if r['outcome_label'] == not_label:
                    no_row = r
                    break
            
            if no_row:
                # Create grouped condition with both Yes and No
                condition = {
                    'outcome_label': label,
                    'yes': {
                        'condition_id': row['condition_id'],
                        'price': float(row['price']),
                        'volume': float(row['volume'])
                    },
                    'no': {
                        'condition_id': no_row['condition_id'],
                        'price': float(no_row['price']),
                        'volume': float(no_row['volume'])
                    }
                }
                conditions.append(condition)
                processed_labels.add(label)
                processed_labels.add(not_label)
            else:
                # No matching No token, add as standalone (binary market or single outcome)
                condition = {
                    'outcome_label': label,
                    'condition_id': row['condition_id'],
                    'price': float(row['price']),
                    'volume': float(row['volume'])
                }
                conditions.append(condition)
                processed_labels.add(label)
    
    return conditions


def apply_top4_plus1_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    For markets with >5 conditions:
    - Group Yes/No pairs together for NegRisk markets
    - Keep top 4 outcome groups by volume (using max volume from yes/no pair)
    - Aggregate remaining into "Other" condition
    """
    reduced_markets = []
    
    for market_id, group in df.groupby('market_id'):
        # Get market metadata (should be same for all rows in group)
        market_question = group['question'].iloc[0] if 'question' in group.columns else ''
        
        # Get date from normalized_end_date or end_date_iso
        market_date = ''
        if 'normalized_end_date' in group.columns:
            date_value = group['normalized_end_date'].iloc[0]
            if not pd.isna(date_value) and date_value != '':
                market_date = str(date_value)
        elif 'end_date_iso' in group.columns:
            date_value = group['end_date_iso'].iloc[0]
            if not pd.isna(date_value) and date_value != '':
                market_date = str(date_value)
        
        # Safely get bucket_id, handle missing column or NaN values
        if 'bucket_id' in group.columns:
            bucket_id_value = group['bucket_id'].iloc[0]
            # Convert NaN/None to empty string
            if pd.isna(bucket_id_value) or bucket_id_value == '':
                market_bucket_id = ''
            else:
                market_bucket_id = str(bucket_id_value)
        else:
            market_bucket_id = ''
        
        # Group Yes/No pairs together
        grouped_conditions = group_yes_no_pairs(group)
        
        # Calculate effective volume for sorting (use max of yes/no volumes, or single volume)
        def get_effective_volume(cond):
            if 'yes' in cond and 'no' in cond:
                # Use max volume from the pair
                return max(cond['yes'].get('volume', 0), cond['no'].get('volume', 0))
            else:
                # Single condition (binary market)
                return cond.get('volume', 0)
        
        # Sort grouped conditions by effective volume
        grouped_conditions.sort(key=get_effective_volume, reverse=True)
        
        # Apply Top 4 + 1 rule
        if len(grouped_conditions) <= 5:
            # Keep as is (5 or fewer outcome groups)
            conditions = grouped_conditions
        else:
            # Top 4 + 1 rule
            top_4 = grouped_conditions[:4]
            others = grouped_conditions[4:]
            
            # Calculate aggregate price and volume for "Other"
            other_yes_price = 0.0
            other_no_price = 0.0
            other_yes_volume = 0.0
            other_no_volume = 0.0
            has_pairs = False
            
            for cond in others:
                if 'yes' in cond and 'no' in cond:
                    # NegRisk pair
                    has_pairs = True
                    other_yes_price += cond['yes']['price']
                    other_no_price += cond['no']['price']
                    other_yes_volume += cond['yes']['volume']
                    other_no_volume += cond['no']['volume']
                else:
                    # Single condition (binary market - shouldn't happen in NegRisk, but handle it)
                    other_yes_price += cond.get('price', 0)
                    other_yes_volume += cond.get('volume', 0)
            
            # Create "Other" condition
            # If we have pairs (NegRisk market), create grouped structure with both yes and no
            if has_pairs:
                other_condition = {
                    'outcome_label': 'Other (Combined)',
                    'yes': {
                        'condition_id': f"{market_id}_OTHER_YES",
                        'price': other_yes_price,
                        'volume': other_yes_volume
                    },
                    'no': {
                        'condition_id': f"{market_id}_OTHER_NO",
                        'price': other_no_price,
                        'volume': other_no_volume
                    }
                }
            else:
                # Only single conditions (should be rare, but handle it)
                other_condition = {
                    'outcome_label': 'Other (Combined)',
                    'condition_id': f"{market_id}_OTHER",
                    'price': other_yes_price,
                    'volume': other_yes_volume
                }
            
            conditions = top_4 + [other_condition]
        
        # Build market dict with correct field order: market_id, question, date, conditions
        market_dict = {
            'market_id': market_id,
            'question': market_question
        }
        
        # Add date field immediately after question if available
        if market_date:
            market_dict['date'] = market_date
        
        # Add conditions
        market_dict['conditions'] = conditions
        
        # Only include bucket_id if it's not "Unknown_*" (add at end if needed)
        if market_bucket_id and not market_bucket_id.startswith('Unknown_'):
            market_dict['bucket_id'] = market_bucket_id
        
        reduced_markets.append(market_dict)
    
    print(f"Top 4 + 1 Rule: {len(df)} raw conditions -> {sum(len(m['conditions']) for m in reduced_markets)} grouped conditions")
    print(f"  Reduced {df['market_id'].nunique()} markets")
    
    return reduced_markets


def reduce_data(raw_json_path: str, output_json_path: str, enrich_from_api: bool = False) -> None:
    """
    Main data reduction pipeline.
    
    Args:
        raw_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
        enrich_from_api: Whether to fetch missing fields from API
    """
    print("=" * 80)
    print("Data Reduction Pipeline")
    print("=" * 80)
    
    # Step 1: Load raw data
    print("\n[Step 1] Loading raw data...")
    with open(raw_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"  Loaded {len(raw_data)} records")
    
    # Step 2: Convert to DataFrame
    print("\n[Step 2] Converting to DataFrame...")
    df = pd.DataFrame(raw_data)
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Step 3: Enrich data if needed
    if enrich_from_api:
        print("\n[Step 3] Enriching data from API...")
        df = enrich_market_data(df)
    else:
        print("\n[Step 3] Using existing data fields...")
        # Check if required fields exist
        required_fields = ['end_date_iso', 'topic', 'question']
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            print(f"  WARNING: Missing fields: {missing_fields}")
            print("  Set enrich_from_api=True to fetch from API")
            # Create placeholder fields
            if 'end_date_iso' not in df.columns:
                df['end_date_iso'] = None
            if 'topic' not in df.columns:
                df['topic'] = 'Unknown'
            if 'question' not in df.columns:
                df['question'] = ''
    
    # Step 4: 7-Day Window Filter
    print("\n[Step 4] Applying 7-Day Window Filter...")
    if 'end_date_iso' in df.columns and df['end_date_iso'].notna().any():
        df = filter_7_day_window(df)
    else:
        print("  Skipping: No end_date_iso data available")
    
    # Step 5: Date Normalization
    print("\n[Step 5] Normalizing Dates...")
    if 'end_date_iso' in df.columns:
        df = normalize_dates(df)
    else:
        print("  Skipping: No end_date_iso data available")
        df['normalized_end_date'] = None
    
    # Step 6: Semantic Bucketing
    print("\n[Step 6] Creating Semantic Buckets...")
    df = create_semantic_buckets(df)
    
    # Step 7: Top 4 + 1 Rule
    print("\n[Step 7] Applying Top 4 + 1 Rule...")
    reduced_markets = apply_top4_plus1_rule(df)
    
    # Step 8: Save output
    print("\n[Step 8] Saving reduced data...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(reduced_markets, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved {len(reduced_markets)} markets to {output_json_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Reduction Summary")
    print("=" * 80)
    print(f"Original records: {len(raw_data)}")
    print(f"Final markets: {len(reduced_markets)}")
    print(f"Total conditions: {sum(len(m['conditions']) for m in reduced_markets)}")
    # Count unique buckets (only count markets that have bucket_id)
    unique_buckets = len(set(m.get('bucket_id') for m in reduced_markets if 'bucket_id' in m and m.get('bucket_id')))
    print(f"Unique buckets: {unique_buckets}")
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reduce Polymarket data for combinatorial arbitrage')
    parser.add_argument('--input', '-i', default='markets_raw.json',
                       help='Input JSON file path (default: markets_raw.json)')
    parser.add_argument('--output', '-o', default='markets_reduced_for_llm.json',
                       help='Output JSON file path (default: markets_reduced_for_llm.json)')
    parser.add_argument('--enrich', '-e', action='store_true',
                       help='Enrich data from API if fields are missing')
    
    args = parser.parse_args()
    
    reduce_data(args.input, args.output, enrich_from_api=args.enrich)


if __name__ == "__main__":
    main()

