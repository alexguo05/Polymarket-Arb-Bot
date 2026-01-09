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
        df['normalized_end_date'] = df['end_date_iso']
    
    # Create bucket_id
    df['bucket_id'] = df['topic'].astype(str) + '_' + df['normalized_end_date'].astype(str)
    
    print(f"Semantic Bucketing: Created {df['bucket_id'].nunique()} unique buckets")
    
    return df


def apply_top4_plus1_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    For markets with >5 conditions:
    - Keep top 4 by volume
    - Aggregate remaining into "Other" condition
    """
    reduced_markets = []
    
    for market_id, group in df.groupby('market_id'):
        # Sort by volume descending
        sorted_group = group.sort_values('volume', ascending=False).reset_index(drop=True)
        
        if len(sorted_group) <= 5:
            # Keep as is (5 or fewer conditions)
            conditions = sorted_group[['condition_id', 'outcome_label', 'price', 'volume']].to_dict('records')
        else:
            # Top 4 + 1 rule
            top_4 = sorted_group.iloc[:4]
            others = sorted_group.iloc[4:]
            
            # Get top 4 conditions
            conditions = top_4[['condition_id', 'outcome_label', 'price', 'volume']].to_dict('records')
            
            # Create "Other" condition
            other_condition = {
                'condition_id': f"{market_id}_OTHER",
                'outcome_label': 'Other (Combined)',
                'price': others['price'].sum(),
                'volume': others['volume'].sum()
            }
            conditions.append(other_condition)
        
        # Get market metadata (should be same for all rows in group)
        market_question = sorted_group['question'].iloc[0] if 'question' in sorted_group.columns else ''
        market_bucket_id = sorted_group['bucket_id'].iloc[0] if 'bucket_id' in sorted_group.columns else ''
        
        reduced_markets.append({
            'market_id': market_id,
            'question': market_question,
            'bucket_id': market_bucket_id,
            'conditions': conditions
        })
    
    print(f"Top 4 + 1 Rule: {len(df)} conditions -> {sum(len(m['conditions']) for m in reduced_markets)} conditions")
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
    with open(raw_json_path, 'r') as f:
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
    with open(output_json_path, 'w') as f:
        json.dump(reduced_markets, f, indent=2)
    
    print(f"  Saved {len(reduced_markets)} markets to {output_json_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Reduction Summary")
    print("=" * 80)
    print(f"Original records: {len(raw_data)}")
    print(f"Final markets: {len(reduced_markets)}")
    print(f"Total conditions: {sum(len(m['conditions']) for m in reduced_markets)}")
    print(f"Unique buckets: {len(set(m['bucket_id'] for m in reduced_markets))}")
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

