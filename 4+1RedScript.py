import pandas as pd

def reduce_market_outcomes(df):
    """
    Reduces a market's conditions to the Top 4 by volume + 'Other'.
    
    Expected df columns: 
    ['market_id', 'condition_id', 'outcome_label', 'volume', 'price']
    """
    
    # Input validation
    if df.empty:
        return pd.DataFrame(columns=['market_id', 'condition_id', 'outcome_label', 'volume', 'price'])
    
    required_columns = ['market_id', 'condition_id', 'outcome_label', 'volume', 'price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    reduced_rows = []
    
    # Process each market individually
    for market_id, group in df.groupby('market_id'):
        
        # If 4 or fewer conditions, keep as is (no need for "Other")
        if len(group) <= 4:
            reduced_rows.extend(group.to_dict('records'))
            continue
            
        # Sort by volume (liquidity) descending
        sorted_group = group.sort_values(by='volume', ascending=False)
        
        # Keep the Top 4
        top_4 = sorted_group.iloc[:4].copy()
        
        # Aggregate the rest into "Other"
        others = sorted_group.iloc[4:]
        
        # Only create "Other" if there are actually other conditions to aggregate
        if len(others) > 0:
            # Create the "Other" condition
            # Note: The price of "Other" is the sum of the prices of the aggregated outcomes
            # because these are mutually exclusive probabilities.
            other_condition = {
                'market_id': market_id,
                'condition_id': f"{market_id}_OTHER",
                'outcome_label': 'Other (Combined)',
                'volume': others['volume'].sum(),
                'price': others['price'].sum() 
            }
            
            # Add to our list
            reduced_rows.extend(top_4.to_dict('records'))
            reduced_rows.append(other_condition)
        else:
            # Edge case: exactly 4 conditions, just keep them
            reduced_rows.extend(top_4.to_dict('records'))
        
    return pd.DataFrame(reduced_rows)

# Example Usage
# dummy_data = pd.DataFrame(...)
# clean_data = reduce_market_outcomes(dummy_data)
# print(clean_data)