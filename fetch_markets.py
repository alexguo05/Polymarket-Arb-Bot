#!/usr/bin/env python3
"""
Polymarket Market Fetcher

Fetches all markets from Polymarket's Gamma API for arbitrage analysis.
The Gamma API provides market metadata including:
- Market slugs and questions
- CLOB token IDs (YES/NO tokens)
- Current prices and activity status
"""

import requests
import json
from dataclasses import dataclass
from typing import Optional
import time


# Gamma API endpoint
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


@dataclass
class PolymarketMarket:
    """Represents a single Polymarket market."""
    condition_id: str
    question: str
    slug: str
    yes_token_id: Optional[str]
    no_token_id: Optional[str]
    yes_price: Optional[float]
    no_price: Optional[float]
    active: bool
    closed: bool
    volume: Optional[float]
    liquidity: Optional[float]
    
    @property
    def is_binary(self) -> bool:
        """Check if this is a simple YES/NO market."""
        return self.yes_token_id is not None and self.no_token_id is not None


def fetch_all_markets(
    limit: int = 100,
    active_only: bool = True,
    closed: bool = False,
    max_pages: Optional[int] = None,
) -> list[PolymarketMarket]:
    """
    Fetch all markets from Polymarket's Gamma API.
    
    Args:
        limit: Number of markets per page (max 100)
        active_only: Only fetch active markets
        closed: Include closed markets
        max_pages: Maximum number of pages to fetch (None = all)
        
    Returns:
        List of PolymarketMarket objects
    """
    markets = []
    offset = 0
    page = 0
    
    print(f"Fetching markets from Polymarket Gamma API...")
    print(f"  Active only: {active_only}, Include closed: {closed}")
    
    while True:
        # Build query parameters
        params = {
            "limit": limit,
            "offset": offset,
        }
        
        if active_only:
            params["active"] = "true"
        if not closed:
            params["closed"] = "false"
        
        url = f"{GAMMA_API_BASE}/markets"
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Error fetching page {page}: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"  Error parsing response: {e}")
            break
        
        if not data:
            print(f"  No more markets found.")
            break
            
        page_markets = parse_markets(data)
        markets.extend(page_markets)
        
        print(f"  Page {page + 1}: fetched {len(page_markets)} markets (total: {len(markets)})")
        
        # Check if we got fewer markets than requested (last page)
        if len(data) < limit:
            break
            
        # Check page limit
        page += 1
        if max_pages and page >= max_pages:
            print(f"  Reached max pages limit ({max_pages})")
            break
            
        offset += limit
        
        # Rate limiting - be nice to the API
        # time.sleep(0.1)
    
    return markets


def parse_markets(data: list[dict]) -> list[PolymarketMarket]:
    """Parse raw API response into PolymarketMarket objects."""
    markets = []
    
    for item in data:
        try:
            # Parse token IDs from JSON string
            token_ids = []
            if item.get("clobTokenIds"):
                try:
                    token_ids = json.loads(item["clobTokenIds"])
                except json.JSONDecodeError:
                    pass
            
            # Parse outcome prices from JSON string
            prices = []
            if item.get("outcomePrices"):
                try:
                    prices = [float(p) for p in json.loads(item["outcomePrices"])]
                except (json.JSONDecodeError, ValueError):
                    pass
            
            market = PolymarketMarket(
                condition_id=item.get("conditionId", ""),
                question=item.get("question", ""),
                slug=item.get("slug", ""),
                yes_token_id=token_ids[0] if len(token_ids) >= 1 else None,
                no_token_id=token_ids[1] if len(token_ids) >= 2 else None,
                yes_price=prices[0] if len(prices) >= 1 else None,
                no_price=prices[1] if len(prices) >= 2 else None,
                active=item.get("active", False),
                closed=item.get("closed", False),
                volume=float(item.get("volume", 0) or 0),
                liquidity=float(item.get("liquidity", 0) or 0),
            )
            markets.append(market)
        except Exception as e:
            print(f"  Warning: Failed to parse market: {e}")
            continue
    
    return markets


def print_market_summary(markets: list[PolymarketMarket]):
    """Print a summary of fetched markets."""
    print("\n" + "=" * 80)
    print("POLYMARKET MARKETS SUMMARY")
    print("=" * 80)
    
    total = len(markets)
    binary = sum(1 for m in markets if m.is_binary)
    active = sum(1 for m in markets if m.active and not m.closed)
    with_liquidity = sum(1 for m in markets if m.liquidity and m.liquidity > 0)
    
    print(f"\nTotal markets fetched: {total}")
    print(f"  Binary (YES/NO) markets: {binary}")
    print(f"  Active markets: {active}")
    print(f"  Markets with liquidity: {with_liquidity}")
    
    # Sort by liquidity and show top markets
    liquid_markets = [m for m in markets if m.liquidity and m.liquidity > 1000]
    liquid_markets.sort(key=lambda m: m.liquidity or 0, reverse=True)
    
    print(f"\n--- Top 20 Markets by Liquidity ---")
    for i, m in enumerate(liquid_markets[:20], 1):
        spread = None
        if m.yes_price and m.no_price:
            total_cost = m.yes_price + m.no_price
            spread = abs(1.0 - total_cost) * 100  # Spread in percentage
        
        spread_str = f" (spread: {spread:.2f}%)" if spread else ""
        print(f"{i:2}. [{m.slug[:40]:<40}] Liq: ${m.liquidity:,.0f}{spread_str}")
        if m.yes_price and m.no_price:
            print(f"     YES: {m.yes_price:.4f}, NO: {m.no_price:.4f}")


def find_arb_opportunities(markets: list[PolymarketMarket], threshold: float = 0.99) -> list[PolymarketMarket]:
    """
    Find potential arbitrage opportunities where YES + NO < threshold.
    
    In a perfect market, YES + NO = 1.00. If YES + NO < 1.00, there's potential
    for risk-free profit by buying both sides.
    
    Args:
        markets: List of markets to analyze
        threshold: Maximum total cost to consider as arb opportunity
        
    Returns:
        List of markets with potential arb opportunities
    """
    opportunities = []
    
    for market in markets:
        if not market.is_binary or not market.active or market.closed:
            continue
            
        if market.yes_price is None or market.no_price is None:
            continue
            
        total_cost = market.yes_price + market.no_price
        
        if total_cost < threshold:
            opportunities.append(market)
    
    # Sort by profit potential
    opportunities.sort(key=lambda m: (m.yes_price or 0) + (m.no_price or 0))
    
    return opportunities


def print_arb_opportunities(opportunities: list[PolymarketMarket]):
    """Print potential arbitrage opportunities."""
    print("\n" + "=" * 80)
    print("POTENTIAL ARBITRAGE OPPORTUNITIES (YES + NO < 99%)")
    print("=" * 80)
    
    if not opportunities:
        print("\nNo arbitrage opportunities found.")
        return
    
    print(f"\nFound {len(opportunities)} potential opportunities:\n")
    
    for i, m in enumerate(opportunities[:50], 1):
        total_cost = (m.yes_price or 0) + (m.no_price or 0)
        profit_pct = (1.0 - total_cost) * 100
        
        print(f"{i:2}. {m.question[:70]}")
        print(f"    Slug: {m.slug}")
        print(f"    YES: {m.yes_price:.4f}  |  NO: {m.no_price:.4f}  |  Total: {total_cost:.4f}")
        print(f"    Potential profit: {profit_pct:.2f}%")
        print(f"    YES Token: {m.yes_token_id}")
        print(f"    NO Token:  {m.no_token_id}")
        print(f"    Liquidity: ${m.liquidity:,.0f}")
        print()


def save_markets_json(markets: list[PolymarketMarket], filename: str = "polymarket_markets.json"):
    """Save markets to a JSON file."""
    data = []
    for m in markets:
        data.append({
            "condition_id": m.condition_id,
            "question": m.question,
            "slug": m.slug,
            "yes_token_id": m.yes_token_id,
            "no_token_id": m.no_token_id,
            "yes_price": m.yes_price,
            "no_price": m.no_price,
            "active": m.active,
            "closed": m.closed,
            "volume": m.volume,
            "liquidity": m.liquidity,
        })
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {len(markets)} markets to {filename}")


def main():
    """Main entry point."""
    print("Polymarket Forest Arb Bot - Market Fetcher")
    print("-" * 50)
    
    # Fetch all active markets
    markets = fetch_all_markets(
        limit=100,
        active_only=True,
        closed=False,
        max_pages=None,  # Fetch all pages
    )
    
    # Print summary
    print_market_summary(markets)
    
    # Find and print arb opportunities
    opportunities = find_arb_opportunities(markets, threshold=0.995)
    print_arb_opportunities(opportunities)
    
    # Save to JSON for further analysis
    save_markets_json(markets)
    
    # Also save opportunities separately
    if opportunities:
        save_markets_json(opportunities, "arb_opportunities.json")


if __name__ == "__main__":
    main()

