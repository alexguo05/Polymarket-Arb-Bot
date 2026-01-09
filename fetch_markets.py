#!/usr/bin/env python3
"""
Polymarket Market Fetcher

Fetches all markets from Polymarket's Gamma API with proper grouping for multi-outcome events.
Uses the /events endpoint to get properly grouped NegRisk markets.
"""

import requests
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import time


# Gamma API endpoint
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DEFAULT_LIMIT = 500  # API max per request


@dataclass
class PolymarketOutcome:
    """Represents a single outcome/condition within a market."""
    condition_id: str
    question: str
    slug: str
    # Token IDs for trading
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None
    # Prices (from mid-market, use CLOB for actual orderbook)
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    # Status
    active: bool = False
    closed: bool = False
    accepting_orders: bool = False
    # Volume & Liquidity
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    # Dates
    end_date: Optional[str] = None
    closed_time: Optional[str] = None
    end_date_iso: Optional[str] = None
    game_start_time: Optional[str] = None
    # Review status
    has_reviewed_dates: Optional[bool] = None
    # Grouping info (for multi-outcome markets)
    group_item_title: Optional[str] = None  # e.g., "Delcy Rodríguez"
    group_item_slug: Optional[str] = None
    # Resolution
    resolution_source: Optional[str] = None
    resolved: bool = False
    
    @property
    def is_binary(self) -> bool:
        """Check if this has YES/NO tokens."""
        return self.yes_token_id is not None and self.no_token_id is not None


@dataclass
class PolymarketEvent:
    """
    Represents a Polymarket event which can contain one or more outcomes.
    
    For single-condition markets: 1 event = 1 outcome (simple YES/NO)
    For multi-outcome (NegRisk) markets: 1 event = N outcomes (e.g., "Who will win?")
    """
    # Event identifiers
    event_id: str
    title: str
    slug: str
    description: Optional[str] = None
    
    # Type flags
    enable_neg_risk: bool = False  # True = multi-outcome market
    neg_risk_augmented: bool = False
    
    # Status
    active: bool = False
    closed: bool = False
    archived: bool = False
    
    # Category & Tags
    category: Optional[str] = None
    tags: list[dict] = field(default_factory=list)
    series_slug: Optional[str] = None
    
    # Volume & Liquidity (aggregate)
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    liquidity_clob: Optional[float] = None
    
    # Dates
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    closed_time: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    # Resolution
    resolution_source: Optional[str] = None
    
    # Display
    image: Optional[str] = None
    icon: Optional[str] = None
    
    # The outcomes/conditions within this event
    outcomes: list[PolymarketOutcome] = field(default_factory=list)
    
    @property
    def is_multi_outcome(self) -> bool:
        """True if this is a NegRisk multi-outcome market."""
        return self.enable_neg_risk or len(self.outcomes) > 1
    
    @property
    def outcome_count(self) -> int:
        """Number of outcomes in this event."""
        return len(self.outcomes)
    
    def sum_yes_prices(self) -> Optional[float]:
        """Sum of all YES prices (for arb detection in multi-outcome markets)."""
        prices = [o.yes_price for o in self.outcomes if o.yes_price is not None]
        return sum(prices) if prices else None


def fetch_all_events(
    limit: int = DEFAULT_LIMIT,
    active_only: bool = True,
    closed: bool = False,
    max_pages: Optional[int] = None,
) -> list[PolymarketEvent]:
    """
    Fetch all events from Polymarket's Gamma API.
    
    Events contain grouped markets - this is the proper way to get multi-outcome markets.
    
    Args:
        limit: Number of events per page (max 100)
        active_only: Only fetch active events
        closed: Include closed events
        max_pages: Maximum number of pages to fetch (None = all)
        
    Returns:
        List of PolymarketEvent objects with their outcomes
    """
    events = []
    offset = 0
    page = 0
    
    print(f"Fetching events from Polymarket Gamma API...")
    print(f"  Active only: {active_only}, Include closed: {closed}")
    
    while True:
        params = {
            "limit": limit,
            "offset": offset,
        }
        
        if active_only:
            params["active"] = "true"
        if not closed:
            params["closed"] = "false"
        
        url = f"{GAMMA_API_BASE}/events"
        
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
            print(f"  No more events found.")
            break
            
        page_events = parse_events(data)
        events.extend(page_events)
        
        outcomes_count = sum(e.outcome_count for e in page_events)
        print(f"  Page {page + 1}: fetched {len(page_events)} events ({outcomes_count} outcomes) | Total: {len(events)} events")
        
        if len(data) < limit:
            break
            
        page += 1
        if max_pages and page >= max_pages:
            print(f"  Reached max pages limit ({max_pages})")
            break
            
        offset += limit
        time.sleep(0.05)  # Small delay to be nice to the API
    
    return events


def parse_events(data: list[dict]) -> list[PolymarketEvent]:
    """Parse raw API response into PolymarketEvent objects."""
    events = []
    
    for item in data:
        try:
            # Parse outcomes (markets within the event)
            outcomes = []
            for market in item.get("markets", []):
                outcome = parse_outcome(market)
                if outcome:
                    outcomes.append(outcome)
            
            event = PolymarketEvent(
                # Identifiers
                event_id=str(item.get("id", "")),
                title=item.get("title", ""),
                slug=item.get("slug", ""),
                description=item.get("description", ""),
                
                # Type flags
                enable_neg_risk=item.get("enableNegRisk", False),
                neg_risk_augmented=item.get("negRiskAugmented", False),
                
                # Status
                active=item.get("active", False),
                closed=item.get("closed", False),
                archived=item.get("archived", False),
                
                # Category & Tags
                category=item.get("category"),
                tags=item.get("tags", []),
                series_slug=item.get("seriesSlug"),
                
                # Volume & Liquidity
                volume=safe_float(item.get("volume")),
                liquidity=safe_float(item.get("liquidity")),
                liquidity_clob=safe_float(item.get("liquidityClob")),
                
                # Dates
                start_date=item.get("startDate"),
                end_date=item.get("endDate"),
                closed_time=item.get("closedTime"),
                created_at=item.get("createdAt"),
                updated_at=item.get("updatedAt"),
                
                # Resolution
                resolution_source=item.get("resolutionSource"),
                
                # Display
                image=item.get("image"),
                icon=item.get("icon"),
                
                # Outcomes
                outcomes=outcomes,
            )
            events.append(event)
            
        except Exception as e:
            print(f"  Warning: Failed to parse event: {e}")
            continue
    
    return events


def parse_outcome(market: dict) -> Optional[PolymarketOutcome]:
    """Parse a market dict into a PolymarketOutcome."""
    try:
        # Parse token IDs from JSON string
        token_ids = []
        if market.get("clobTokenIds"):
            try:
                token_ids = json.loads(market["clobTokenIds"])
            except json.JSONDecodeError:
                pass
        
        # Parse outcome prices from JSON string
        prices = []
        if market.get("outcomePrices"):
            try:
                prices = [float(p) for p in json.loads(market["outcomePrices"])]
            except (json.JSONDecodeError, ValueError):
                pass
        
        return PolymarketOutcome(
            condition_id=market.get("conditionId", ""),
            question=market.get("question", ""),
            slug=market.get("slug", ""),
            # Token IDs
            yes_token_id=token_ids[0] if len(token_ids) >= 1 else None,
            no_token_id=token_ids[1] if len(token_ids) >= 2 else None,
            # Prices
            yes_price=prices[0] if len(prices) >= 1 else None,
            no_price=prices[1] if len(prices) >= 2 else None,
            # Status
            active=market.get("active", False),
            closed=market.get("closed", False),
            accepting_orders=market.get("acceptingOrders", False),
            # Volume & Liquidity
            volume=safe_float(market.get("volume")),
            liquidity=safe_float(market.get("liquidity")),
            # Dates
            end_date=market.get("endDate"),
            closed_time=market.get("closedTime"),
            end_date_iso=market.get("endDateIso"),
            game_start_time=market.get("gameStartTime"),
            # Review status
            has_reviewed_dates=market.get("hasReviewedDates"),
            # Grouping info
            group_item_title=market.get("groupItemTitle"),
            group_item_slug=market.get("groupItemSlug"),
            # Resolution
            resolution_source=market.get("resolutionSource"),
            resolved=market.get("resolved", False),
        )
    except Exception as e:
        print(f"  Warning: Failed to parse outcome: {e}")
        return None


def safe_float(value) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def print_summary(events: list[PolymarketEvent]):
    """Print a summary of fetched events."""
    print("\n" + "=" * 80)
    print("POLYMARKET EVENTS SUMMARY")
    print("=" * 80)
    
    total_events = len(events)
    total_outcomes = sum(e.outcome_count for e in events)
    
    single_outcome = sum(1 for e in events if e.outcome_count == 1)
    multi_outcome = sum(1 for e in events if e.outcome_count > 1)
    neg_risk_events = sum(1 for e in events if e.enable_neg_risk)
    
    active_events = sum(1 for e in events if e.active and not e.closed)
    
    print(f"\nTotal events: {total_events}")
    print(f"Total outcomes/conditions: {total_outcomes}")
    print(f"\n  Single-outcome events: {single_outcome}")
    print(f"  Multi-outcome events: {multi_outcome}")
    print(f"  NegRisk (true multi-outcome): {neg_risk_events}")
    print(f"  Active events: {active_events}")
    
    # Show top multi-outcome events by outcome count
    multi_events = [e for e in events if e.outcome_count > 1]
    multi_events.sort(key=lambda e: e.outcome_count, reverse=True)
    
    print(f"\n--- Top 10 Multi-Outcome Events ---")
    for i, e in enumerate(multi_events[:10], 1):
        sum_yes = e.sum_yes_prices()
        sum_str = f" | Sum YES: {sum_yes:.1%}" if sum_yes else ""
        neg_risk_str = " [NegRisk]" if e.enable_neg_risk else ""
        print(f"{i:2}. {e.title[:50]:<50} | {e.outcome_count} outcomes{neg_risk_str}{sum_str}")
    
    # Show top events by liquidity
    liquid_events = [e for e in events if e.liquidity and e.liquidity > 1000]
    liquid_events.sort(key=lambda e: e.liquidity or 0, reverse=True)
    
    print(f"\n--- Top 10 Events by Liquidity ---")
    for i, e in enumerate(liquid_events[:10], 1):
        print(f"{i:2}. [{e.slug[:40]:<40}] Liq: ${e.liquidity:,.0f} | {e.outcome_count} outcomes")


def find_multi_outcome_arbs(events: list[PolymarketEvent], threshold: float = 0.02) -> list[tuple[PolymarketEvent, float, str]]:
    """
    Find arbitrage opportunities in multi-outcome (NegRisk) markets.
    
    For NegRisk markets, exactly one outcome wins, so:
    - Sum of YES prices should = 100%
    - If < 100%: Long arb (buy all YES positions)
    - If > 100%: Short arb (buy all NO positions)
    
    Args:
        events: List of events to analyze
        threshold: Minimum deviation from 100% to report
        
    Returns:
        List of (event, deviation, arb_type) tuples
    """
    opportunities = []
    
    for event in events:
        # Only analyze NegRisk events (true multi-outcome)
        if not event.enable_neg_risk:
            continue
        
        if not event.active or event.closed:
            continue
        
        sum_yes = event.sum_yes_prices()
        if sum_yes is None:
            continue
        
        deviation = abs(1.0 - sum_yes)
        
        if deviation >= threshold:
            arb_type = "LONG" if sum_yes < 1.0 else "SHORT"
            opportunities.append((event, deviation, arb_type))
    
    # Sort by deviation (highest first)
    opportunities.sort(key=lambda x: x[1], reverse=True)
    return opportunities


def print_arb_opportunities(opportunities: list[tuple[PolymarketEvent, float, str]]):
    """Print multi-outcome arbitrage opportunities."""
    print("\n" + "=" * 80)
    print("MULTI-OUTCOME ARBITRAGE OPPORTUNITIES (NegRisk markets)")
    print("=" * 80)
    
    if not opportunities:
        print("\nNo significant arbitrage opportunities found (threshold: 2%).")
        return
    
    print(f"\nFound {len(opportunities)} potential opportunities:\n")
    
    for i, (event, deviation, arb_type) in enumerate(opportunities[:20], 1):
        sum_yes = event.sum_yes_prices()
        print(f"{i:2}. {event.title[:65]}")
        print(f"    Event ID: {event.event_id} | Outcomes: {event.outcome_count}")
        print(f"    Sum of YES: {sum_yes:.2%} | Deviation: {deviation:.2%} | Type: {arb_type}")
        print(f"    Liquidity: ${event.liquidity:,.0f}" if event.liquidity else "    Liquidity: N/A")
        
        # Show individual outcomes
        for outcome in event.outcomes[:8]:
            title = outcome.group_item_title or outcome.question[:30]
            price = f"{outcome.yes_price:.1%}" if outcome.yes_price else "N/A"
            print(f"      - {price:>6} : {title[:40]}")
        
        if len(event.outcomes) > 8:
            print(f"      ... and {len(event.outcomes) - 8} more outcomes")
        print()


def save_events_json(events: list[PolymarketEvent], filename: str = "polymarket_events.json"):
    """Save events to a JSON file."""
    data = []
    for e in events:
        event_dict = {
            "event_id": e.event_id,
            "title": e.title,
            "slug": e.slug,
            "description": e.description,
            "enable_neg_risk": e.enable_neg_risk,
            "neg_risk_augmented": e.neg_risk_augmented,
            "active": e.active,
            "closed": e.closed,
            "archived": e.archived,
            "category": e.category,
            "tags": e.tags,
            "series_slug": e.series_slug,
            "volume": e.volume,
            "liquidity": e.liquidity,
            "liquidity_clob": e.liquidity_clob,
            "start_date": e.start_date,
            "end_date": e.end_date,
            "closed_time": e.closed_time,
            "created_at": e.created_at,
            "updated_at": e.updated_at,
            "resolution_source": e.resolution_source,
            "image": e.image,
            "icon": e.icon,
            "outcome_count": e.outcome_count,
            "sum_yes_prices": e.sum_yes_prices(),
            "outcomes": [
                {
                    "condition_id": o.condition_id,
                    "question": o.question,
                    "slug": o.slug,
                    "yes_token_id": o.yes_token_id,
                    "no_token_id": o.no_token_id,
                    "yes_price": o.yes_price,
                    "no_price": o.no_price,
                    "active": o.active,
                    "closed": o.closed,
                    "accepting_orders": o.accepting_orders,
                    "volume": o.volume,
                    "liquidity": o.liquidity,
                    "end_date": o.end_date,
                    "closed_time": o.closed_time,
                    "end_date_iso": o.end_date_iso,
                    "game_start_time": o.game_start_time,
                    "has_reviewed_dates": o.has_reviewed_dates,
                    "group_item_title": o.group_item_title,
                    "group_item_slug": o.group_item_slug,
                    "resolution_source": o.resolution_source,
                    "resolved": o.resolved,
                }
                for o in e.outcomes
            ],
        }
        data.append(event_dict)
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved {len(events)} events to {filename}")


def main():
    """Main entry point."""
    print("Polymarket Market Fetcher")
    print("-" * 50)
    
    # Fetch all active events (includes grouped multi-outcome markets)
    events = fetch_all_events(
        limit=DEFAULT_LIMIT,
        active_only=True,
        closed=False,
        max_pages=None,  # Fetch all pages
    )
    
    # Print summary
    print_summary(events)
    
    # Find multi-outcome arb opportunities
    opportunities = find_multi_outcome_arbs(events, threshold=0.02)
    print_arb_opportunities(opportunities)
    
    # Save to JSON
    save_events_json(events)


if __name__ == "__main__":
    main()