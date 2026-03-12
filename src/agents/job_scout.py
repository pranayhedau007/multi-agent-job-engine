"""
Author: Pranay Hedau
Purpose: Job Scout Agent : discovers relevant job listings via web search using Tavily.

Pipeline:
    1. Build targeted search queries from user preferences
    2. Execute searches via Tavily API
    3. Parse raw results into structured JobListing objects (LLM)
    4. Deduplicate (same company + title = one listing)
    5. Filter by visa requirements
    6. Return top results

This is the FIRST agent in every pipeline run. Every other agent
depends on its output and no job listings means nothing else can run.

Usage:
    from src.agents.job_scout import run_job_scout

    listings = run_job_scout(user_preferences)
"""

import logging

from src.models.schemas import JobListing, UserPreferences
from src.tools.tavily_search import search_jobs, parse_search_to_listings

logger = logging.getLogger(__name__)


def build_search_queries(prefs: UserPreferences) -> list[str]:
    """Purpose: Turn user preferences into effective Tavily search queries.

    Strategy: Generate 2 queries per target role -
        1. Role + location + timeframe (broad discovery)
        2. Role + tech stack keywords (precision targeting)

    Why multiple queries? A single query might miss listings that
    use different terminology. "AI ML intern" finds different results
    than "machine learning engineer intern Python LangChain".

    Capped at 4 queries to manage Tavily API costs (free tier = 1000/month).
    """
    queries = []

    for role in prefs.target_roles:
        # Query 1: Broad - role + location + timeframe
        location_str = " ".join(prefs.locations[:2])
        queries.append(f"{role} intern summer 2026 {location_str}")

        # Query 2: Specific - role + tech stack
        if prefs.tech_stack:
            stack_str = " ".join(prefs.tech_stack[:3])
            queries.append(f"{role} {stack_str} intern 2026")

    return queries[:4]


def deduplicate(listings: list[JobListing]) -> list[JobListing]:
    """Purpose: Remove duplicate listings based on company + title combo.

    Why this happens: Different queries often return the same popular
    listings. "ML intern California" and "ML intern Python LangChain"
    might both return the same NVIDIA posting.

    We normalize to lowercase and strip whitespace before comparing
    so "NVIDIA" and "nvidia " are treated as the same company.
    """
    seen = set()
    unique = []

    for job in listings:
        key = (job.company.lower().strip(), job.title.lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(job)

    if len(listings) != len(unique):
        logger.info(f"Deduplicated: {len(listings)} -> {len(unique)} listings")

    return unique


def filter_by_preferences(
    listings: list[JobListing], prefs: UserPreferences
) -> list[JobListing]:
    """Purpose: Filter listings based on user requirements.

    Currently filters on visa sponsorship only. Could be extended
    to filter by location, salary range, company size, etc.

    Why filter AFTER search instead of in the search query?
        Tavily can't filter by visa status that info is buried
        inside JD text. The LLM extracts it during parsing, and
        we filter here based on the extracted field.
    """
    if not prefs.visa_required:
        return listings

    filtered = []
    for job in listings:
        if job.visa_friendly:
            filtered.append(job)
        else:
            logger.debug(f"Filtered out: {job.title} at {job.company} (no visa)")

    if len(listings) != len(filtered):
        logger.info(f"Visa filter: {len(listings)} -> {len(filtered)} listings")

    return filtered


def run_job_scout(prefs: UserPreferences) -> list[JobListing]:
    """Purpose: Execute the full Job Scout pipeline.

    This is the function that the LangGraph node calls.
    It orchestrates the entire search -> parse -> filter flow.

    Returns up to 15 listings, sorted by search relevance.
    """
    queries = build_search_queries(prefs)
    logger.info(f"Job Scout: running {len(queries)} search queries")

    all_listings: list[JobListing] = []

    for query in queries:
        logger.info(f"  Searching: '{query}'")
        raw = search_jobs.invoke({"query": query, "max_results": 8})
        parsed = parse_search_to_listings(raw, query)
        all_listings.extend(parsed)
        logger.info(f"  -> {len(parsed)} listings parsed")

    # Clean up results
    unique = deduplicate(all_listings)
    filtered = filter_by_preferences(unique, prefs)

    logger.info(f"Job Scout complete: {len(filtered)} listings (from {len(all_listings)} raw)")
    return filtered[:15]