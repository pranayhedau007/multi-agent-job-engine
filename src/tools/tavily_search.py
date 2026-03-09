"""
Author: Pranay Hedau
Purpose: Tavily web search tool for job discovery.

Two-step pipeline:
    1. search_jobs() — raw Tavily API search, returns JSON
    2. parse_search_to_listings() — LLM parses raw results into JobListing objects


The @tool decorator:
    LangChain's @tool turns a regular function into something agents
    can "call" during execution. It auto-generates the tool's name,
    description, and argument schema from the function signature and
    docstring. When an agent decides "I need to search for jobs",
    LangGraph invokes this tool automatically.

Usage:
    from src.tools.tavily_search import search_jobs, parse_search_to_listings

    raw = search_jobs.invoke({"query": "ML intern California 2026"})
    listings = parse_search_to_listings(raw, "ML intern California 2026")
"""

import json
import logging

from tavily import TavilyClient
from langchain_core.tools import tool

from src.config import settings
from src.models.schemas import JobListing
from src.models.llm import get_llm

logger = logging.getLogger(__name__)

# Module-level client — created once, reused across calls
# Why did not create inside the function? TavilyClient does internal
# setup (session pooling, auth) that's wasteful to repeat per call
_client = None

"""Purpose: Lazy-initialize the Tavily client on first use."""
def _get_client() -> TavilyClient:
    
    global _client
    if _client is None:
        if not settings.tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY not set. Get a free key at https://tavily.com"
            )
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


"""Purpose: Search for job listings using Tavily web search.
    This is a LangChain tool — the @tool decorator means LangGraph
    can invoke it automatically when an agent needs to search.

    Why include_domains?
        Without domain filtering, Tavily might return Reddit threads,
        Medium articles, or random blogs. By restricting to known job
        boards, we get higher-quality results that are more likely to
        be actual job postings.
    """
@tool
def search_jobs(query: str, max_results: int = 10) -> str:
    """Search for job listings using Tavily web search.

    Args:
        query: Search string like 'AI ML intern summer 2026 California'
        max_results: How many results to return (default 10)

    Returns:
        JSON string of raw search results.
    """
    client = _get_client()

    results = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",  # "advanced" does deeper crawling, better for job pages
        include_domains=[
            "linkedin.com",
            "greenhouse.io",
            "lever.co",
            "builtin.com",
            "indeed.com",
            "ziprecruiter.com",
            "careers.google.com",
            "jobs.ashbyhq.com",
        ],
    )

    raw = results.get("results", [])
    logger.info(f"Tavily search '{query}' → {len(raw)} results")
    return json.dumps(raw, indent=2)


"""Purpose: Use LLM to parse raw Tavily results into structured JobListing objects.

    Args:
        raw_results: JSON string from search_jobs()
        user_query: Original search query (helps LLM understand context)

    Returns:
        List of validated JobListing objects, empty list if parsing fails

    Why not regex or rule-based parsing?
        Job board HTML structures vary wildly. LinkedIn formats differently
        from Greenhouse, which formats differently from Indeed. An LLM
        handles this variation naturally — it "reads" the content the way
        a human would, regardless of source format.
    """
def parse_search_to_listings(raw_results: str, user_query: str) -> list[JobListing]:
    
    llm = get_llm(temperature=0.1)  # Low temp = precise, consistent extraction

#Setting context for LLM using a system prompt for a guided outputs
    prompt = f"""You are a job listing parser. Given raw web search results for the query "{user_query}",
extract ONLY actual job postings (not blog posts, news articles, career guides, or listicles).

For each real job listing found, extract:
- title: exact job title as posted
- company: company name
- location: job location (city, state or "Remote")
- url: direct application URL
- summary: 2-3 sentence summary of the role and requirements
- key_skills: list of technical skills mentioned in the posting
- posted_date: when posted (if visible, otherwise "Unknown")
- visa_friendly: true UNLESS the posting explicitly says "no sponsorship" or "must be authorized to work without sponsorship"

Return a JSON array of objects. If no real job listings are found, return [].
Return ONLY valid JSON — no markdown fences, no explanation, no preamble.

Raw search results:
{raw_results}"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    # LLMs sometimes wrap JSON in ```json ... ``` markdown fences
    # so we gonna Strip them if present
    if content.startswith("```"):
        # Remove first line (```json) and last line (```)
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        parsed = json.loads(content)
        # Validate each item through the Pydantic model
        listings = []
        for item in parsed:
            if isinstance(item, dict):
                try:
                    listings.append(JobListing(**item))
                except Exception as e:
                    logger.warning(f"Skipped invalid listing: {e}")
        logger.info(f"Parsed {len(listings)} valid job listings from {len(parsed)} raw items")
        return listings
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Raw LLM response: {content[:500]}")
        return []