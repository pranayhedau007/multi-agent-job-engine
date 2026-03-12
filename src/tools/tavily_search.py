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
import re

from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

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


def _extract_json(text: str) -> str:
    """Purpose: Robustly extract a JSON array from LLM output.

    Small local LLMs (like llama3.2) often ignore "return ONLY JSON"
    instructions. They wrap JSON in markdown fences, add preamble like
    "Here is the JSON:", or append explanations after the array.

    Strategy (ordered by reliability):
        1. Strip markdown fences (```json ... ``` or ``` ... ```)
        2. Try parsing the stripped text directly
        3. Regex fallback: find the outermost [...] in the text
    """
    content = text.strip()

    # Step 1: Strip markdown fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```json or ```) and last line (```)
        start = 1
        end = len(lines)
        if lines[-1].strip() == "```":
            end = -1
        content = "\n".join(lines[start:end]).strip()
    elif content.endswith("```"):
        content = content[:-3].strip()

    # Step 2: If it looks like valid JSON already, return it
    if content.startswith("["):
        return content

    # Step 3: Regex fallback — find the outermost JSON array in prose
    # This handles: "Here is the JSON:\n\n[{...}, {...}]\n\nThis is..."
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        return match.group(0)

    # Nothing found — return whatever we have so json.loads gives a clear error
    return content


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
    
    # json_mode=True activates Ollama's format='json' which constrains
    # token generation to produce valid JSON — critical for small models
    # like llama3.2 that ignore "return ONLY JSON" text instructions
    llm = get_llm(temperature=0.1, json_mode=True)

    # Using SystemMessage + HumanMessage split helps Ollama models
    # separate instructions from data, improving JSON-only compliance.
    # The few-shot example shows the exact output shape we expect.
    system_msg = SystemMessage(content=(
        "You are a job listing parser that outputs JSON.\n"
        "Given raw web search results, extract ONLY actual job postings "
        "(not blog posts, news articles, career guides, or listicles).\n\n"
        "For each real job listing found, return a JSON object with these fields:\n"
        "  title, company, location, url, summary, key_skills, posted_date, visa_friendly\n\n"
        "Return a JSON object with a \"listings\" key containing an array of job objects.\n"
        "If no real job listings are found, return {\"listings\": []}.\n\n"
        "Example output:\n"
        "{\"listings\": [{\"title\": \"ML Engineer Intern\", \"company\": \"NVIDIA\", "
        "\"location\": \"Santa Clara, CA\", \"url\": \"https://nvidia.com/jobs/123\", "
        "\"summary\": \"Work on deep learning models for autonomous driving.\", "
        "\"key_skills\": [\"Python\", \"PyTorch\", \"CUDA\"], "
        "\"posted_date\": \"Unknown\", \"visa_friendly\": true}]}"
    ))

    # Truncate raw results to avoid overwhelming the small model's context
    truncated = raw_results[:6000] if len(raw_results) > 6000 else raw_results

    human_msg = HumanMessage(content=(
        f"Search query: \"{user_query}\"\n\n"
        f"Raw search results:\n{truncated}"
    ))

    response = llm.invoke([system_msg, human_msg])
    raw_content = response.content.strip() if response.content else ""

    if not raw_content:
        logger.error("LLM returned empty response — no content to parse")
        return []

    # Use robust extraction to handle any remaining formatting issues
    content = _extract_json(raw_content)

    try:
        parsed = json.loads(content)

        # Handle different response shapes from the LLM:
        # 1. {"listings": [...]} — the expected shape from our prompt
        # 2. [...] — a plain array (some models skip the wrapper)
        # 3. {...} — a single job object (Ollama JSON mode quirk)
        if isinstance(parsed, dict):
            if "listings" in parsed:
                items = parsed["listings"]
            else:
                # Single job object — wrap in list
                items = [parsed]
        elif isinstance(parsed, list):
            items = parsed
        else:
            logger.warning(f"Unexpected JSON type: {type(parsed)}")
            return []

        # Validate each item through the Pydantic model
        listings = []
        for item in items:
            if isinstance(item, dict):
                try:
                    listings.append(JobListing(**item))
                except Exception as e:
                    logger.warning(f"Skipped invalid listing: {e}")
        logger.info(f"Parsed {len(listings)} valid job listings from {len(items)} raw items")
        return listings
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.info(f"Raw LLM response (first 500 chars): {raw_content[:500]}")
        return []