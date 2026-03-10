"""
Author: Pranay Hedau
Purpose: Qdrant vector store for resume-JD semantic matching.

Core concept: Convert text into numerical vectors (embeddings), then
measure similarity between vectors using cosine similarity.

Why I took this approach of embeddings?
    Keyword matching: "built microservices" does NOT match "distributed systems experience"
    Vector matching:  "built microservices" DOES match "distributed systems experience"
    because embeddings capture MEANING, not just exact words.

Two embedding providers (as I'm having hybrid LLM setup):
    - OpenAI text-embedding-3-small → 1536 dimensions
    - Ollama nomic-embed-text → 768 dimensions

Usage:
    from src.tools.vector_store import compute_similarity

    score = compute_similarity("Built Kafka pipelines at scale", "Event streaming experience required")
    print(score)  # ~0.82 (high similarity despite zero word overlap)
"""

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import settings

logger = logging.getLogger(__name__)

# Vector dimensions per provider
# OpenAI text-embedding-3-small produces 1536-dimensional vectors
# Ollama nomic-embed-text produces 768-dimensional vectors
VECTOR_DIMS = {
    "openai": 1536,
    "ollama": 768,
}

"""Purpose: To Create a Qdrant client connected to our Docker instance."""
def get_qdrant_client() -> QdrantClient:
    
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


def get_embeddings():
    """Purpose: Return the embedding model matching our configured LLM provider.

    Why match the LLM provider?
        If someone uses Ollama for LLM (to avoid OpenAI costs), they
        probably don't want to pay for OpenAI embeddings either.
        This keeps the entire stack consistent either fully cloud
        or fully local.

    Returns:
        A LangChain Embeddings object with an .embed_query() method
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
    else:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=settings.ollama_base_url,
        )


def get_vector_dim() -> int:
    """Purpose: Get vector dimension for the current embedding provider."""
    return VECTOR_DIMS.get(settings.llm_provider.lower(), 1536)


def ensure_collection(client: QdrantClient) -> None:
    """Purpose: Create the Qdrant collection if it doesn't exist.

    A 'collection' in Qdrant is like a table in a database
    it holds vectors of a fixed dimension with a specific distance metric.
    We use COSINE distance because it measures directional similarity
    regardless of vector magnitude.
    """
    existing = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=get_vector_dim(),
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")
    else:
        logger.debug(f"Collection {settings.qdrant_collection} already exists")


def compute_similarity(text_a: str, text_b: str) -> float:
    """Purpose: Compute cosine similarity between two pieces of text.

    This is the core function the Fit Analyst agent will use. It:
    1. Converts both texts into embedding vectors
    2. Computes cosine similarity between the vectors
    3. Returns a score from 0.0 (completely different) to 1.0 (identical meaning)

    Args:
        text_a: First text (typically resume content)
        text_b: Second text (typically JD content)

    Returns:
        Float between 0.0 and 1.0

    Why compute manually instead of using Qdrant's search?
        Qdrant search is for "find the most similar vectors in a collection."
        Here we're comparing exactly TWO texts so we don't need a database
        lookup, just a direct similarity calculation. Faster and simpler.

    The math (cosine similarity):
        cos(A, B) = (A · B) / (|A| × |B|)
        where · is dot product and |X| is vector magnitude
    """
    embedder = get_embeddings()

    # Generate embedding vectors for both texts
    vec_a = embedder.embed_query(text_a)
    vec_b = embedder.embed_query(text_b)

    # Dot product: sum of element-wise multiplication
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Magnitudes: square root of sum of squares
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5

    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    # Cosine similarity, clamped to [0, 1]
    similarity = dot_product / (magnitude_a * magnitude_b)
    return max(0.0, min(1.0, similarity))