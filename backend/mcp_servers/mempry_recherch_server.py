# mcp_server.py

from typing import Any
from pathlib import Path
import os

from dotenv import load_dotenv
import certifi
from fastmcp import FastMCP

from rag.rag import retrieve_from_rag
from memory.lang_memory import memory


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP(
    name="Memory-RAG-Server"
)


# ============================================================
# RAG TOOL
# ============================================================

@mcp.tool
def search_rag(
    query: str,
    k: int = 4,
) -> Any:
    """
    Search the RAG knowledge base.

    Use this tool when the user asks about information
    stored in the application's documents or knowledge base.

    Args:
        query: The search question or query.
        k: Number of relevant documents to retrieve.

    Returns:
        Relevant information retrieved from the RAG system.
    """

    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    k = min(k, 20)

    return retrieve_from_rag(
        query=query,
        k=k,
    )


# ============================================================
# MEMORY SEARCH
# ============================================================

@mcp.tool
def memory_search(
    query: str,
    user_id: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Search the user's long-term memories.

    Use this tool when the LLM needs to retrieve previously
    stored information about the user.

    Args:
        query: Natural-language search query.
        user_id: Unique identifier of the user.
        top_k: Maximum number of memories to return.

    Returns:
        Relevant long-term memories with scores.
    """

    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    if not user_id or not user_id.strip():
        raise ValueError("user_id cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    top_k = min(top_k, 20)

    results = memory.search_memories(
        query=query,
        user_id=user_id,
        top_k=top_k,
    )

    return [
        {
            "id": result.memory.id,
            "content": result.memory.content,
            "memory_type": result.memory.memory_type,
            "score": round(float(result.score), 4),
            "semantic_score": round(
                float(result.semantic_score), 4
            ),
            "keyword_score": round(
                float(result.keyword_score), 4
            ),
            "entity_score": round(
                float(result.entity_score), 4
            ),
            "importance": round(
                float(result.memory.importance_score), 4
            ),
            "confidence": round(
                float(result.memory.confidence_score), 4
            ),
        }
        for result in results
    ]


# ============================================================
# MEMORY SAVE
# ============================================================

@mcp.tool
def memory_save(
    content: str,
    user_id: str,
    memory_type: str = "auto",
) -> dict[str, Any]:
    """
    Save information into long-term memory.

    The LLM should use this tool when it identifies information
    that should be remembered for future conversations.

    Args:
        content: Information to store.
        user_id: Unique identifier of the user.
        memory_type: Memory type or 'auto'.

    Returns:
        Information about the created or updated memory.
    """

    if not content or not content.strip():
        raise ValueError("content cannot be empty")

    if not user_id or not user_id.strip():
        raise ValueError("user_id cannot be empty")

    allowed_types = {
        "auto",
        "semantic",
        "episodic",
        "profile",
        "procedural",
        "entity",
        "relationship",
        "preference",
        "fact",
        "event",
    }

    if memory_type not in allowed_types:
        raise ValueError(
            f"Invalid memory_type. "
            f"Allowed values: {sorted(allowed_types)}"
        )

    result = memory.add_memory(
        content=content,
        user_id=user_id,
        memory_type=memory_type,
        source_type="agent",
    )

    return {
        "success": True,
        "memory_id": result.id,
        "content": result.content,
        "memory_type": result.memory_type,
        "importance": result.importance_score,
        "confidence": result.confidence_score,
    }


# ============================================================
# MEMORY CONTEXT
# ============================================================

@mcp.tool
def memory_context(
    query: str,
    user_id: str,
    max_tokens: int = 1000,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Build an LLM-ready context from long-term memory.

    This is useful when the agent wants a compact context
    containing the most relevant memories.

    Args:
        query: The current user query or topic.
        user_id: Unique identifier of the user.
        max_tokens: Maximum estimated context size.
        top_k: Number of memories considered.

    Returns:
        A context string and metadata.
    """

    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    if not user_id or not user_id.strip():
        raise ValueError("user_id cannot be empty")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    top_k = min(top_k, 50)

    context = memory.build_context(
        query=query,
        user_id=user_id,
        max_tokens=max_tokens,
        top_k=top_k,
    )

    return {
        "success": True,
        "query": query,
        "context": context,
    }


# ============================================================
# MEMORY UPDATE
# ============================================================

@mcp.tool
def memory_update(
    memory_id: str,
    content: str,
) -> dict[str, Any]:
    """
    Update an existing long-term memory.

    Use this tool when an existing memory is no longer correct
    or needs to be modified.

    Args:
        memory_id: ID of the memory to update.
        content: New memory content.

    Returns:
        Updated memory information.
    """

    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id cannot be empty")

    if not content or not content.strip():
        raise ValueError("content cannot be empty")

    result = memory.update_memory(
        memory_id=memory_id,
        content=content,
    )

    return {
        "success": True,
        "memory_id": result.id,
        "content": result.content,
        "memory_type": result.memory_type,
        "version": result.version,
        "importance": result.importance_score,
        "confidence": result.confidence_score,
    }


# ============================================================
# MEMORY DELETE
# ============================================================

@mcp.tool
def memory_delete(
    memory_id: str,
) -> dict[str, Any]:
    """
    Soft-delete a long-term memory.

    The memory is not permanently destroyed. It is marked
    as deleted and removed from normal retrieval.

    Args:
        memory_id: ID of the memory to delete.

    Returns:
        Deletion status.
    """

    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id cannot be empty")

    memory.soft_delete_memory(
        memory_id=memory_id,
    )

    return {
        "success": True,
        "memory_id": memory_id,
        "status": "deleted",
    }


# ============================================================
# SERVER ENTRY POINT
# ============================================================

# if __name__ == "__main__":
#     mcp.run()