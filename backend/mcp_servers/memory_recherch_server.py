from typing import Any
from typing import Any, Callable, Optional, Dict, List
from dotenv import load_dotenv
import os
import certifi
from backend.memory.lang_memory import MemoryType
from fastmcp import FastMCP, Context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers

from ..rag.rag import retrieve_from_rag
from ..memory.lang_memory import memory


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP("Memory-RAG-Server")


# ============================================================
# REQUEST CONTEXT MIDDLEWARE
# ============================================================

class UserContextMiddleware(Middleware):
    async def on_request(
        self,
        context: MiddlewareContext,
        call_next,
    ):
        headers = get_http_headers()

        user_id = headers.get("x-user-id", "test_user")
        thread_id = headers.get("x-thread-id", "test_thread")

        await context.fastmcp_context.set_state("user_id", user_id)
        await context.fastmcp_context.set_state("thread_id", thread_id)

        return await call_next(context)


mcp.add_middleware(UserContextMiddleware())


# ============================================================
# TOOL 1 — RAG SEARCH
# ============================================================

@mcp.tool
def search_rag(
    query: str,
    k: int = 4,
    ctx: Context | None = None,
) -> list:
    """
    Search information from the RAG system.

    The user_id and thread_id are obtained from the
    MCP request context and are not provided by the LLM.
    """

    if ctx is None:
        raise RuntimeError("MCP Context is required")

    thread_id = ctx.get_state("thread_id")

    if not thread_id:
        raise RuntimeError("thread_id is missing from context")

    return retrieve_from_rag(
        query=query,
        thread_id=thread_id,
        k=k,
    )


# ============================================================
# TOOL 2 — MEMORY SEARCH
# ============================================================

@mcp.tool
def memory_search(
    query: str,
    top_k: int = 5,
    memory_types: Optional[list[str]] = None,
    ctx: Context | None = None,
) -> list[dict]:
    """
    Search long-term memories relevant to a query.

    user_id is obtained automatically from the MCP context.
    """

    if ctx is None:
        raise RuntimeError("MCP Context is required")

    user_id = ctx.get_state("user_id")

    if user_id is None or not isinstance(user_id, str) or not user_id.strip():
        # raise RuntimeError("user is none or not str")
        user_id = "user_123"

    results = memory.search_memories(
        query=query,
        user_id=user_id,
        memory_types=memory_types,
        top_k=top_k,
    )

    return [
        {
            "id": result.memory.id,
            "content": result.memory.content,
            "memory_type": result.memory.memory_type,
            "score": result.score,
            "importance": result.memory.importance_score,
            "confidence": result.memory.confidence_score,
        }
        for result in results
    ]


# ============================================================
# TOOL 3 — MEMORY SAVE
# ============================================================

@mcp.tool
def memory_save(
    content: str,
    memory_type: str = "auto",
    ctx: Context | None = None,
) -> dict:
    """
    Save durable user-specific information into long-term memory.

    Use this tool only for information that is useful beyond
    the current conversation.

    Examples:
    - user preferences
    - user profile information
    - long-term goals
    - travel preferences
    - plans
    - decisions
    - previous experiences

    Do NOT save temporary information such as:
    - current weather
    - temporary hotel prices
    - temporary flight prices
    - temporary availability
    - temporary search results

    The user_id is obtained automatically from the MCP request context.
    """

    if ctx is None:
        raise RuntimeError("MCP Context is required")

    user_id = ctx.get_state("user_id")

    if user_id is None or not isinstance(user_id, str) or not user_id.strip():
        # raise RuntimeError("user is none or not str")
        user_id = "user_123"

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
# TOOL 4 — MEMORY CONTEXT
# ============================================================

@mcp.tool
def memory_context(
    query: str,
    max_tokens: int = 1000,
    top_k: int = 10,
    ctx: Context | None = None,
) -> str:
    """
    Build an LLM-ready context from long-term memories.

    user_id is obtained automatically from the MCP context.
    """

    if ctx is None:
        raise RuntimeError("MCP Context is required")

    user_id = ctx.get_state("user_id")

    if user_id is None or not isinstance(user_id, str) or not user_id.strip():
        # raise RuntimeError("user is none or not str")
        user_id = "user_123"

    return memory.build_context(
        query=query,
        user_id=user_id,
        max_tokens=max_tokens,
        top_k=top_k,
    )


# ============================================================
# TOOL 5 — MEMORY UPDATE
# ============================================================

# @mcp.tool
# def memory_update(
#     memory_id: str,
#     content: str,
#     ctx: Context | None = None,
# ) -> dict:
#     """
#     Update an existing long-term memory.
#     """

#     if ctx is None:
#         raise RuntimeError("MCP Context is required")

#     user_id = ctx.get_state("user_id")

#     if not user_id:
#         raise RuntimeError("user_id is missing from context")

#     # IMPORTANT:
#     # Ideally your memory layer should verify that
#     # memory_id belongs to this user_id before updating.

#     result = memory.update_memory(
#         memory_id=memory_id,
#         content=content,
#     )

#     return {
#         "success": True,
#         "memory_id": result.id,
#         "content": result.content,
#         "memory_type": result.memory_type,
#         "version": result.version,
#     }


# ============================================================
# TOOL 6 — MEMORY DELETE
# ============================================================

# @mcp.tool
# def memory_delete(
#     memory_id: str,
#     ctx: Context | None = None,
# ) -> dict:
#     """
#     Soft-delete a long-term memory.
#     """

#     if ctx is None:
#         raise RuntimeError("MCP Context is required")

#     user_id = ctx.get_state("user_id")

#     if user_id is None or not isinstance(user_id, str) or not user_id.strip():
#         # raise RuntimeError("user is none or not str")
#         user_id = "user_123"

#     # IMPORTANT:
#     # Ideally your memory layer should verify that
#     # memory_id belongs to this user_id before deleting.

#     memory.soft_delete_memory(memory_id)

#     return {
#         "success": True,
#         "memory_id": memory_id,
#         "status": "deleted",
#     }


# ============================================================
# TOOL 7 — GET MEMORY DESCRIPTIONS 
# ============================================================


@mcp.tool
def memory_get_types() -> List[Dict[str, str]]:
    """
    Return all available long-term memory types and their descriptions.

    Use this tool when you need to understand which memory type
    should be used when saving user information.
    """
    return MemoryType.get_llm_descriptions()

@mcp.tool
def memory_get_values() -> List[str]:
    """
    Return all available memory type values, including AUTO.
    """
    return MemoryType.all_values()

@mcp.tool
def memory_get_type_description(memory_type: str) -> str:
    """
    Return the description of a specific memory type.

    Args:
        memory_type: Memory type value.
    """
    return MemoryType.get_description(memory_type)

@mcp.tool
def memory_get_schema() -> Dict[str, Any]:
    """
    Return the JSON schema used to select a memory type.

    This schema can be used by the LLM when deciding
    which memory type should be used.
    """
    return MemoryType.get_llm_schema()

# ============================================================
# SERVER ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    mcp.run(
        transport="sse",
        host="127.0.0.1",
        port=8001
    )    