"""
EnterpriseLongTermMemory — Production-Grade Long-Term Memory Engine
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import re
import sqlite3
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

logger = logging.getLogger("EnterpriseLongTermMemory")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"\b[a-zA-Z0-9_'-]+\b", text)


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    if np is not None:
        a = np.asarray(vec1, dtype=np.float32)
        b = np.asarray(vec2, dtype=np.float32)
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
    else:
        dot = sum(x * y for x, y in zip(vec1, vec2))
        norm_a = math.sqrt(sum(x * x for x in vec1))
        norm_b = math.sqrt(sum(x * x for x in vec2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _sha256_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "i",
    "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "could", "should",
    "ought", "i'm", "you're", "he's", "she's", "it's", "we're",
    "they're", "i've", "you've", "we've", "they've", "i'd", "you'd",
    "he'd", "she'd", "we'd", "they'd", "i'll", "you'll", "he'll",
    "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't",
    "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't",
    "cannot", "couldn't", "mustn't", "let's", "that's", "who's",
    "what's", "here's", "there's",
}


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROFILE = "profile"
    PROCEDURAL = "procedural"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    PREFERENCE = "preference"
    FACT = "fact"
    EVENT = "event"
    WORKING = "working"
    TASK = "task"
    GOAL = "goal"
    CONVERSATION = "conversation"
    SUMMARY = "summary"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    persist_directory: str = "./memory_db"
    vector_backend: str = "auto"  # "auto", "memory", "chroma"
    storage_backend: str = "sqlite"
    collection_name: str = "long_term_memories"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache_ttl: int = 3600

    default_top_k: int = 10
    candidate_pool_size: int = 50
    reranker_candidate_pool_size: int = 50
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_weight: float = 0.4

    similarity_threshold: float = 0.80
    duplicate_threshold: float = 0.90
    contradiction_threshold: float = 0.75

    decay_rate: float = 0.005
    enable_decay: bool = True

    enable_graph: bool = True

    enable_consolidation: bool = True
    retention_archive_threshold: float = 0.20
    retention_delete_threshold: float = 0.05
    allow_hard_delete: bool = False

    enable_cache: bool = True
    cache_ttl: int = 300

    scope_memory_by_user: bool = True
    max_content_length: int = 2000
    default_namespace: str = "default"
    log_level: str = "INFO"

    entity_terms: list[str] = field(
        default_factory=lambda: [
            "Python", "LangChain", "LangGraph", "MCP", "ChromaDB",
            "SQLite", "PostgreSQL", "OpenAI", "Azure", "AWS",
            "Docker", "Kubernetes", "FastAPI", "Pydantic",
            "TensorFlow", "PyTorch",
        ]
    )

    retrieval_weights: dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.30, "keyword": 0.15, "entity": 0.15,
            "recency": 0.10, "importance": 0.10, "confidence": 0.10,
            "frequency": 0.10,
        }
    )

    duplicate_weights: dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.50, "lexical": 0.30, "metadata": 0.10,
            "entity": 0.10,
        }
    )

    def __post_init__(self) -> None:
        if not self.persist_directory:
            raise ValueError("persist_directory must not be empty")
        if self.vector_backend not in ("auto", "memory", "chroma"):
            raise ValueError("vector_backend must be one of: auto, memory, chroma")

        for weights_name in ("retrieval_weights", "duplicate_weights"):
            weights = getattr(self, weights_name)
            if weights is None:
                continue
            total = sum(weights.values())
            if total <= 0:
                continue
            for key in weights:
                weights[key] = weights[key] / total

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    @property
    def db_path(self) -> str:
        return str(Path(self.persist_directory) / "memories.db")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    content: str = ""
    normalized_content: str = ""
    memory_type: str = MemoryType.SEMANTIC.value

    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    namespace: str = "default"

    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    last_accessed_at: Optional[datetime] = None

    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    importance_score: float = 0.0
    confidence_score: float = 0.0
    relevance_score: float = 1.0

    access_count: int = 0
    retrieval_count: int = 0

    source: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    created_by: Optional[str] = None
    evidence: Optional[str] = None

    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    embedding: Optional[list[float]] = None
    hash: str = ""
    version: int = 1
    parent_id: Optional[str] = None
    supersedes_id: Optional[str] = None

    is_active: bool = True
    is_archived: bool = False
    is_deleted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "normalized_content": self.normalized_content,
            "memory_type": self.memory_type,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "created_at": _format_dt(self.created_at),
            "updated_at": _format_dt(self.updated_at),
            "last_accessed_at": _format_dt(self.last_accessed_at),
            "valid_from": _format_dt(self.valid_from),
            "valid_until": _format_dt(self.valid_until),
            "expires_at": _format_dt(self.expires_at),
            "importance_score": self.importance_score,
            "confidence_score": self.confidence_score,
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "retrieval_count": self.retrieval_count,
            "source": self.source,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "created_by": self.created_by,
            "evidence": self.evidence,
            "entities": self.entities,
            "topics": self.topics,
            "tags": self.tags,
            "metadata": self.metadata,
            "provenance": self.provenance,
            "embedding": self.embedding,
            "hash": self.hash,
            "version": self.version,
            "parent_id": self.parent_id,
            "supersedes_id": self.supersedes_id,
            "is_active": self.is_active,
            "is_archived": self.is_archived,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        return cls(
            id=data.get("id", uuid.uuid4().hex),
            content=data.get("content", ""),
            normalized_content=data.get("normalized_content", ""),
            memory_type=data.get("memory_type", MemoryType.SEMANTIC.value),
            user_id=data.get("user_id"),
            tenant_id=data.get("tenant_id"),
            session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
            namespace=data.get("namespace", "default"),
            created_at=_parse_dt(data.get("created_at")) or _utc_now(),
            updated_at=_parse_dt(data.get("updated_at")) or _utc_now(),
            last_accessed_at=_parse_dt(data.get("last_accessed_at")),
            valid_from=_parse_dt(data.get("valid_from")),
            valid_until=_parse_dt(data.get("valid_until")),
            expires_at=_parse_dt(data.get("expires_at")),
            importance_score=data.get("importance_score", 0.0),
            confidence_score=data.get("confidence_score", 0.0),
            relevance_score=data.get("relevance_score", 1.0),
            access_count=data.get("access_count", 0),
            retrieval_count=data.get("retrieval_count", 0),
            source=data.get("source"),
            source_type=data.get("source_type"),
            source_id=data.get("source_id"),
            created_by=data.get("created_by"),
            evidence=data.get("evidence"),
            entities=data.get("entities", []),
            topics=data.get("topics", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            provenance=data.get("provenance", {}),
            embedding=data.get("embedding"),
            hash=data.get("hash", ""),
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            supersedes_id=data.get("supersedes_id"),
            is_active=data.get("is_active", True),
            is_archived=data.get("is_archived", False),
            is_deleted=data.get("is_deleted", False),
        )


@dataclass
class MemoryVersion:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    memory_id: str = ""
    version: int = 1
    content: str = ""
    created_at: datetime = field(default_factory=_utc_now)
    reason: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class Entity:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    type: str = "Concept"
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)


@dataclass
class Relationship:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    subject_entity_id: str = ""
    predicate: str = "related_to"
    object_entity_id: str = ""
    confidence: float = 1.0
    memory_id: Optional[str] = None
    created_at: datetime = field(default_factory=_utc_now)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class MemorySearchResult:
    memory: Memory
    score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    entity_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    confidence_score: float = 0.0
    frequency_score: float = 0.0
    reranker_score: Optional[float] = None
    rank: int = 0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "entity_score": self.entity_score,
            "recency_score": self.recency_score,
            "importance_score": self.importance_score,
            "confidence_score": self.confidence_score,
            "frequency_score": self.frequency_score,
            "reranker_score": self.reranker_score,
            "rank": self.rank,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MemoryError(Exception):
    """Base exception."""


class MemoryValidationError(MemoryError):
    """Raised when validation fails."""


class MemoryNotFoundError(MemoryError):
    """Raised when a memory is not found."""


class MemoryStorageError(MemoryError):
    """Raised when storage operation fails."""


class MemorySearchError(MemoryError):
    """Raised when search fails."""


class MemoryEmbeddingError(MemoryError):
    """Raised when embedding generation fails."""


class MemorySecurityError(MemoryError):
    """Raised when security checks fail."""


class MemoryConfigurationError(MemoryError):
    """Raised when configuration is invalid."""


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class _TTLCache:
    def __init__(self, ttl: int = 300) -> None:
        self.ttl = ttl
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            timestamp, value = item
            if _utc_now().timestamp() - timestamp > self.ttl:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (_utc_now().timestamp(), value)
            if len(self._data) > 10_000:
                oldest_key = min(self._data, key=lambda k: self._data[k][0])
                self._data.pop(oldest_key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# ---------------------------------------------------------------------------
# Storage Backends
# ---------------------------------------------------------------------------

class MemoryStorageBackend:
    def initialize(self) -> None: raise NotImplementedError
    def create_memory(self, memory: Memory) -> None: raise NotImplementedError
    def get_memory(self, memory_id: str) -> Optional[Memory]: raise NotImplementedError
    def list_memories(self, user_id: Optional[str] = None, tenant_id: Optional[str] = None,
                      namespace: Optional[str] = None, active_only: bool = True,
                      include_deleted: bool = False, date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None, limit: Optional[int] = None,
                      offset: int = 0, memory_type: Optional[str] = None) -> list[Memory]:
        raise NotImplementedError
    def update_memory(self, memory: Memory) -> None: raise NotImplementedError
    def delete_memory(self, memory_id: str) -> None: raise NotImplementedError
    def soft_delete_memory(self, memory_id: str) -> None: raise NotImplementedError
    def restore_memory(self, memory_id: str) -> None: raise NotImplementedError
    def archive_memory(self, memory_id: str) -> None: raise NotImplementedError
    def add_version(self, version: MemoryVersion) -> None: raise NotImplementedError
    def get_versions(self, memory_id: str) -> list[MemoryVersion]: raise NotImplementedError
    def get_or_create_entity(self, name: str, entity_type: str,
                             metadata: Optional[dict] = None) -> str: raise NotImplementedError
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]: raise NotImplementedError
    def get_entity_by_name(self, name: str) -> Optional[Entity]: raise NotImplementedError
    def list_entities(self, entity_type: Optional[str] = None) -> list[Entity]: raise NotImplementedError
    def add_relationship(self, relationship: Relationship) -> None: raise NotImplementedError
    def get_relationships_for_entity(self, entity_id: str) -> list[Relationship]: raise NotImplementedError
    def delete_relationship(self, relationship_id: str) -> None: raise NotImplementedError
    def add_memory_entity(self, memory_id: str, entity_id: str) -> None: raise NotImplementedError
    def get_memory_ids_for_entity(self, entity_id: str) -> list[str]: raise NotImplementedError
    def search_by_keywords(self, terms: list[str], user_id: Optional[str] = None,
                           tenant_id: Optional[str] = None, namespace: Optional[str] = None,
                           date_from: Optional[datetime] = None,
                           date_to: Optional[datetime] = None,
                           limit: int = 50) -> list[str]: raise NotImplementedError
    def get_all_memory_ids(self, user_id: Optional[str] = None,
                           tenant_id: Optional[str] = None,
                           namespace: Optional[str] = None) -> list[str]: raise NotImplementedError
    def close(self) -> None: raise NotImplementedError


class _SQLiteMemoryStore(MemoryStorageBackend):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = None
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL;")
            self._connection.execute("PRAGMA foreign_keys=ON;")
        return self._connection

    def initialize(self) -> None:
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    normalized_content TEXT,
                    memory_type TEXT NOT NULL,
                    user_id TEXT, tenant_id TEXT, session_id TEXT, agent_id TEXT,
                    namespace TEXT DEFAULT 'default',
                    created_at TEXT, updated_at TEXT, last_accessed_at TEXT,
                    valid_from TEXT, valid_until TEXT, expires_at TEXT,
                    importance_score REAL DEFAULT 0.0,
                    confidence_score REAL DEFAULT 0.0,
                    relevance_score REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    retrieval_count INTEGER DEFAULT 0,
                    source TEXT, source_type TEXT, source_id TEXT,
                    created_by TEXT, evidence TEXT,
                    entities TEXT, topics TEXT, tags TEXT,
                    metadata TEXT, provenance TEXT,
                    embedding TEXT,
                    version INTEGER DEFAULT 1,
                    parent_id TEXT, supersedes_id TEXT, hash TEXT,
                    is_active INTEGER DEFAULT 1,
                    is_archived INTEGER DEFAULT 0,
                    is_deleted INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS memory_versions (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT,
                    created_at TEXT,
                    reason TEXT,
                    metadata TEXT,
                    FOREIGN KEY(memory_id) REFERENCES memories(id)
                );
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    type TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    subject_entity_id TEXT,
                    predicate TEXT,
                    object_entity_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    memory_id TEXT,
                    created_at TEXT,
                    valid_from TEXT,
                    valid_until TEXT,
                    metadata TEXT,
                    FOREIGN KEY(subject_entity_id) REFERENCES entities(id),
                    FOREIGN KEY(object_entity_id) REFERENCES entities(id)
                );
                CREATE TABLE IF NOT EXISTS memory_entities (
                    memory_id TEXT,
                    entity_id TEXT,
                    PRIMARY KEY (memory_id, entity_id),
                    FOREIGN KEY(memory_id) REFERENCES memories(id),
                    FOREIGN KEY(entity_id) REFERENCES entities(id)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user_active ON memories(user_id, is_active, updated_at);
                CREATE INDEX IF NOT EXISTS idx_memories_tenant ON memories(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(subject_entity_id);
                CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(object_entity_id);
            """)
            self._ensure_columns()
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to initialize SQLite storage: {exc}") from exc

    def _ensure_columns(self) -> None:
        """Add new columns to existing tables for backward compatibility."""
        conn = self._connect()
        try:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
            desired = {
                "tenant_id": "TEXT", "namespace": "TEXT DEFAULT 'default'",
                "expires_at": "TEXT", "relevance_score": "REAL DEFAULT 1.0",
                "tags": "TEXT", "metadata": "TEXT", "provenance": "TEXT",
                "parent_id": "TEXT", "supersedes_id": "TEXT", "hash": "TEXT",
                "is_deleted": "INTEGER DEFAULT 0"
            }
            for col, typ in desired.items():
                if col not in columns:
                    conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {typ}")
        except sqlite3.Error as exc:
            # Ignore duplicate column errors, raise others
            if "duplicate column name" not in str(exc).lower():
                raise MemoryStorageError(f"Failed to migrate schema: {exc}") from exc

    @staticmethod
    def _parse_json(value: Optional[str]) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except Exception:
            return value

    @staticmethod
    def _serialize_json(value: Any) -> str:
        return json.dumps(value, default=str)

    def create_memory(self, memory: Memory) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO memories (
                    id, content, normalized_content, memory_type, user_id, tenant_id,
                    session_id, agent_id, namespace, created_at, updated_at,
                    last_accessed_at, valid_from, valid_until, expires_at,
                    importance_score, confidence_score, relevance_score,
                    access_count, retrieval_count, source, source_type, source_id,
                    created_by, evidence, entities, topics, tags, metadata, provenance,
                    embedding, version, parent_id, supersedes_id, hash,
                    is_active, is_archived, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id, memory.content, memory.normalized_content, memory.memory_type,
                memory.user_id, memory.tenant_id, memory.session_id, memory.agent_id,
                memory.namespace, _format_dt(memory.created_at), _format_dt(memory.updated_at),
                _format_dt(memory.last_accessed_at), _format_dt(memory.valid_from),
                _format_dt(memory.valid_until), _format_dt(memory.expires_at),
                memory.importance_score, memory.confidence_score, memory.relevance_score,
                memory.access_count, memory.retrieval_count, memory.source,
                memory.source_type, memory.source_id, memory.created_by, memory.evidence,
                self._serialize_json(memory.entities),
                self._serialize_json(memory.topics),
                self._serialize_json(memory.tags),
                self._serialize_json(memory.metadata),
                self._serialize_json(memory.provenance),
                self._serialize_json(memory.embedding),
                memory.version, memory.parent_id, memory.supersedes_id, memory.hash,
                1 if memory.is_active else 0,
                1 if memory.is_archived else 0,
                1 if memory.is_deleted else 0,
            ))
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to create memory: {exc}") from exc

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            normalized_content=row["normalized_content"],
            memory_type=row["memory_type"],
            user_id=row["user_id"],
            tenant_id=row["tenant_id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            namespace=row["namespace"] or "default",
            created_at=_parse_dt(row["created_at"]) or _utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or _utc_now(),
            last_accessed_at=_parse_dt(row["last_accessed_at"]),
            valid_from=_parse_dt(row["valid_from"]),
            valid_until=_parse_dt(row["valid_until"]),
            expires_at=_parse_dt(row["expires_at"]),
            importance_score=row["importance_score"],
            confidence_score=row["confidence_score"],
            relevance_score=row["relevance_score"],
            access_count=row["access_count"],
            retrieval_count=row["retrieval_count"],
            source=row["source"],
            source_type=row["source_type"],
            source_id=row["source_id"],
            created_by=row["created_by"],
            evidence=row["evidence"],
            entities=self._parse_json(row["entities"]) or [],
            topics=self._parse_json(row["topics"]) or [],
            tags=self._parse_json(row["tags"]) or [],
            metadata=self._parse_json(row["metadata"]) or {},
            provenance=self._parse_json(row["provenance"]) or {},
            embedding=self._parse_json(row["embedding"]),
            version=row["version"],
            parent_id=row["parent_id"],
            supersedes_id=row["supersedes_id"],
            hash=row["hash"] or "",
            is_active=bool(row["is_active"]),
            is_archived=bool(row["is_archived"]),
            is_deleted=bool(row["is_deleted"]),
        )

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_memory(row)
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to get memory: {exc}") from exc

    def list_memories(self, user_id: Optional[str] = None, tenant_id: Optional[str] = None,
                      namespace: Optional[str] = None, active_only: bool = True,
                      include_deleted: bool = False, date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None, limit: Optional[int] = None,
                      offset: int = 0, memory_type: Optional[str] = None) -> list[Memory]:
        conn = self._connect()
        query = "SELECT * FROM memories WHERE 1=1"
        params: list[Any] = []
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        if active_only:
            query += " AND is_active = 1"
        if not include_deleted:
            query += " AND is_deleted = 0"
        if date_from is not None:
            query += " AND created_at >= ?"
            params.append(_format_dt(date_from))
        if date_to is not None:
            query += " AND created_at <= ?"
            params.append(_format_dt(date_to))
        if memory_type is not None:
            query += " AND memory_type = ?"
            params.append(memory_type)
        query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        try:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_memory(row) for row in rows]
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to list memories: {exc}") from exc

    def update_memory(self, memory: Memory) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                UPDATE memories SET
                    content=?, normalized_content=?, memory_type=?, user_id=?, tenant_id=?,
                    session_id=?, agent_id=?, namespace=?,
                    updated_at=?, last_accessed_at=?, valid_from=?, valid_until=?, expires_at=?,
                    importance_score=?, confidence_score=?, relevance_score=?,
                    access_count=?, retrieval_count=?, source=?, source_type=?, source_id=?,
                    created_by=?, evidence=?, entities=?, topics=?, tags=?, metadata=?,
                    provenance=?, embedding=?, version=?, parent_id=?, supersedes_id=?, hash=?,
                    is_active=?, is_archived=?, is_deleted=?
                WHERE id=?
            """, (
                memory.content, memory.normalized_content, memory.memory_type,
                memory.user_id, memory.tenant_id, memory.session_id, memory.agent_id,
                memory.namespace, _format_dt(memory.updated_at),
                _format_dt(memory.last_accessed_at), _format_dt(memory.valid_from),
                _format_dt(memory.valid_until), _format_dt(memory.expires_at),
                memory.importance_score, memory.confidence_score, memory.relevance_score,
                memory.access_count, memory.retrieval_count, memory.source,
                memory.source_type, memory.source_id, memory.created_by, memory.evidence,
                self._serialize_json(memory.entities),
                self._serialize_json(memory.topics),
                self._serialize_json(memory.tags),
                self._serialize_json(memory.metadata),
                self._serialize_json(memory.provenance),
                self._serialize_json(memory.embedding),
                memory.version, memory.parent_id, memory.supersedes_id, memory.hash,
                1 if memory.is_active else 0,
                1 if memory.is_archived else 0,
                1 if memory.is_deleted else 0,
                memory.id,
            ))
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to update memory: {exc}") from exc

    def delete_memory(self, memory_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM relationships WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memory_versions WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to delete memory: {exc}") from exc

    def soft_delete_memory(self, memory_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("UPDATE memories SET is_deleted = 1, is_active = 0 WHERE id = ?", (memory_id,))
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to soft delete memory: {exc}") from exc

    def restore_memory(self, memory_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("UPDATE memories SET is_deleted = 0, is_active = 1 WHERE id = ?", (memory_id,))
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to restore memory: {exc}") from exc

    def archive_memory(self, memory_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("UPDATE memories SET is_archived = 1, is_active = 0 WHERE id = ?", (memory_id,))
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to archive memory: {exc}") from exc

    def add_version(self, version: MemoryVersion) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO memory_versions (id, memory_id, version, content, created_at, reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (version.id, version.memory_id, version.version, version.content,
                  _format_dt(version.created_at), version.reason,
                  self._serialize_json(version.metadata)))
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to add memory version: {exc}") from exc

    def get_versions(self, memory_id: str) -> list[MemoryVersion]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM memory_versions WHERE memory_id = ? ORDER BY version ASC",
                (memory_id,)
            ).fetchall()
            return [MemoryVersion(
                id=row["id"], memory_id=row["memory_id"], version=row["version"],
                content=row["content"], created_at=_parse_dt(row["created_at"]) or _utc_now(),
                reason=row["reason"], metadata=self._parse_json(row["metadata"])
            ) for row in rows]
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to get versions: {exc}") from exc

    def get_or_create_entity(self, name: str, entity_type: str,
                             metadata: Optional[dict] = None) -> str:
        conn = self._connect()
        try:
            row = conn.execute("SELECT id FROM entities WHERE name = ? COLLATE NOCASE",
                               (name,)).fetchone()
            if row is not None:
                conn.execute("UPDATE entities SET updated_at = ? WHERE id = ?",
                             (_format_dt(_utc_now()), row["id"]))
                conn.commit()
                return row["id"]
            entity_id = uuid.uuid4().hex
            now = _utc_now()
            conn.execute("""
                INSERT INTO entities (id, name, type, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (entity_id, name, entity_type, self._serialize_json(metadata),
                  _format_dt(now), _format_dt(now)))
            conn.commit()
            return entity_id
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to get or create entity: {exc}") from exc

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if row is None:
                return None
            return Entity(
                id=row["id"], name=row["name"], type=row["type"],
                metadata=self._parse_json(row["metadata"]),
                created_at=_parse_dt(row["created_at"]) or _utc_now(),
                updated_at=_parse_dt(row["updated_at"]) or _utc_now(),
            )
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to get entity: {exc}") from exc

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM entities WHERE name = ? COLLATE NOCASE",
                               (name,)).fetchone()
            if row is None:
                return None
            return self.get_entity_by_id(row["id"])
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to get entity by name: {exc}") from exc

    def list_entities(self, entity_type: Optional[str] = None) -> list[Entity]:
        conn = self._connect()
        query = "SELECT * FROM entities"
        params: list[Any] = []
        if entity_type is not None:
            query += " WHERE type = ?"
            params.append(entity_type)
        try:
            rows = conn.execute(query, params).fetchall()
            return [Entity(
                id=row["id"], name=row["name"], type=row["type"],
                metadata=self._parse_json(row["metadata"]),
                created_at=_parse_dt(row["created_at"]) or _utc_now(),
                updated_at=_parse_dt(row["updated_at"]) or _utc_now(),
            ) for row in rows]
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to list entities: {exc}") from exc

    def add_relationship(self, relationship: Relationship) -> None:
        conn = self._connect()
        try:
            existing = conn.execute("""
                SELECT id FROM relationships
                WHERE subject_entity_id = ? AND predicate = ? AND object_entity_id = ?
                  AND valid_until IS NULL
            """, (relationship.subject_entity_id, relationship.predicate,
                  relationship.object_entity_id)).fetchone()
            if existing is not None:
                return
            conn.execute("""
                INSERT INTO relationships (
                    id, subject_entity_id, predicate, object_entity_id,
                    confidence, memory_id, created_at, valid_from, valid_until, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship.id, relationship.subject_entity_id,
                relationship.predicate, relationship.object_entity_id,
                relationship.confidence, relationship.memory_id,
                _format_dt(relationship.created_at),
                _format_dt(relationship.valid_from),
                _format_dt(relationship.valid_until),
                self._serialize_json(relationship.metadata),
            ))
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to add relationship: {exc}") from exc

    def get_relationships_for_entity(self, entity_id: str) -> list[Relationship]:
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT * FROM relationships
                WHERE subject_entity_id = ? OR object_entity_id = ?
            """, (entity_id, entity_id)).fetchall()
            return [Relationship(
                id=row["id"], subject_entity_id=row["subject_entity_id"],
                predicate=row["predicate"], object_entity_id=row["object_entity_id"],
                confidence=row["confidence"], memory_id=row["memory_id"],
                created_at=_parse_dt(row["created_at"]) or _utc_now(),
                valid_from=_parse_dt(row["valid_from"]),
                valid_until=_parse_dt(row["valid_until"]),
                metadata=self._parse_json(row["metadata"]),
            ) for row in rows]
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to get relationships: {exc}") from exc

    def delete_relationship(self, relationship_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM relationships WHERE id = ?", (relationship_id,))
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to delete relationship: {exc}") from exc

    def add_memory_entity(self, memory_id: str, entity_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                (memory_id, entity_id)
            )
            conn.commit()
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to add memory entity: {exc}") from exc

    def get_memory_ids_for_entity(self, entity_id: str) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT memory_id FROM memory_entities WHERE entity_id = ?", (entity_id,)
            ).fetchall()
            return [row["memory_id"] for row in rows]
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to get memories for entity: {exc}") from exc

    def search_by_keywords(self, terms: list[str], user_id: Optional[str] = None,
                           tenant_id: Optional[str] = None, namespace: Optional[str] = None,
                           date_from: Optional[datetime] = None,
                           date_to: Optional[datetime] = None,
                           limit: int = 50) -> list[str]:
        if not terms:
            return []
        conn = self._connect()
        query = "SELECT id FROM memories WHERE 1=1"
        params: list[Any] = []
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        if date_from is not None:
            query += " AND created_at >= ?"
            params.append(_format_dt(date_from))
        if date_to is not None:
            query += " AND created_at <= ?"
            params.append(_format_dt(date_to))
        term_clauses = []
        for term in terms:
            term_clauses.append("LOWER(content) LIKE ?")
            params.append(f"%{term.lower()}%")
        query += " AND (" + " OR ".join(term_clauses) + ")"
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        try:
            rows = conn.execute(query, params).fetchall()
            return [row["id"] for row in rows]
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to search by keywords: {exc}") from exc

    def get_all_memory_ids(self, user_id: Optional[str] = None,
                           tenant_id: Optional[str] = None,
                           namespace: Optional[str] = None) -> list[str]:
        conn = self._connect()
        query = "SELECT id FROM memories WHERE is_active = 1 AND is_deleted = 0"
        params: list[Any] = []
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        if namespace is not None:
            query += " AND namespace = ?"
            params.append(namespace)
        try:
            rows = conn.execute(query, params).fetchall()
            return [row["id"] for row in rows]
        except sqlite3.Error as exc:
            raise MemoryStorageError(f"Failed to get memory IDs: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None


# ---------------------------------------------------------------------------
# Vector Backends
# ---------------------------------------------------------------------------

class VectorStoreBackend:
    def add_embedding(self, memory_id: str, vector: list[float],
                      metadata: dict[str, Any]) -> None: raise NotImplementedError
    def search(self, query_vector: list[float], top_k: int = 10,
               filter_callback: Optional[Callable[[dict[str, Any]], bool]] = None
               ) -> list[tuple[str, float]]: raise NotImplementedError
    def delete_embedding(self, memory_id: str) -> None: raise NotImplementedError
    def count(self) -> int: raise NotImplementedError
    def persist(self) -> None: raise NotImplementedError
    def close(self) -> None: raise NotImplementedError


class _InMemoryVectorStore(VectorStoreBackend):
    def __init__(self, persist_file: Optional[str] = None) -> None:
        self.persist_file = persist_file
        self._vectors: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self._lock = threading.RLock()
        if self.persist_file and os.path.exists(self.persist_file):
            self._load()

    def _load(self) -> None:
        try:
            with open(self.persist_file, "rb") as file:
                self._vectors = pickle.load(file)
        except Exception as exc:
            logger.warning("Failed to load vector store from %s: %s", self.persist_file, exc)
            self._vectors = {}

    def add_embedding(self, memory_id: str, vector: list[float],
                      metadata: dict[str, Any]) -> None:
        with self._lock:
            self._vectors[memory_id] = (vector, metadata)

    def search(self, query_vector: list[float], top_k: int = 10,
               filter_callback: Optional[Callable[[dict[str, Any]], bool]] = None
               ) -> list[tuple[str, float]]:
        with self._lock:
            results: list[tuple[str, float]] = []
            for memory_id, (vector, metadata) in self._vectors.items():
                if filter_callback is not None and not filter_callback(metadata):
                    continue
                score = _cosine_similarity(query_vector, vector)
                results.append((memory_id, score))
            results.sort(key=lambda item: item[1], reverse=True)
            return results[:top_k]

    def delete_embedding(self, memory_id: str) -> None:
        with self._lock:
            self._vectors.pop(memory_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._vectors)

    def persist(self) -> None:
        if self.persist_file is None:
            return
        with self._lock:
            try:
                Path(self.persist_file).parent.mkdir(parents=True, exist_ok=True)
                with open(self.persist_file, "wb") as file:
                    pickle.dump(self._vectors, file)
            except Exception as exc:
                logger.warning("Failed to persist vector store: %s", exc)

    def close(self) -> None:
        self.persist()


class _ChromaVectorStore(VectorStoreBackend):
    def __init__(self, persist_directory: str, collection_name: str = "long_term_memories") -> None:
        if chromadb is None:
            raise ImportError("ChromaDB is not installed")
        self.client = chromadb.PersistentClient(
            path=str(Path(persist_directory) / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_embedding(self, memory_id: str, vector: list[float],
                      metadata: dict[str, Any]) -> None:
        safe_metadata = {k: str(v) for k, v in metadata.items()}
        self.collection.upsert(
            ids=[memory_id],
            embeddings=[vector],
            metadatas=[safe_metadata],
        )

    def search(self, query_vector: list[float], top_k: int = 10,
               filter_callback: Optional[Callable[[dict[str, Any]], bool]] = None
               ) -> list[tuple[str, float]]:
        fetch_k = max(top_k * 5, 50)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        output: list[tuple[str, float]] = []
        for memory_id, distance, metadata in zip(ids, distances, metadatas):
            if filter_callback is not None and not filter_callback(metadata):
                continue
            similarity = 1.0 / (1.0 + float(distance))
            output.append((memory_id, similarity))
        output.sort(key=lambda item: item[1], reverse=True)
        return output[:top_k]

    def delete_embedding(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def count(self) -> int:
        return self.collection.count()

    def persist(self) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class _EmbeddingModel:
    def __init__(self, model_name: str, cache_ttl: int = 3600) -> None:
        self.model_name = model_name
        self.cache = _TTLCache(ttl=cache_ttl)
        self._model = None
        self._lock = threading.RLock()

    def _load_model(self) -> None:
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not installed; using hash embedding")
            self._model = None
            return
        try:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as exc:
            logger.warning("Failed to load embedding model '%s': %s; falling back to hash",
                           self.model_name, exc)
            self._model = None

    def encode(self, text: str) -> list[float]:
        normalized = _normalize_text(text)
        cache_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        with self._lock:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
            if self._model is None:
                self._load_model()
            try:
                if self._model is not None:
                    vector = self._model.encode(normalized, normalize_embeddings=True).tolist()
                else:
                    vector = _hash_embedding(normalized, dim=384)
            except Exception as exc:
                raise MemoryEmbeddingError(f"Embedding generation failed: {exc}") from exc
            self.cache.set(cache_key, vector)
            return vector


def _hash_embedding(text: str, dim: int = 384) -> list[float]:
    tokens = _tokenize(text)
    vector = [0.0] * dim
    for token in tokens:
        for ngram_size in (1, 2):
            for i in range(len(token) - ngram_size + 1):
                ngram = token[i : i + ngram_size]
                digest = hashlib.md5(ngram.encode("utf-8")).hexdigest()
                index = int(digest[:8], 16) % dim
                sign = 1.0 if int(digest[8:16], 16) % 2 == 0 else -1.0
                vector[index] += sign
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class _Reranker:
    def __init__(self, model_name: str, enabled: bool) -> None:
        self.enabled = enabled and CrossEncoder is not None
        self.model = None
        self._lock = threading.RLock()
        if self.enabled:
            try:
                logger.info("Loading reranker model: %s", model_name)
                self.model = CrossEncoder(model_name)
            except Exception as exc:
                logger.warning("Failed to load reranker model '%s': %s; reranking disabled",
                               model_name, exc)
                self.enabled = False
                self.model = None

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not self.enabled or self.model is None:
            return [0.0] * len(documents)
        with self._lock:
            pairs = [(query, doc) for doc in documents]
            scores = self.model.predict(pairs)
            return [float(score) for score in scores]


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------

class EnterpriseLongTermMemory:
    """Advanced long-term memory engine.

    Public API:
        - save_memory / add_memory
        - get_memory / get_memories
        - search_memories / search
        - update_memory
        - delete_memory / soft_delete_memory / restore_memory / archive_memory
        - detect_duplicates / detect_contradictions
        - consolidate_memories
        - forget_memories / forget_memory
        - build_context
        - search_by_entity / search_by_type / search_by_time
        - get_statistics
        - save / close
    """

    def __init__(
        self,
        persist_directory: str = "./memory_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_backend: str = "auto",
        collection_name: str = "long_term_memories",
        similarity_threshold: float = 0.80,
        duplicate_threshold: float = 0.90,
        contradiction_threshold: float = 0.75,
        decay_rate: float = 0.005,
        default_top_k: int = 10,
        enable_reranker: bool = True,
        enable_graph: bool = True,
        enable_consolidation: bool = True,
        enable_decay: bool = True,
        enable_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        config_kwargs: dict[str, Any] = {
            "persist_directory": persist_directory,
            "embedding_model": embedding_model,
            "vector_backend": vector_backend,
            "collection_name": collection_name,
            "similarity_threshold": similarity_threshold,
            "duplicate_threshold": duplicate_threshold,
            "contradiction_threshold": contradiction_threshold,
            "decay_rate": decay_rate,
            "default_top_k": default_top_k,
            "enable_reranker": enable_reranker,
            "enable_graph": enable_graph,
            "enable_consolidation": enable_consolidation,
            "enable_decay": enable_decay,
            "enable_cache": enable_cache,
        }
        config_kwargs.update(kwargs)
        self.config = MemoryConfig(**config_kwargs)

        self._lock = threading.RLock()
        self.storage: MemoryStorageBackend = _SQLiteMemoryStore(self.config.db_path)
        self.vector_store: VectorStoreBackend = self._create_vector_store()
        self.embedding_model = _EmbeddingModel(
            model_name=self.config.embedding_model,
            cache_ttl=self.config.embedding_cache_ttl,
        )
        self.reranker = _Reranker(
            model_name=self.config.reranker_model,
            enabled=self.config.enable_reranker,
        )
        self._embedding_cache = _TTLCache(ttl=self.config.embedding_cache_ttl)
        self._search_cache = _TTLCache(ttl=self.config.cache_ttl)
        self._entity_cache = _TTLCache(ttl=self.config.cache_ttl)

        logger.info("EnterpriseLongTermMemory initialized with persist_directory=%s",
                    persist_directory)

    def _create_vector_store(self) -> VectorStoreBackend:
        backend = self.config.vector_backend
        if backend == "chroma":
            if chromadb is not None:
                return _ChromaVectorStore(self.config.persist_directory, self.config.collection_name)
            logger.warning("ChromaDB not installed; falling back to in-memory vector store")
            return _InMemoryVectorStore(
                persist_file=str(Path(self.config.persist_directory) / "vectors.pkl")
            )
        if backend in ("memory", "auto"):
            return _InMemoryVectorStore(
                persist_file=str(Path(self.config.persist_directory) / "vectors.pkl")
            )
        raise MemoryConfigurationError(f"Unknown vector backend: {backend}")

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        text = _normalize_text(text)
        if len(text) > self.config.max_content_length:
            raise MemoryValidationError(
                f"Content too long: {len(text)} chars (max {self.config.max_content_length})"
            )
        return text

    def _validate_memory_input(self, content: str, user_id: Optional[str],
                               memory_type: str, source_type: str,
                               tenant_id: Optional[str], namespace: str) -> None:
        if content is None or not content.strip():
            raise MemoryValidationError("Content must not be empty")
        allowed_types = {mt.value for mt in MemoryType} | {"auto"}
        if memory_type not in allowed_types:
            raise MemoryValidationError(f"Unknown memory_type: {memory_type}")
        allowed_source_types = {"conversation", "document", "user_input", "agent", "tool", "import"}
        if source_type not in allowed_source_types:
            raise MemoryValidationError(f"Unknown source_type: {source_type}")
        if user_id is not None and not isinstance(user_id, str):
            raise MemoryValidationError("user_id must be a string")
        if tenant_id is not None and not isinstance(tenant_id, str):
            raise MemoryValidationError("tenant_id must be a string")
        if not namespace:
            raise MemoryValidationError("namespace cannot be empty")

    def _classify_memory_type(self, content: str) -> str:
        lowered = content.lower()
        if any(phrase in lowered for phrase in ("i prefer", "i like", "i don't like",
                                                "i love", "i hate", "preference")):
            return MemoryType.PREFERENCE.value
        if any(phrase in lowered for phrase in ("my name is", "i am a", "i'm a",
                                                "i work as", "my job", "my role",
                                                "i live in", "my background")):
            return MemoryType.PROFILE.value
        if any(phrase in lowered for phrase in ("how to", "steps to", "procedure",
                                                "workflow", "instructions", "first,",
                                                "then,")):
            return MemoryType.PROCEDURAL.value
        if any(phrase in lowered for phrase in ("i attended", "i visited", "yesterday",
                                                "last week", "last month", "happened",
                                                "event")):
            return MemoryType.EVENT.value
        if any(phrase in lowered for phrase in ("works on", "uses", "likes", "knows",
                                                "belongs to", "depends on", "replaced by",
                                                "related to")):
            return MemoryType.RELATIONSHIP.value
        if content.startswith("fact:") or content.startswith("FACT:"):
            return MemoryType.FACT.value
        if content.startswith("entity:") or content.startswith("ENTITY:"):
            return MemoryType.ENTITY.value
        return MemoryType.SEMANTIC.value

    def _extract_facts(self, content: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _extract_entities(self, content: str) -> list[dict[str, str]]:
        cache_key = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if self.config.enable_cache:
            cached = self._entity_cache.get(cache_key)
            if cached is not None:
                return cached

        seen: dict[str, dict[str, str]] = {}
        for term in self.config.entity_terms:
            if re.search(rf"\b{re.escape(term)}\b", content, re.IGNORECASE):
                seen[term.lower()] = {"name": term, "type": "Technology"}

        sentences = re.split(r"(?<=[.!?])\s+", content)
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                if i == 0:
                    continue
                if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", word):
                    name = word.rstrip(",.;:!?")
                    if len(name) > 1 and name.lower() not in _STOPWORDS:
                        seen[name.lower()] = {"name": name, "type": "Concept"}

        for email in re.findall(r"\S+@\S+\.\S+", content):
            seen[email.lower()] = {"name": email, "type": "Person"}
        for url in re.findall(r"https?://\S+", content):
            seen[url.lower()] = {"name": url, "type": "Organization"}

        result = list(seen.values())
        if self.config.enable_cache:
            self._entity_cache.set(cache_key, result)
        return result

    def _extract_topics(self, content: str, max_topics: int = 5) -> list[str]:
        tokens = [t for t in _tokenize(content) if t not in _STOPWORDS and len(t) > 2]
        counter = Counter(tokens)
        return [word for word, _ in counter.most_common(max_topics)]

    def _extract_tags(self, content: str, entities: list[dict[str, str]]) -> list[str]:
        return list({e["name"].lower() for e in entities})

    def _calculate_importance(self, content: str, memory_type: str,
                              entities: list[dict[str, str]], topics: list[str],
                              source_type: str,
                              explicit: Optional[float] = None) -> float:
        if explicit is not None:
            return _clamp(float(explicit))
        score = 0.40
        word_count = len(content.split())
        if word_count > 30:
            score += 0.10
        elif word_count > 10:
            score += 0.05
        if entities:
            score += 0.10
        if topics:
            score += 0.05
        source_bonus = {"document": 0.20, "user_input": 0.10, "agent": 0.05,
                        "conversation": 0.05, "tool": 0.00, "import": 0.15}.get(
                            source_type, 0.05)
        score += source_bonus
        important_markers = ("important", "always", "never", "remember", "must")
        if any(marker in content.lower() for marker in important_markers):
            score += 0.10
        if memory_type in (MemoryType.PREFERENCE.value, MemoryType.PROCEDURAL.value,
                           MemoryType.RELATIONSHIP.value):
            score += 0.05
        return _clamp(score)

    def _calculate_confidence(self, content: str, entities: list[dict[str, str]],
                              source_type: str,
                              explicit: Optional[float] = None) -> float:
        if explicit is not None:
            return _clamp(float(explicit))
        base = {"document": 0.90, "import": 0.80, "user_input": 0.70,
                "conversation": 0.60, "agent": 0.60, "tool": 0.50}.get(source_type, 0.60)
        if entities:
            base += 0.05
        if content.strip().endswith("?"):
            base -= 0.10
        return _clamp(base, 0.10, 0.99)

    def _extract_relationships_from_content(self, content: str,
                                            entities: list[dict[str, str]]) -> list[Relationship]:
        if not self.config.enable_graph:
            return []
        entity_names = [e["name"] for e in entities]
        relationships: list[Relationship] = []
        predicate_patterns = ["works on", "work on", "uses", "use", "likes", "prefers",
                              "knows", "belongs to", "depends on", "replaced by",
                              "related to"]
        lowered = content.lower()
        for predicate in predicate_patterns:
            for match in re.finditer(rf"\b{re.escape(predicate)}\b", lowered):
                left_part = content[:match.start()].strip()
                right_part = content[match.end():].strip()
                subject = None
                object_entity = None
                for name in sorted(entity_names, key=len, reverse=True):
                    if re.search(rf"\b{re.escape(name)}\b", left_part, re.IGNORECASE):
                        subject = name
                        break
                for name in sorted(entity_names, key=len, reverse=True):
                    if re.search(rf"\b{re.escape(name)}\b", right_part, re.IGNORECASE):
                        object_entity = name
                        break
                if subject and object_entity and subject != object_entity:
                    relationships.append(Relationship(
                        subject_entity_id=subject,
                        predicate=predicate,
                        object_entity_id=object_entity,
                        confidence=0.8,
                    ))
        return relationships

    def _lexical_similarity(self, text1: str, text2: str) -> float:
        sequence_ratio = SequenceMatcher(None, text1, text2).ratio()
        tokens1 = set(_tokenize(text1))
        tokens2 = set(_tokenize(text2))
        if not tokens1 or not tokens2:
            token_jaccard = 0.0
        else:
            token_jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        return 0.5 * sequence_ratio + 0.5 * token_jaccard

    def _entity_overlap_score(self, entities1: list[str], entities2: list[str]) -> float:
        set1 = set(e.lower() for e in entities1)
        set2 = set(e.lower() for e in entities2)
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def _metadata_similarity(self, mem1: Memory, mem2: Memory) -> float:
        score = 0.0
        if mem1.memory_type == mem2.memory_type:
            score += 0.4
        if mem1.user_id == mem2.user_id:
            score += 0.3
        if mem1.tenant_id == mem2.tenant_id:
            score += 0.2
        if mem1.source_type == mem2.source_type:
            score += 0.1
        if set(mem1.topics) & set(mem2.topics):
            score += 0.1
        return _clamp(score)

    def _detect_contradiction(self, new_content: str, existing: Memory,
                              new_entities: list[str]) -> float:
        if not new_entities or not existing.entities:
            return 0.0
        if not (set(new_entities) & set(existing.entities)):
            return 0.0
        negation_terms = ("not ", "no longer", "stopped", "never", "don't", "doesn't",
                          "isn't", "aren't", "wasn't", "weren't")
        new_has_negation = any(term in new_content.lower() for term in negation_terms)
        existing_has_negation = any(term in existing.content.lower() for term in negation_terms)
        if new_has_negation and not existing_has_negation:
            lexical = self._lexical_similarity(new_content, existing.content)
            if lexical > 0.30:
                return 0.85
        return 0.0

    # ------------------------------------------------------------------
    # Memory Write Pipeline
    # ------------------------------------------------------------------

    def save_memory(self, content: str, user_id: Optional[str] = None,
                    **kwargs: Any) -> Memory:
        """Alias for add_memory."""
        return self.add_memory(content=content, user_id=user_id, **kwargs)

    def add(self, content: str, user_id: Optional[str] = None, **kwargs: Any) -> Memory:
        """Alias for add_memory."""
        return self.add_memory(content=content, user_id=user_id, **kwargs)

    def add_memory(self, content: str, user_id: Optional[str] = None,
                   tenant_id: Optional[str] = None, session_id: Optional[str] = None,
                   agent_id: Optional[str] = None, memory_type: str = "auto",
                   namespace: str = "default", source: Optional[str] = None,
                   source_type: str = "user_input", source_id: Optional[str] = None,
                   created_by: Optional[str] = None, evidence: Optional[str] = None,
                   valid_from: Optional[datetime] = None,
                   valid_until: Optional[datetime] = None,
                   expires_at: Optional[datetime] = None,
                   importance_override: Optional[float] = None,
                   confidence_override: Optional[float] = None) -> Memory:
        """Add a new memory through the full write pipeline."""
        with self._lock:
            try:
                content = self._normalize_text(content)
                self._validate_memory_input(content, user_id, memory_type, source_type,
                                            tenant_id, namespace)

                if memory_type == "auto":
                    memory_type = self._classify_memory_type(content)

                extracted_entities = self._extract_entities(content)
                extracted_topics = self._extract_topics(content)
                extracted_tags = self._extract_tags(content, extracted_entities)
                facts = self._extract_facts(content)

                importance = self._calculate_importance(
                    content, memory_type, extracted_entities, extracted_topics,
                    source_type, importance_override)
                confidence = self._calculate_confidence(
                    content, extracted_entities, source_type, confidence_override)

                embedding = self.embedding_model.encode(content)

                hash_value = _sha256_id(user_id, tenant_id, namespace, _normalize_text(content))

                action, existing_memory, duplicate_info = self._find_duplicate_or_contradiction(
                    content=content, embedding=embedding, user_id=user_id,
                    tenant_id=tenant_id, namespace=namespace,
                    memory_type=memory_type, extracted_entities=extracted_entities,
                    extracted_topics=extracted_topics)

                if action == "DUPLICATE":
                    logger.debug("Duplicate memory detected for '%s'", content[:60])
                    existing_memory.access_count += 1
                    existing_memory.last_accessed_at = _utc_now()
                    existing_memory.confidence_score = _clamp(
                        existing_memory.confidence_score + 0.02, high=0.99)
                    self.storage.update_memory(existing_memory)
                    self.vector_store.add_embedding(
                        existing_memory.id, embedding, {"user_id": user_id})
                    return existing_memory

                if action == "CONTRADICTION":
                    logger.debug("Contradiction detected for '%s'", content[:60])
                    # Archive old memory, link new memory as superseding
                    existing_memory.valid_until = _utc_now()
                    existing_memory.is_active = False
                    existing_memory.is_archived = True
                    existing_memory.updated_at = _utc_now()
                    self.storage.update_memory(existing_memory)
                    self.vector_store.delete_embedding(existing_memory.id)

                if action in ("UPDATE", "MERGE"):
                    return self._merge_into_existing_memory(
                        existing=existing_memory, new_content=content, user_id=user_id,
                        tenant_id=tenant_id, namespace=namespace,
                        session_id=session_id, agent_id=agent_id,
                        memory_type=memory_type, source=source, source_type=source_type,
                        source_id=source_id, created_by=created_by, evidence=evidence,
                        extracted_entities=extracted_entities,
                        extracted_topics=extracted_topics,
                        extracted_tags=extracted_tags,
                        importance=importance, confidence=confidence,
                        valid_from=valid_from, valid_until=valid_until,
                        expires_at=expires_at)

                # NEW memory
                new_memory = Memory(
                    content=content,
                    normalized_content=_normalize_text(content),
                    memory_type=memory_type,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    namespace=namespace,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    expires_at=expires_at,
                    importance_score=importance,
                    confidence_score=confidence,
                    relevance_score=1.0,
                    source=source,
                    source_type=source_type,
                    source_id=source_id,
                    created_by=created_by,
                    evidence=evidence,
                    entities=[e["name"] for e in extracted_entities],
                    topics=extracted_topics,
                    tags=extracted_tags,
                    metadata={"facts": facts},
                    provenance={"source_type": source_type, "created_by": created_by},
                    embedding=embedding,
                    hash=hash_value,
                    version=1,
                    parent_id=None,
                    supersedes_id=existing_memory.id if action == "CONTRADICTION" else None,
                )
                self._store_new_memory_unlocked(new_memory, extracted_entities)
                return new_memory

            except MemoryError:
                raise
            except Exception as exc:
                raise MemoryError(f"Failed to add memory: {exc}") from exc

    def _find_duplicate_or_contradiction(self, content: str, embedding: list[float],
                                         user_id: Optional[str], tenant_id: Optional[str],
                                         namespace: str, memory_type: str,
                                         extracted_entities: list[dict[str, str]],
                                         extracted_topics: list[str]
                                         ) -> tuple[str, Optional[Memory], dict[str, float]]:
        entity_names = [e["name"] for e in extracted_entities]

        def filter_fn(metadata: dict[str, Any]) -> bool:
            if self.config.scope_memory_by_user and user_id is not None:
                if metadata.get("user_id") != user_id:
                    return False
            if tenant_id is not None and metadata.get("tenant_id") != tenant_id:
                return False
            if metadata.get("namespace") != namespace:
                return False
            return metadata.get("is_active", True)

        vector_results = self.vector_store.search(
            embedding, top_k=self.config.candidate_pool_size,
            filter_callback=filter_fn)

        best_action = "NEW"
        best_memory: Optional[Memory] = None
        best_scores: dict[str, float] = {}

        for memory_id, semantic_score in vector_results:
            existing = self.storage.get_memory(memory_id)
            if existing is None or existing.id == "" or existing.is_deleted:
                continue

            lexical = self._lexical_similarity(content, existing.content)
            entity_overlap = self._entity_overlap_score(entity_names, existing.entities)
            metadata_score = self._metadata_similarity(
                Memory(content=content, memory_type=memory_type, user_id=user_id,
                       tenant_id=tenant_id, namespace=namespace,
                       entities=entity_names, topics=extracted_topics),
                existing)

            dup_weights = self.config.duplicate_weights
            duplicate_score = (
                dup_weights["semantic"] * semantic_score +
                dup_weights["lexical"] * lexical +
                dup_weights["metadata"] * metadata_score +
                dup_weights["entity"] * entity_overlap)

            contradiction_score = self._detect_contradiction(content, existing, entity_names)

            scores = {
                "semantic_score": semantic_score,
                "lexical_score": lexical,
                "entity_overlap": entity_overlap,
                "metadata_score": metadata_score,
                "duplicate_score": duplicate_score,
                "contradiction_score": contradiction_score,
            }

            if contradiction_score > self.config.contradiction_threshold:
                return "CONTRADICTION", existing, scores

            if duplicate_score >= self.config.duplicate_threshold:
                # Only treat as duplicate if nearly identical (very high lexical or semantic)
                if semantic_score > 0.995 or lexical > 0.98:
                    return "DUPLICATE", existing, scores
                if entity_overlap > 0.70 and metadata_score > 0.70:
                    return "MERGE", existing, scores
                return "UPDATE", existing, scores


            if duplicate_score > best_scores.get("duplicate_score", 0.0):
                best_scores = scores
                best_memory = existing
                best_action = "NEW"

        return best_action, best_memory, best_scores

    def _store_new_memory_unlocked(self, memory: Memory,
                                   extracted_entities: list[dict[str, str]]) -> None:
        self.storage.create_memory(memory)
        self.vector_store.add_embedding(
            memory.id,
            memory.embedding if memory.embedding else [],
            {
                "user_id": memory.user_id,
                "tenant_id": memory.tenant_id,
                "namespace": memory.namespace,
                "memory_type": memory.memory_type,
                "is_active": memory.is_active,
                "created_at": _format_dt(memory.created_at),
            },
        )
        if self.config.enable_graph:
            for entity_dict in extracted_entities:
                entity_id = self.storage.get_or_create_entity(
                    entity_dict["name"], entity_dict["type"])
                self.storage.add_memory_entity(memory.id, entity_id)

            relationships = self._extract_relationships_from_content(
                memory.content, extracted_entities)
            for rel in relationships:
                subj_id = self.storage.get_or_create_entity(rel.subject_entity_id, "Concept")
                obj_id = self.storage.get_or_create_entity(rel.object_entity_id, "Concept")
                rel.subject_entity_id = subj_id
                rel.object_entity_id = obj_id
                rel.memory_id = memory.id
                self.storage.add_relationship(rel)
  
    def _merge_into_existing_memory(self, existing: Memory, new_content: str,
                                    user_id: Optional[str], tenant_id: Optional[str],
                                    namespace: str, session_id: Optional[str],
                                    agent_id: Optional[str], memory_type: str,
                                    source: Optional[str], source_type: str,
                                    source_id: Optional[str], created_by: Optional[str],
                                    evidence: Optional[str],
                                    extracted_entities: list[dict[str, str]],
                                    extracted_topics: list[str],
                                    extracted_tags: list[str],
                                    importance: float, confidence: float,
                                    valid_from: Optional[datetime],
                                    valid_until: Optional[datetime],
                                    expires_at: Optional[datetime]) -> Memory:
        version = MemoryVersion(
            memory_id=existing.id,
            version=existing.version,
            content=existing.content,
            created_at=_utc_now(),
            reason="merge_update",
        )
        self.storage.add_version(version)

        existing.content = new_content
        existing.normalized_content = _normalize_text(new_content)
        existing.updated_at = _utc_now()
        existing.memory_type = memory_type
        existing.user_id = user_id or existing.user_id
        existing.tenant_id = tenant_id or existing.tenant_id
        existing.namespace = namespace or existing.namespace
        existing.session_id = session_id or existing.session_id
        existing.agent_id = agent_id or existing.agent_id
        existing.source = source or existing.source
        existing.source_type = source_type
        existing.source_id = source_id or existing.source_id
        existing.created_by = created_by or existing.created_by
        existing.evidence = evidence or existing.evidence
        existing.valid_from = valid_from or existing.valid_from
        existing.valid_until = valid_until
        existing.expires_at = expires_at
        existing.importance_score = max(existing.importance_score, importance)
        existing.confidence_score = max(existing.confidence_score, confidence)
        existing.entities = list(set(existing.entities) | {e["name"] for e in extracted_entities})
        existing.topics = list(set(existing.topics) | set(extracted_topics))
        existing.tags = list(set(existing.tags) | set(extracted_tags))
        existing.metadata = {**existing.metadata,
                             **{"facts": self._extract_facts(new_content)}}
        existing.provenance = {**existing.provenance,
                               "updated_by": created_by, "updated_at": _format_dt(_utc_now())}
        existing.embedding = self.embedding_model.encode(new_content)
        existing.hash = _sha256_id(existing.user_id, existing.tenant_id,
                                   existing.namespace, _normalize_text(new_content))
        existing.version += 1

        self.storage.update_memory(existing)
        self.vector_store.add_embedding(
            existing.id, existing.embedding,
            {"user_id": existing.user_id, "tenant_id": existing.tenant_id,
             "namespace": existing.namespace, "memory_type": existing.memory_type,
             "is_active": existing.is_active, "created_at": _format_dt(existing.created_at)}
        )
        if self.config.enable_graph:
            for entity_dict in extracted_entities:
                entity_id = self.storage.get_or_create_entity(
                    entity_dict["name"], entity_dict["type"])
                self.storage.add_memory_entity(existing.id, entity_id)

            relationships = self._extract_relationships_from_content(
                new_content, extracted_entities)
            for rel in relationships:
                subj_id = self.storage.get_or_create_entity(rel.subject_entity_id, "Concept")
                obj_id = self.storage.get_or_create_entity(rel.object_entity_id, "Concept")
                rel.subject_entity_id = subj_id
                rel.object_entity_id = obj_id
                rel.memory_id = existing.id
                self.storage.add_relationship(rel)
        return existing

    # ------------------------------------------------------------------
    # Retrieval Methods
    # ------------------------------------------------------------------

    def search(self, query: str, user_id: Optional[str] = None, top_k: int = 10,
               **kwargs: Any) -> list[MemorySearchResult]:
        return self.search_memories(query=query, user_id=user_id, top_k=top_k, **kwargs)

    def retrieve(self, query: str, user_id: Optional[str] = None, top_k: int = 10,
                 **kwargs: Any) -> list[MemorySearchResult]:
        return self.search_memories(query=query, user_id=user_id, top_k=top_k, **kwargs)

    def retrieve_relevant_memories(self, query: str, user_id: Optional[str] = None,
                                   top_k: int = 10, **kwargs: Any) -> list[MemorySearchResult]:
        return self.search_memories(query=query, user_id=user_id, top_k=top_k, **kwargs)

    def search_memories(self, query: str, user_id: Optional[str] = None,
                        tenant_id: Optional[str] = None, namespace: str = "default",
                        top_k: int = 10, memory_types: Optional[list[str]] = None,
                        filters: Optional[dict[str, Any]] = None,
                        min_score: float = 0.0,
                        include_archived: bool = False,
                        include_deleted: bool = False,
                        time_range: Optional[tuple[datetime, datetime]] = None,
                        rerank: bool = True) -> list[MemorySearchResult]:
        """Hybrid search combining semantic, keyword, entity, recency, importance,
        confidence, and frequency. Returns ranked MemorySearchResult objects."""
        with self._lock:
            try:
                if top_k <= 0:
                    return []
                top_k = min(top_k, self.config.default_top_k * 5)
                query = _normalize_text(query)
                cache_key = hashlib.sha256(
                    f"{query}|{user_id}|{tenant_id}|{namespace}|{top_k}|"
                    f"{memory_types}|{filters}|{min_score}|{include_archived}|"
                    f"{include_deleted}|{time_range}|{rerank}".encode()
                ).hexdigest()
                if self.config.enable_cache:
                    cached = self._search_cache.get(cache_key)
                    if cached is not None:
                        return cached

                query_embedding = self.embedding_model.encode(query)
                query_tokens = _tokenize(query)
                query_entities = self._extract_entities(query)

                def filter_fn(metadata: dict[str, Any]) -> bool:
                    if self.config.scope_memory_by_user and user_id is not None:
                        if metadata.get("user_id") != user_id:
                            return False
                    if tenant_id is not None and metadata.get("tenant_id") != tenant_id:
                        return False
                    if metadata.get("namespace") != namespace:
                        return False
                    if not include_deleted and metadata.get("is_deleted", False):
                        return False
                    if not include_archived and metadata.get("is_archived", False):
                        return False
                    return metadata.get("is_active", True)

                semantic_candidates = self.vector_store.search(
                    query_embedding,
                    top_k=self.config.candidate_pool_size,
                    filter_callback=filter_fn,
                )

                keyword_ids = self.storage.search_by_keywords(
                    query_tokens, user_id=user_id, tenant_id=tenant_id, namespace=namespace,
                    date_from=time_range[0] if time_range else None,
                    date_to=time_range[1] if time_range else None,
                    limit=self.config.candidate_pool_size)

                if memory_types:
                    memory_types_set = set(memory_types)
                else:
                    memory_types_set = None

                candidate_ids = set()
                for mid, _ in semantic_candidates:
                    candidate_ids.add(mid)
                for mid in keyword_ids:
                    candidate_ids.add(mid)

                results: list[MemorySearchResult] = []
                now = _utc_now()
                for mid in candidate_ids:
                    mem = self.storage.get_memory(mid)
                    if mem is None:
                        continue
                    if mem.is_deleted and not include_deleted:
                        continue
                    if mem.is_archived and not include_archived:
                        continue
                    if self.config.scope_memory_by_user and user_id is not None and mem.user_id != user_id:
                        continue
                    if tenant_id is not None and mem.tenant_id != tenant_id:
                        continue
                    if mem.namespace != namespace:
                        continue
                    if memory_types_set and mem.memory_type not in memory_types_set:
                        continue
                    if time_range:
                        if mem.created_at < time_range[0] or mem.created_at > time_range[1]:
                            continue
                    if filters:
                        skip = False
                        for key, value in filters.items():
                            if getattr(mem, key, None) != value:
                                skip = True
                                break
                        if skip:
                            continue

                    semantic_score = 0.0
                    for sid, score in semantic_candidates:
                        if sid == mid:
                            semantic_score = score
                            break

                    keyword_score = 0.0
                    mem_tokens = set(_tokenize(mem.content))
                    if query_tokens:
                        overlap = len(set(query_tokens) & mem_tokens)
                        keyword_score = overlap / len(set(query_tokens))

                    entity_score = self._entity_overlap_score(
                        [e["name"] for e in query_entities], mem.entities)

                    age_seconds = max(0.0, (now - mem.updated_at).total_seconds())
                    recency_score = math.exp(-self.config.decay_rate * age_seconds / 86400.0) \
                        if self.config.enable_decay else 1.0

                    importance_score = mem.importance_score
                    confidence_score = mem.confidence_score

                    frequency_score = _clamp(
                        math.log(1 + mem.access_count + mem.retrieval_count) / 10.0)

                    weights = self.config.retrieval_weights
                    final_score = (
                        weights["semantic"] * semantic_score +
                        weights["keyword"] * keyword_score +
                        weights["entity"] * entity_score +
                        weights["recency"] * recency_score +
                        weights["importance"] * importance_score +
                        weights["confidence"] * confidence_score +
                        weights["frequency"] * frequency_score
                    )

                    if final_score < min_score:
                        continue

                    explanation = (
                        f"semantic={semantic_score:.2f}, keyword={keyword_score:.2f}, "
                        f"entity={entity_score:.2f}, recency={recency_score:.2f}, "
                        f"importance={importance_score:.2f}, confidence={confidence_score:.2f}"
                    )
                    results.append(MemorySearchResult(
                        memory=mem,
                        score=final_score,
                        semantic_score=semantic_score,
                        keyword_score=keyword_score,
                        entity_score=entity_score,
                        recency_score=recency_score,
                        importance_score=importance_score,
                        confidence_score=confidence_score,
                        frequency_score=frequency_score,
                        explanation=explanation,
                    ))

                results.sort(key=lambda r: r.score, reverse=True)

                if rerank and self.config.enable_reranker and len(results) > 1:
                    top_candidates = results[:self.config.reranker_candidate_pool_size]
                    docs = [r.memory.content for r in top_candidates]
                    reranker_scores = self.reranker.rerank(query, docs)
                    for i, r in enumerate(top_candidates):
                        r.reranker_score = reranker_scores[i]
                        r.score = ((1 - self.config.reranker_weight) * r.score +
                                   self.config.reranker_weight * reranker_scores[i])
                    results.sort(key=lambda r: r.score, reverse=True)

                results = results[:top_k]

                for idx, res in enumerate(results, start=1):
                    res.rank = idx
                    mem = res.memory
                    mem.access_count += 1
                    mem.retrieval_count += 1
                    mem.last_accessed_at = now
                    self.storage.update_memory(mem)

                if self.config.enable_cache:
                    self._search_cache.set(cache_key, results)
                return results

            except MemoryError:
                raise
            except Exception as exc:
                raise MemorySearchError(f"Search failed: {exc}") from exc

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get_memory(self, memory_id: str) -> Memory:
        with self._lock:
            memory = self.storage.get_memory(memory_id)
            if memory is None:
                raise MemoryNotFoundError(f"Memory {memory_id} not found")
            return memory

    def get_memories(self, user_id: Optional[str] = None,
                     tenant_id: Optional[str] = None,
                     namespace: Optional[str] = None,
                     active_only: bool = True,
                     include_deleted: bool = False,
                     limit: Optional[int] = None,
                     memory_type: Optional[str] = None) -> list[Memory]:
        with self._lock:
            return self.storage.list_memories(
                user_id=user_id, tenant_id=tenant_id, namespace=namespace,
                active_only=active_only, include_deleted=include_deleted,
                limit=limit, memory_type=memory_type)

    def update(self, memory_id: str, **kwargs: Any) -> Memory:
        return self.update_memory(memory_id, **kwargs)

    def update_memory(self, memory_id: str, content: Optional[str] = None,
                      user_id: Optional[str] = None,
                      tenant_id: Optional[str] = None,
                      namespace: Optional[str] = None,
                      memory_type: Optional[str] = None,
                      importance_override: Optional[float] = None,
                      confidence_override: Optional[float] = None) -> Memory:
        with self._lock:
            memory = self.storage.get_memory(memory_id)
            if memory is None:
                raise MemoryNotFoundError(f"Memory {memory_id} not found")

            version = MemoryVersion(
                memory_id=memory.id, version=memory.version,
                content=memory.content, created_at=_utc_now(), reason="manual_update")
            self.storage.add_version(version)

            if content is not None:
                memory.content = self._normalize_text(content)
                memory.normalized_content = _normalize_text(content)
            if user_id is not None:
                memory.user_id = user_id
            if tenant_id is not None:
                memory.tenant_id = tenant_id
            if namespace is not None:
                memory.namespace = namespace
            if memory_type is not None:
                memory.memory_type = memory_type
            if importance_override is not None:
                memory.importance_score = _clamp(float(importance_override))
            if confidence_override is not None:
                memory.confidence_score = _clamp(float(confidence_override))

            memory.updated_at = _utc_now()
            memory.version += 1
            if content is not None:
                memory.embedding = self.embedding_model.encode(memory.content)
                memory.entities = [e["name"] for e in self._extract_entities(memory.content)]
                memory.topics = self._extract_topics(memory.content)
                memory.tags = self._extract_tags(memory.content,
                                                 self._extract_entities(memory.content))
                memory.hash = _sha256_id(memory.user_id, memory.tenant_id,
                                         memory.namespace, memory.normalized_content)

            self.storage.update_memory(memory)
            self.vector_store.add_embedding(
                memory.id, memory.embedding if memory.embedding else [],
                {"user_id": memory.user_id, "tenant_id": memory.tenant_id,
                 "namespace": memory.namespace, "memory_type": memory.memory_type,
                 "is_active": memory.is_active, "created_at": _format_dt(memory.created_at)}
            )
            return memory

    def delete(self, memory_id: str) -> None:
        return self.delete_memory(memory_id)

    def delete_memory(self, memory_id: str) -> None:
        with self._lock:
            if self.storage.get_memory(memory_id) is None:
                raise MemoryNotFoundError(f"Memory {memory_id} not found")
            self.storage.delete_memory(memory_id)
            self.vector_store.delete_embedding(memory_id)

    def soft_delete_memory(self, memory_id: str) -> None:
        with self._lock:
            if self.storage.get_memory(memory_id) is None:
                raise MemoryNotFoundError(f"Memory {memory_id} not found")
            self.storage.soft_delete_memory(memory_id)
            self.vector_store.delete_embedding(memory_id)

    def restore_memory(self, memory_id: str) -> None:
        with self._lock:
            if self.storage.get_memory(memory_id) is None:
                raise MemoryNotFoundError(f"Memory {memory_id} not found")
            self.storage.restore_memory(memory_id)
            memory = self.storage.get_memory(memory_id)
            self.vector_store.add_embedding(
                memory_id, memory.embedding if memory.embedding else [],
                {"user_id": memory.user_id, "tenant_id": memory.tenant_id,
                 "namespace": memory.namespace, "memory_type": memory.memory_type,
                 "is_active": memory.is_active, "created_at": _format_dt(memory.created_at)}
            )

    def archive_memory(self, memory_id: str) -> None:
        with self._lock:
            if self.storage.get_memory(memory_id) is None:
                raise MemoryNotFoundError(f"Memory {memory_id} not found")
            self.storage.archive_memory(memory_id)
            self.vector_store.delete_embedding(memory_id)

    def clear_memory(self, user_id: Optional[str] = None,
                     tenant_id: Optional[str] = None,
                     namespace: Optional[str] = None) -> int:
        with self._lock:
            memories = self.storage.list_memories(user_id=user_id, tenant_id=tenant_id,
                                                  namespace=namespace,
                                                  active_only=False,
                                                  include_deleted=True)
            for mem in memories:
                self.storage.delete_memory(mem.id)
                self.vector_store.delete_embedding(mem.id)
            return len(memories)

    # ------------------------------------------------------------------
    # Entity / Relationship
    # ------------------------------------------------------------------

    def related_entities(self, entity_name: str) -> list[Entity]:
        with self._lock:
            entity = self.storage.get_entity_by_name(entity_name)
            if entity is None:
                return []
            relationships = self.storage.get_relationships_for_entity(entity.id)
            related_ids = set()
            for rel in relationships:
                if rel.subject_entity_id == entity.id:
                    related_ids.add(rel.object_entity_id)
                else:
                    related_ids.add(rel.subject_entity_id)
            return [self.storage.get_entity_by_id(eid) for eid in related_ids
                    if self.storage.get_entity_by_id(eid)]

    def get_related_memories(self, entity: str, top_k: int = 10) -> list[Memory]:
        with self._lock:
            entity_obj = self.storage.get_entity_by_name(entity)
            if entity_obj is None:
                return []
            memory_ids = self.storage.get_memory_ids_for_entity(entity_obj.id)
            memories = []
            for mid in memory_ids:
                mem = self.storage.get_memory(mid)
                if mem and mem.is_active and not mem.is_deleted:
                    memories.append(mem)
            memories.sort(key=lambda m: (m.importance_score, m.updated_at), reverse=True)
            return memories[:top_k]

    def search_by_entity(self, entity_name: str, user_id: Optional[str] = None,
                         tenant_id: Optional[str] = None,
                         namespace: Optional[str] = None,
                         top_k: int = 10) -> list[Memory]:
        with self._lock:
            memories = self.get_related_memories(entity_name, top_k=top_k)
            if user_id is not None:
                memories = [m for m in memories if m.user_id == user_id]
            if tenant_id is not None:
                memories = [m for m in memories if m.tenant_id == tenant_id]
            if namespace is not None:
                memories = [m for m in memories if m.namespace == namespace]
            return memories[:top_k]

    def search_by_type(self, memory_type: str, user_id: Optional[str] = None,
                       tenant_id: Optional[str] = None,
                       namespace: Optional[str] = None,
                       top_k: int = 10) -> list[Memory]:
        with self._lock:
            memories = self.storage.list_memories(
                user_id=user_id, tenant_id=tenant_id, namespace=namespace,
                memory_type=memory_type, active_only=True,
                include_deleted=False, limit=top_k)
            return memories[:top_k]

    def search_by_time(self, date_from: datetime, date_to: datetime,
                       user_id: Optional[str] = None,
                       tenant_id: Optional[str] = None,
                       namespace: Optional[str] = None,
                       top_k: int = 10) -> list[Memory]:
        with self._lock:
            memories = self.storage.list_memories(
                user_id=user_id, tenant_id=tenant_id, namespace=namespace,
                date_from=date_from, date_to=date_to, active_only=True,
                include_deleted=False, limit=top_k)
            return memories[:top_k]

    # ------------------------------------------------------------------
    # Duplicate / Contradiction Detection
    # ------------------------------------------------------------------

    def detect_duplicates(self, content: str, user_id: Optional[str] = None,
                          tenant_id: Optional[str] = None,
                          namespace: str = "default") -> list[MemorySearchResult]:
        """Return similar memories for a given content string."""
        with self._lock:
            embedding = self.embedding_model.encode(content)
            query_tokens = _tokenize(content)
            results: list[MemorySearchResult] = []
            for mid in self.storage.get_all_memory_ids(user_id=user_id,
                                                       tenant_id=tenant_id,
                                                       namespace=namespace):
                mem = self.storage.get_memory(mid)
                if mem is None:
                    continue
                lexical = self._lexical_similarity(content, mem.content)
                entity_overlap = self._entity_overlap_score(
                    [e["name"] for e in self._extract_entities(content)], mem.entities)
                score = 0.5 * lexical + 0.5 * entity_overlap
                if score >= self.config.duplicate_threshold:
                    results.append(MemorySearchResult(
                        memory=mem, score=score, semantic_score=0.0,
                        keyword_score=lexical, entity_score=entity_overlap,
                        explanation=f"lexical={lexical:.2f}, entity={entity_overlap:.2f}"
                    ))
            results.sort(key=lambda r: r.score, reverse=True)
            return results

    def detect_contradictions(self, content: str, user_id: Optional[str] = None,
                              tenant_id: Optional[str] = None,
                              namespace: str = "default") -> list[MemorySearchResult]:
        """Return memories that potentially contradict the given content."""
        with self._lock:
            embedding = self.embedding_model.encode(content)
            results: list[MemorySearchResult] = []
            for mid in self.storage.get_all_memory_ids(user_id=user_id,
                                                       tenant_id=tenant_id,
                                                       namespace=namespace):
                mem = self.storage.get_memory(mid)
                if mem is None:
                    continue
                contradiction_score = self._detect_contradiction(
                    content, mem, [e["name"] for e in self._extract_entities(content)])
                if contradiction_score > self.config.contradiction_threshold:
                    results.append(MemorySearchResult(
                        memory=mem, score=contradiction_score,
                        semantic_score=0.0, keyword_score=0.0,
                        entity_score=0.0, explanation="contradiction"
                    ))
            results.sort(key=lambda r: r.score, reverse=True)
            return results

    def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> Memory:
        """Mark new_memory_id as superseding old_memory_id."""
        with self._lock:
            old = self.storage.get_memory(old_memory_id)
            new = self.storage.get_memory(new_memory_id)
            if old is None or new is None:
                raise MemoryNotFoundError("One or both memories not found")
            old.is_active = False
            old.is_archived = True
            old.valid_until = _utc_now()
            old.updated_at = _utc_now()
            self.storage.update_memory(old)
            self.vector_store.delete_embedding(old.id)

            new.supersedes_id = old.id
            new.parent_id = old.id
            new.updated_at = _utc_now()
            self.storage.update_memory(new)
            return new

    def get_memory_history(self, memory_id: str) -> list[MemoryVersion]:
        with self._lock:
            return self.storage.get_versions(memory_id)

    # ------------------------------------------------------------------
    # Consolidation / Forgetting
    # ------------------------------------------------------------------

    def consolidate(self, user_id: Optional[str] = None) -> dict[str, Any]:
        return self.consolidate_memories(user_id=user_id)

    def consolidate_memories(self, user_id: Optional[str] = None,
                             tenant_id: Optional[str] = None,
                             namespace: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            all_memories = self.storage.list_memories(
                user_id=user_id, tenant_id=tenant_id, namespace=namespace,
                active_only=True)
            stats = {"merged": 0, "archived": 0, "updated": 0}

            merged_ids = set()
            for i, mem1 in enumerate(all_memories):
                if mem1.id in merged_ids:
                    continue
                vec1 = mem1.embedding
                if vec1 is None:
                    continue
                for j, mem2 in enumerate(all_memories):
                    if i == j or mem2.id in merged_ids:
                        continue
                    vec2 = mem2.embedding
                    if vec2 is None:
                        continue
                    similarity = _cosine_similarity(vec1, vec2)
                    if similarity >= self.config.duplicate_threshold:
                        if mem1.importance_score >= mem2.importance_score:
                            primary, secondary = mem1, mem2
                        else:
                            primary, secondary = mem2, mem1

                        secondary.valid_until = _utc_now()
                        secondary.is_active = False
                        secondary.is_archived = True
                        secondary.updated_at = _utc_now()
                        self.storage.update_memory(secondary)
                        self.vector_store.delete_embedding(secondary.id)
                        merged_ids.add(secondary.id)

                        primary.confidence_score = _clamp(
                            primary.confidence_score + 0.05, high=0.99)
                        primary.importance_score = max(primary.importance_score,
                                                      secondary.importance_score)
                        primary.entities = list(set(primary.entities) | set(secondary.entities))
                        primary.topics = list(set(primary.topics) | set(secondary.topics))
                        primary.tags = list(set(primary.tags) | set(secondary.tags))
                        primary.metadata = {**primary.metadata,
                                            **{"merged_from": secondary.id}}
                        primary.updated_at = _utc_now()
                        primary.version += 1
                        self.storage.update_memory(primary)
                        self.vector_store.add_embedding(
                            primary.id, primary.embedding if primary.embedding else [],
                            {"user_id": primary.user_id, "tenant_id": primary.tenant_id,
                             "namespace": primary.namespace,
                             "memory_type": primary.memory_type,
                             "is_active": primary.is_active,
                             "created_at": _format_dt(primary.created_at)}
                        )
                        stats["merged"] += 1
                        stats["archived"] += 1
                        break
            return stats

    def forget(self, user_id: Optional[str] = None) -> dict[str, Any]:
        return self.forget_memories(user_id=user_id)

    def forget_memory(self, memory_id: str) -> None:
        """Archive a single memory if retention score is low."""
        with self._lock:
            mem = self.storage.get_memory(memory_id)
            if mem is None:
                raise MemoryNotFoundError(f"Memory {memory_id} not found")
            mem.is_active = False
            mem.is_archived = True
            mem.updated_at = _utc_now()
            self.storage.update_memory(mem)
            self.vector_store.delete_embedding(mem.id)

    def forget_memories(self, user_id: Optional[str] = None,
                        tenant_id: Optional[str] = None,
                        namespace: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            all_memories = self.storage.list_memories(
                user_id=user_id, tenant_id=tenant_id, namespace=namespace,
                active_only=True)
            stats = {"archived": 0, "deleted": 0, "kept": 0}
            now = _utc_now()

            for mem in all_memories:
                age_seconds = max(0.0, (now - mem.updated_at).total_seconds())
                recency_factor = math.exp(-self.config.decay_rate * age_seconds / 86400.0)
                retention_score = (
                    mem.importance_score * mem.confidence_score *
                    mem.relevance_score * recency_factor
                )
                if retention_score < self.config.retention_delete_threshold and \
                        self.config.allow_hard_delete:
                    self.storage.delete_memory(mem.id)
                    self.vector_store.delete_embedding(mem.id)
                    stats["deleted"] += 1
                elif retention_score < self.config.retention_archive_threshold:
                    mem.is_active = False
                    mem.is_archived = True
                    mem.updated_at = now
                    self.storage.update_memory(mem)
                    self.vector_store.delete_embedding(mem.id)
                    stats["archived"] += 1
                else:
                    stats["kept"] += 1
            return stats

    # ------------------------------------------------------------------
    # Context Builder
    # ------------------------------------------------------------------

    def build_context(self, query: str, user_id: Optional[str] = None,
                      tenant_id: Optional[str] = None,
                      namespace: str = "default",
                      max_tokens: int = 1000, top_k: Optional[int] = None) -> str:
        """Build LLM-ready context from relevant memories."""
        if top_k is None:
            top_k = self.config.default_top_k
        results = self.search_memories(query=query, user_id=user_id,
                                       tenant_id=tenant_id, namespace=namespace,
                                       top_k=top_k)
        if not results:
            return ""

        lines = ["Relevant memories:"]
        current_tokens = 0
        for i, res in enumerate(results, start=1):
            snippet = res.memory.content
            estimated_tokens = len(snippet) // 4 + 1
            if current_tokens + estimated_tokens > max_tokens:
                break
            lines.append(f"{i}. {snippet}")
            current_tokens += estimated_tokens
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            all_memories = self.storage.list_memories(active_only=False,
                                                      include_deleted=True)
            total = len(all_memories)
            active = len([m for m in all_memories if m.is_active and not m.is_deleted])
            archived = len([m for m in all_memories if m.is_archived])
            deleted = len([m for m in all_memories if m.is_deleted])
            by_type = Counter(m.memory_type for m in all_memories)
            by_user = Counter(str(m.user_id) for m in all_memories if m.user_id)
            avg_confidence = sum(m.confidence_score for m in all_memories) / max(total, 1)
            avg_importance = sum(m.importance_score for m in all_memories) / max(total, 1)
            most_accessed = sorted(all_memories, key=lambda m: m.access_count, reverse=True)[:5]
            recently_created = sorted(all_memories, key=lambda m: m.created_at, reverse=True)[:5]
            return {
                "total_memories": total,
                "active_memories": active,
                "archived_memories": archived,
                "deleted_memories": deleted,
                "memories_by_type": dict(by_type),
                "memories_by_user": dict(by_user),
                "average_confidence": avg_confidence,
                "average_importance": avg_importance,
                "most_accessed": [m.to_dict() for m in most_accessed],
                "recently_created": [m.to_dict() for m in recently_created],
                "storage": {
                    "vector_count": self.vector_store.count(),
                },
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        with self._lock:
            self.vector_store.persist()
            logger.info("Memory state saved")

    def close(self) -> None:
        with self._lock:
            self.save()
            self.storage.close()
            self.vector_store.close()
            logger.info("EnterpriseLongTermMemory closed")

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def async_add_memory(self, *args: Any, **kwargs: Any) -> Memory:
        return await asyncio.to_thread(self.add_memory, *args, **kwargs)

    async def async_search_memories(self, *args: Any, **kwargs: Any) -> list[MemorySearchResult]:
        return await asyncio.to_thread(self.search_memories, *args, **kwargs)

    async def async_update_memory(self, *args: Any, **kwargs: Any) -> Memory:
        return await asyncio.to_thread(self.update_memory, *args, **kwargs)

    async def async_delete_memory(self, *args: Any, **kwargs: Any) -> None:
        return await asyncio.to_thread(self.delete_memory, *args, **kwargs)

    async def async_consolidate_memories(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.consolidate_memories, *args, **kwargs)

    async def async_forget_memories(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.forget_memories, *args, **kwargs)

    async def async_build_context(self, *args: Any, **kwargs: Any) -> str:
        
        return await asyncio.to_thread(self.build_context, *args, **kwargs)

        