from __future__ import annotations
import hashlib
from logging import exception
import os, re
from typing import Optional
from functools import lru_cache

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), 'knowledge_base.txt')
#KNOWLEDGE_BASE_DIR = os.getenv('KNOWLEDGE_BASE_DIR', os.path.join(os.getcwd(), 'knowledge_base.txt'))
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', os.path.join(os.getcwd(), 'chroma_persist'))
COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "accuentry")
COLLECTION_NAME = f"{COLLECTION_PREFIX}_rag_collection"
EMBED_MODEL_NAME    = os.getenv("CHROMA_EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = 5


TOP_K = 5 #you can define it in env variable as well

def _load_chunks(path : str = KNOWLEDGE_BASE_DIR) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split by double newlines and strip whitespace
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    return chunks


def _lexical_retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    chunks = _load_chunks()
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_tokens:
        return chunks[:top_k]

    scored = sorted(
        chunks,
        key=lambda chunk: (
            len(query_tokens & set(re.findall(r"[a-z0-9]+", chunk.lower()))),
            len(chunk),
        ),
        reverse=True,
    )
    return scored[:top_k]

def _chunk_id(text: str) -> str:
    # Create a unique ID based on the text content (you can use hashing or a simple counter)
    return hashlib.md5(text.encode('utf-8')).hexdigest()  # Simple example: use MD5 hash

#embedding based retrieval
@lru_cache(maxsize=1)
def _get_collection():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        client = chromadb.PersistentClient(path = CHROMA_PERSIST_DIR)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef, metadata={"hnsw:space": "cosine"})

        chunks = _load_chunks()
        ids = [_chunk_id(c) for c in chunks]

        existing_ids = set(collection.get(ids=ids)['ids'])
        new_chunks = [(id, c) for id, c in zip(ids, chunks) if id not in existing_ids]

        if new_chunks:
            texts, new_ids= zip(*new_chunks)
            collection.add(documents=list(texts), ids=list(new_ids))    
        return collection

    except Exception as e:
        print(f"Error initializing ChromaDB collection: {e}")
        raise RuntimeError("Failed to initialize ChromaDB collection. Please check the logs for details.")
    

#api
def retrieve(query: str, top_k: int = TOP_K) -> list[str]:
    try:
        collection = _get_collection()
        if collection is not None:
            try:
                results = collection.query(query_texts=[query], n_results=top_k)
                documents = results.get('documents') if isinstance(results, dict) else None
                if documents and documents[0]:
                    return [chunk for chunk in documents[0] if chunk]
            except Exception as e:
                print(f"Error retrieving relevant chunks from ChromaDB: {e}")
    except Exception as e:
        print(f"Error initializing ChromaDB collection: {e}")

    return _lexical_retrieve(query, top_k=top_k)



def retrieve_as_context(query: str, top_k: int = TOP_K) -> str:
    """Returns chunks joined as a single context block for prompt injection."""
    chunks = retrieve(query, top_k)
    if not chunks:
        return ""
    return "\n\n---\n\n".join(chunks)


def reindex(path: str = KNOWLEDGE_BASE_DIR) -> int:
    """
    Force a full re-index (call this when knowledge_base.txt changes).
    Returns count of newly added chunks.
    Returns -1 if ChromaDB is unavailable.
    """
    _get_collection.cache_clear()

    collection = _get_collection()
    if collection is None:
        return -1

    chunks = _load_chunks(path)
    ids    = [_chunk_id(c) for c in chunks]

    existing_ids = set(collection.get(ids=ids)["ids"])
    new_chunks   = [(c, i) for c, i in zip(chunks, ids) if i not in existing_ids]

    if new_chunks:
        texts, new_ids = zip(*new_chunks)
        collection.add(documents=list(texts), ids=list(new_ids))

    return len(new_chunks)