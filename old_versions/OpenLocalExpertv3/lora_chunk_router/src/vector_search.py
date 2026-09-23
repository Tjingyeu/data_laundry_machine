"""
FAISS Vector Search baseline for chunk retrieval.

Uses BGE-small embeddings with FAISS IndexFlatIP for similarity search.
"""

import os
import sys
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, load_json, get_project_root, ensure_dir


class VectorSearchIndex:
    """
    FAISS-based vector search index for chunk retrieval.

    Uses BGE-small embeddings with Inner Product (cosine similarity) search.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_dim: int = 384,
        index_type: str = "IndexFlatIP",
        index_path: Optional[str] = None,
    ):
        """
        Initialize vector search index.

        Args:
            embedding_model: Sentence transformer model name
            embedding_dim: Embedding dimension
            index_type: FAISS index type (only IndexFlatIP supported)
            index_path: Path to save/load index
        """
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index_path = index_path

        # Will be initialized later
        self.model = None
        self.index = None
        self.chunks = None
        self.id_to_chunk = None

    def _init_model(self):
        """Initialize the embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)

    def _init_index(self):
        """Initialize FAISS index."""
        if self.index is None:
            print(f"Creating FAISS index: {self.index_type} (dim={self.embedding_dim})")
            self.index = faiss.IndexFlatIP(self.embedding_dim)

    def build_index(
        self,
        chunks: List[Dict],
        batch_size: int = 32,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of chunk dictionaries with "chunk_id" and "text"
            batch_size: Batch size for encoding
            save_path: Path to save index and chunks (optional)
        """
        self._init_model()
        self._init_index()

        self.chunks = chunks
        self.id_to_chunk = {c["chunk_id"]: c for c in chunks}

        # Extract texts
        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        print(f"Encoding {len(texts)} chunks...")

        # Encode all texts
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # For cosine similarity via inner product
        )

        # Convert to float32
        embeddings = embeddings.astype(np.float32)

        # Add to index
        self.index.add(embeddings)

        print(f"Index built with {self.index.ntotal} vectors")

        # Save if path provided
        if save_path:
            self.save(save_path)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, any]]:
        """
        Search for top-K relevant chunks.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            [{"id": str, "score": float, "text": str}, ...]
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index not built. Call build_index() first.")

        self._init_model()

        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_embedding, k=min(top_k, self.index.ntotal))

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            results.append({
                "id": chunk["chunk_id"],
                "score": float(score),
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
            })

        return results

    def save(self, path: str) -> None:
        """
        Save index and metadata to disk.

        Args:
            path: Directory to save to
        """
        ensure_dir(path)

        # Save FAISS index
        index_path = os.path.join(path, "faiss_index.bin")
        print(f"Saving FAISS index to: {index_path}")
        faiss.write_index(self.index, index_path)

        # Save chunks
        chunks_path = os.path.join(path, "chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)

        # Save config
        config_path = os.path.join(path, "index_config.json")
        config = {
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "num_chunks": len(self.chunks) if self.chunks else 0,
        }
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Index saved to {path}")

    def load(self, path: str) -> None:
        """
        Load index and metadata from disk.

        Args:
            path: Directory to load from
        """
        # Load FAISS index
        index_path = os.path.join(path, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        print(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(index_path)

        # Load chunks
        chunks_path = os.path.join(path, "chunks.pkl")
        if os.path.exists(chunks_path):
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            self.id_to_chunk = {c["chunk_id"]: c for c in self.chunks}

        # Load config
        config_path = os.path.join(path, "index_config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.embedding_model_name = config.get("embedding_model", self.embedding_model_name)
            self.embedding_dim = config.get("embedding_dim", self.embedding_dim)
            self.index_type = config.get("index_type", self.index_type)

        # Initialize model
        self._init_model()

        print(f"Index loaded: {self.index.ntotal} vectors")


def build_index_from_chunks(
    chunks_path: str,
    output_path: str,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
) -> VectorSearchIndex:
    """
    Build vector search index from chunks file.

    Args:
        chunks_path: Path to chunks.json
        output_path: Path to save index
        embedding_model: Embedding model name
        batch_size: Batch size for encoding

    Returns:
        VectorSearchIndex instance
    """
    print(f"Loading chunks from: {chunks_path}")
    chunks = load_json(chunks_path)
    print(f"Loaded {len(chunks)} chunks")

    # Create and build index
    index = VectorSearchIndex(
        embedding_model=embedding_model,
        index_path=output_path,
    )

    index.build_index(
        chunks=chunks,
        batch_size=batch_size,
        save_path=output_path,
    )

    return index


def main():
    """CLI for vector search operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Vector Search")
    parser.add_argument("--mode", choices=["build", "search"], default="search",
                        help="Mode: build index or search")
    parser.add_argument("--query", type=str, help="Query to search (for search mode)")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Path to chunks.json (for build mode)")
    parser.add_argument("--index_dir", type=str, default=None,
                        help="Index directory path")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K results")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5",
                        help="Embedding model")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    # Get paths
    root = get_project_root()
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = root / "configs" / "config.yaml"
    config = load_config(str(config_path))

    if args.mode == "build":
        chunks_path = args.chunks if args.chunks else str(root / config["data"]["chunks_output"])
        index_dir = args.index_dir if args.index_dir else str(root / "models" / "embeddings")

        build_index_from_chunks(
            chunks_path=chunks_path,
            output_path=index_dir,
            embedding_model=args.embedding_model,
            batch_size=config["vector_search"]["batch_size"],
        )
        print("Index built successfully!")

    elif args.mode == "search":
        index_dir = args.index_dir if args.index_dir else str(root / "models" / "embeddings")

        # Load or build index
        if os.path.exists(os.path.join(index_dir, "faiss_index.bin")):
            print(f"Loading index from: {index_dir}")
            index = VectorSearchIndex(index_path=index_dir)
            index.load(index_dir)
        else:
            print("Index not found. Building new index...")
            chunks_path = str(root / config["data"]["chunks_output"])
            index = build_index_from_chunks(
                chunks_path=chunks_path,
                output_path=index_dir,
                embedding_model=args.embedding_model,
            )

        if args.query:
            results = index.search(query=args.query, top_k=args.top_k)
            print(f"\nQuery: {args.query}")
            print(f"Top-{args.top_k} Results:")
            for r in results:
                print(f"  {r['id']}: {r['score']:.4f}")
                print(f"    {r['text'][:100]}...")
        else:
            # Interactive mode
            print("Vector Search - Enter a query (or 'quit' to exit)")

            while True:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break

                results = index.search(query=query, top_k=args.top_k)
                print(f"Top-{args.top_k} Results:")
                for r in results:
                    print(f"  {r['id']}: {r['score']:.4f}")


if __name__ == "__main__":
    main()
