import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import torch
    # Optimize PyTorch CPU thread count for fast parallel encoding
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers and torch are required. Install with: py -m pip install sentence-transformers torch")

from retrieval.act_matcher import matches_act_filter


class BGEM3SemanticRetriever:
    """
    Reusable BGE-M3 Multilingual Semantic Retriever for VidhiSetu.
    """

    def __init__(
        self,
        chunks_path: str = "data/processed/bns_chunks.json",
        embeddings_path: str = "data/processed/bns_embeddings.npy",
        model_name: str = "BAAI/bge-m3"
    ):
        self.chunks_path = chunks_path
        self.embeddings_path = embeddings_path
        self.model_name = model_name

        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[SentenceTransformer] = None
        self.is_initialized: bool = False

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            # Optimize sequence length for high throughput on CPU
            self.model.max_seq_length = 256

    def load_chunks(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks file not found at: {self.chunks_path}")

        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        return self.chunks

    def build_embeddings(self, force_rebuild: bool = False) -> np.ndarray:
        if not self.chunks:
            self.load_chunks()

        cache_meta_path = self.embeddings_path.replace(".npy", "_meta.json")
        loaded_emb = None
        cached_meta = {}

        if not force_rebuild and os.path.exists(self.embeddings_path) and os.path.exists(cache_meta_path):
            try:
                loaded_emb = np.load(self.embeddings_path)
                with open(cache_meta_path, 'r', encoding='utf-8') as f:
                    cached_meta = json.load(f)
            except Exception as e:
                print(f"[SemanticRetriever] Warning: Could not load cache meta ({e}).")
                loaded_emb = None
                cached_meta = {}

        # Compute current content hashes for all chunks
        import hashlib
        current_hashes = [
            hashlib.sha256(
                f"{c.get('chunk_id')}::{c.get('chapter', '')}::{c.get('section_title', '')}::{c.get('text', '')}".encode('utf-8')
            ).hexdigest()
            for c in self.chunks
        ]

        # Fast path: exactly identical chunks and existing embeddings
        if (
            loaded_emb is not None
            and loaded_emb.shape[0] == len(self.chunks)
            and len(cached_meta) == len(self.chunks)
            and all(cached_meta[i].get("hash") == current_hashes[i] for i in range(len(self.chunks)))
        ):
            self.embeddings = loaded_emb.astype(np.float32)
            self.is_initialized = True
            return self.embeddings

        # Incremental caching: identify which chunks are unchanged vs new/modified
        new_embeddings = np.zeros((len(self.chunks), 1024), dtype=np.float32)
        reused_count = 0
        missing_indices = []

        if loaded_emb is not None and cached_meta:
            chunk_to_prev = {
                m["chunk_id"]: (idx, m.get("hash"))
                for idx, m in enumerate(cached_meta)
                if idx < loaded_emb.shape[0]
            }
            for i, c in enumerate(self.chunks):
                cid = c.get("chunk_id")
                curr_h = current_hashes[i]
                if cid in chunk_to_prev and chunk_to_prev[cid][1] == curr_h:
                    prev_idx = chunk_to_prev[cid][0]
                    new_embeddings[i] = loaded_emb[prev_idx]
                    reused_count += 1
                else:
                    missing_indices.append(i)
        else:
            missing_indices = list(range(len(self.chunks)))

        print(f"[SemanticRetriever] Preserved {reused_count} cached embeddings. Encoding {len(missing_indices)} new/modified chunks...")

        if missing_indices:
            self._load_model()
            texts_to_encode = [
                f"{self.chunks[i].get('chapter', '')} {self.chunks[i].get('section_title', '')}\n{self.chunks[i].get('text', '')}"
                for i in missing_indices
            ]
            encoded_vecs = self.model.encode(
                texts_to_encode,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype(np.float32)

            for slot, enc_idx in enumerate(missing_indices):
                new_embeddings[enc_idx] = encoded_vecs[slot]

        self.embeddings = new_embeddings

        # Save embeddings and metadata cache
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        new_meta = [
            {"chunk_id": self.chunks[i].get("chunk_id"), "hash": current_hashes[i]}
            for i in range(len(self.chunks))
        ]
        with open(cache_meta_path, 'w', encoding='utf-8') as f:
            json.dump(new_meta, f, indent=2)

        print(f"[SemanticRetriever] Saved embeddings (Shape: {self.embeddings.shape}) and metadata cache.")
        self.is_initialized = True
        return self.embeddings

    def search(self, query: str, top_k: int = 5, act_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Encodes query using BGE-M3, calculates cosine similarity against chunk embeddings,
        ranks results by similarity descending, and returns structured metadata.
        Optionally filters candidates by act_filter.
        """
        if not self.is_initialized or self.embeddings is None:
            self.build_embeddings()

        if not query or not query.strip():
            return []

        self._load_model()

        # Encode query with unit normalization
        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        q_emb = q_emb.astype(np.float32)

        # Cosine similarity (dot product of normalized vectors)
        scores = np.dot(self.embeddings, q_emb)

        if act_filter:
            ranked_indices = [
                idx for idx in np.argsort(scores)[::-1]
                if matches_act_filter(
                    act_filter,
                    self.chunks[idx].get("act", ""),
                    self.chunks[idx].get("document_id", ""),
                    self.chunks[idx].get("source", ""),
                    self.chunks[idx].get("chunk_id", "")
                )
            ][:top_k]
        else:
            ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "document_id": chunk.get("document_id", ""),
                "act": chunk.get("act", ""),
                "section_number": chunk.get("section_number", 0),
                "section_title": chunk.get("section_title", ""),
                "chapter": chunk.get("chapter", ""),
                "page_start": chunk.get("page_start", 0),
                "page_end": chunk.get("page_end", 0),
                "source": chunk.get("source", ""),
                "score": round(score, 4),
                "text": chunk.get("text", "")
            })

        return results


if __name__ == "__main__":
    retriever = BGEM3SemanticRetriever()
    emb = retriever.build_embeddings()
    print(f"[SemanticRetriever] Embeddings ready. Shape: {emb.shape}")
