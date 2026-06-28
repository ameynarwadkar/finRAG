import hashlib
import numpy as np
from datetime import datetime, timedelta

def cosine_similarity(v1, v2):
    if not isinstance(v1, np.ndarray): v1 = np.array(v1)
    if not isinstance(v2, np.ndarray): v2 = np.array(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 0.0
    return np.dot(v1, v2) / norm

class SemanticResponseCache:
    def __init__(self, threshold=0.92, ttl_seconds=3600):
        self._exact_cache = {}
        self._semantic_cache = []
        self.threshold = threshold
        self.ttl = timedelta(seconds=ttl_seconds)
        
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
        
    def get_exact(self, prompt: str):
        return self._exact_cache.get(self._hash(prompt))
        
    def set_exact(self, prompt: str, response):
        self._exact_cache[self._hash(prompt)] = response
        
    def get_semantic(self, embedding: list):
        now = datetime.now()
        # Clean up expired items
        self._semantic_cache = [c for c in self._semantic_cache if c["expires_at"] > now]
        
        best_sim = -1
        best_resp = None
        for item in self._semantic_cache:
            sim = cosine_similarity(embedding, item["embedding"])
            if sim > best_sim and sim >= self.threshold:
                best_sim = sim
                best_resp = item["response"]
                
        return best_resp
        
    def set_semantic(self, embedding: list, response):
        self._semantic_cache.append({
            "embedding": embedding,
            "response": response,
            "expires_at": datetime.now() + self.ttl
        })

# Global caches
refiner_cache = SemanticResponseCache()
generator_cache = SemanticResponseCache()
