from typing import List

from FlagEmbedding import FlagReranker

from pymilvus.model.base import BaseRerankerFunction


class FlagRerankerFunction(BaseRerankerFunction):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        batch_size: int = 32,
        normalize: bool = True,
        device: str = ""
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16, device=device)

    def _batchify(self, texts: List[str], batch_size: int) -> List[List[str]]:
        return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def __call__(self, query: str, documents: List[str], top_k: int = 5) -> List[float]:
        query_document_pairs = [[query, doc] for doc in documents]
        batched_texts = self._batchify(documents, self.batch_size)
        scores = []
        for batched_text in batched_texts:
            query_document_pairs = [[query, text] for text in batched_text]
            batch_score = self.reranker.compute_score(
                query_document_pairs, normalize=self.normalize
            )
            scores.extend(batch_score)
        ranked_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        if top_k:
            ranked_order = ranked_order[:top_k]

        results = []
        for index in ranked_order:
            results.append({"text": documents[index], "score": scores[index], "index": index})
        return results
