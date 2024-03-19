from pymilvus.model.base import BaseReranker
from typing import List
import cohere

class CohereRerankerFunction(BaseRerankerFunction):
    def __init__(model_name: str = "rerank-english-v2.0", api_key: str)
        self.model_name = model_name
        self.client = co = cohere.Client(api_key)

    def __call__(self, query: str, documents: List[str], top_k=5)
        results = co.rerank(query=query, documents=documents, top_n=3, model="rerank-english-v2.0")
        return results

