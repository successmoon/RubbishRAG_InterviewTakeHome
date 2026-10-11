# Decorators guide — the mechanism (not the solution)

Each pipeline stage is a function registered with a decorator.
To change a stage's behavior, define a function with the same decorator in your code.
It automatically replaces the default:

```python
import rubbish_rag as R

@R.chunker
def chunk(docs):
    # Input: [{"doc_id","Question","Category","BriefAnswer","Answer","Keyword"}]
    # Output: [{"doc_id","chunk_id","text","category"}]
    # Default: 300-char fixed slices of Answer

@R.retriever
def retrieve(query, topk=5):
    # Input: raw query. Output: list of hits with bm25/dense/fused scores.
    # Default: forwards the query to the remote API unchanged.

@R.reranker
def rerank(query, hits):
    # Sorts hits. Default: shared-word count.

@R.resolver
def resolve(query, hits):
    # Must return answer or clarify contract (see README.md or docs/index.html).
    # Default: always answer.
```

Which stage to override? Your choice. The rule:
- The remote API is not yours to change.
- Normalization, query rewriting, filtering, re-ranking, and the final decision are all on your side.
- Every change must be justified with logged traces and plots.