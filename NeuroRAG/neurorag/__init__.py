"""NeuroRAG - citation-grounded retrieval-augmented generation over
Alzheimer's disease and neuroimaging literature.

Typical use::

    from neurorag.config import Config
    from neurorag.pipeline import RAGPipeline

    pipeline = RAGPipeline.build(Config.load())   # or RAGPipeline.load()
    answer = pipeline.ask("Why does slice-level cross-validation inflate accuracy?")
    print(answer.text)
    for citation in answer.citations:
        print(citation.as_reference())
"""

from .config import Config
from .pipeline import RAGPipeline
from .schema import Answer, Chunk, Citation, Document, Hit

__version__ = "1.0.0"
__all__ = ["Config", "RAGPipeline", "Answer", "Chunk", "Citation", "Document", "Hit", "__version__"]
