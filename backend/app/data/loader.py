"""Data loader for RAG datasets"""

from typing import Dict, Optional
from datasets import load_dataset
from app.core.models import Document


class WikipediaDataLoader:
    """Load and prepare mini-wikipedia RAG dataset"""

    def __init__(self, max_documents: int = 1000):
        self.max_documents = max_documents
        self.corpus_dataset = None
        self.qa_dataset = None

    def load(self) -> Dict[str, Document]:
        """Load and process the dataset"""
        print("Loading mini-wikipedia RAG dataset...")

        self.corpus_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
        self.qa_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

        documents = self._process_corpus()

        print(f"Loaded {len(documents)} documents")
        return documents

    def _process_corpus(self) -> Dict[str, Document]:
        """Process corpus dataset into Document objects"""
        documents = {}

        if "test" in self.corpus_dataset:
            data = self.corpus_dataset["test"]
        elif "train" in self.corpus_dataset:
            data = self.corpus_dataset["train"]
        else:
            data = list(self.corpus_dataset.values())[0]

        for i, item in enumerate(data):
            if i >= self.max_documents:
                break

            doc_id = item.get("id", f"doc_{i}")
            content = item.get("text", item.get("content", ""))
            title = item.get("title", "")

            metadata = {
                "title": title,
                "source": "mini-wikipedia",
                "index": i,
            }

            documents[doc_id] = Document(
                id=doc_id,
                content=content,
                metadata=metadata,
            )

        return documents

    def get_qa_pairs(self, limit: int = 100) -> list:
        """Get question-answer pairs for evaluation"""
        if not self.qa_dataset:
            self.qa_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

        if "test" in self.qa_dataset:
            data = self.qa_dataset["test"]
        elif "train" in self.qa_dataset:
            data = self.qa_dataset["train"]
        else:
            data = list(self.qa_dataset.values())[0]

        qa_pairs = []
        for i, item in enumerate(data):
            if i >= limit:
                break
            qa_pairs.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "id": item.get("id", f"qa_{i}"),
            })

        return qa_pairs

    def print_sample(self):
        """Print sample from dataset"""
        if not self.corpus_dataset:
            self.load()

        print("--- Sample Corpus Chunk ---")
        if "test" in self.corpus_dataset and len(self.corpus_dataset["test"]) > 0:
            print(self.corpus_dataset["test"][0])

        print("\n--- Sample Evaluation Q&A ---")
        if self.qa_dataset and "test" in self.qa_dataset and len(self.qa_dataset["test"]) > 0:
            print(self.qa_dataset["test"][0])