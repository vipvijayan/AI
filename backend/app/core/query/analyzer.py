"""Query analysis and preprocessing"""

from typing import List
from dataclasses import dataclass
import re


@dataclass
class QueryAnalysis:
    """Result of query analysis"""
    original: str
    cleaned: str
    intent: str
    keywords: List[str]
    entities: List[str]
    length: int
    is_question: bool


class QueryAnalyzer:
    """Analyze queries for better retrieval"""

    def __init__(self):
        self.entity_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "url": r"https?://[^\s]+",
            "number": r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
            "date": r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        }

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze query and extract components"""
        cleaned = self._clean_text(query)
        keywords = self._extract_keywords(cleaned)
        entities = self._extract_entities(query)
        intent = self._detect_intent(cleaned)
        is_question = query.rstrip().endswith("?")

        return QueryAnalysis(
            original=query,
            cleaned=cleaned,
            intent=intent,
            keywords=keywords,
            entities=entities,
            length=len(cleaned.split()),
            is_question=is_question,
        )

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\-\?]', '', text)
        text = text.lower()
        return text

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        tokens = text.split()
        keywords = [t for t in tokens if len(t) > 3 and not t.startswith('-')]
        # Remove punctuation from keywords
        keywords = [re.sub(r'[^\w\-]', '', k) for k in keywords]
        return list(set(keywords))[:10]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities"""
        entities = []

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            entities.extend(matches)

        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)

        # Also add lowercase versions for case-insensitive matching
        entities = list(set(entities + [e.lower() for e in entities]))
        return entities

    def _detect_intent(self, query: str) -> str:
        """Detect query intent"""
        query_lower = query.lower()

        if any(q in query_lower for q in ["what", "definition", "meaning"]):
            return "definition"
        elif any(q in query_lower for q in ["how", "steps", "guide"]):
            return "how-to"
        elif any(q in query_lower for q in ["where", "location", "find"]):
            return "location"
        elif any(q in query_lower for q in ["when", "time", "date"]):
            return "temporal"
        elif any(q in query_lower for q in ["why", "reason", "cause"]):
            return "reason"
        else:
            return "informational"

    def suggest_strategy(self, query: str) -> str:
        """Suggest retrieval strategy for query"""
        analysis = self.analyze(query)

        if analysis.length < 3:
            return "expand"
        elif analysis.is_question and len(analysis.keywords) < 2:
            return "expand"
        elif len(analysis.entities) > 0:
            return "hybrid"
        else:
            return "semantic"