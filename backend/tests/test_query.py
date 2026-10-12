"""Tests for query analyzer and optimizer"""

import pytest
from app.core.query.analyzer import QueryAnalyzer
from app.core.query.optimizer import QueryOptimizer


class TestQueryAnalyzer:
    def test_analyze_simple_query(self):
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("What is machine learning?")

        assert analysis.original == "What is machine learning?"
        assert analysis.intent == "definition"
        assert analysis.is_question is True
        assert "machine" in analysis.keywords
        assert "learning" in analysis.keywords

    def test_analyze_how_to_query(self):
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("How to train a neural network?")

        assert analysis.intent == "how-to"
        assert analysis.is_question is True

    def test_analyze_with_entities(self):
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("When was Google founded?")

        assert "google" in analysis.entities
        assert analysis.intent == "temporal"

    def test_suggest_strategy_short(self):
        analyzer = QueryAnalyzer()
        strategy = analyzer.suggest_strategy("AI")
        assert strategy == "expand"

    def test_suggest_strategy_with_entities(self):
        analyzer = QueryAnalyzer()
        strategy = analyzer.suggest_strategy("When was Microsoft founded?")
        assert strategy == "hybrid"


class TestQueryOptimizer:
    def test_remove_stop_words(self):
        optimizer = QueryOptimizer()
        result = optimizer.remove_stop_words("what is the meaning of life")
        assert "what" not in result
        assert "is" not in result
        assert "the" not in result
        assert "of" not in result
        assert "meaning" in result
        assert "life" in result

    def test_expand_query(self):
        optimizer = QueryOptimizer()
        expanded = optimizer.expand_query("what is big data")
        assert len(expanded) > 1
        assert "what is big data" in expanded

    def test_rewrite_query(self):
        optimizer = QueryOptimizer()
        rewritten = optimizer.rewrite_query("  What is AI?  ")
        assert rewritten == "What is AI"

    def test_optimize_strategy_expand(self):
        optimizer = QueryOptimizer()
        optimized = optimizer.optimize("AI", strategy="expand")
        assert len(optimized) > 1

    def test_optimize_strategy_rewrite(self):
        optimizer = QueryOptimizer()
        optimized = optimizer.optimize("  What is ML?  ", strategy="rewrite")
        assert len(optimized) == 1
        assert optimized[0] == "What is ML"

    def test_detect_intent(self):
        optimizer = QueryOptimizer()
        assert optimizer.detect_intent("What is Python?") == "factual"
        assert optimizer.detect_intent("How to code?") == "how-to"
        assert optimizer.detect_intent("Python vs Java") == "comparison"
        assert optimizer.detect_intent("What do you think?") == "opinion"