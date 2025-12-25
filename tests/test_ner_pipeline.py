"""Unit tests for NER pipeline."""
import pytest
from src.ner_pipeline import NERPipeline


def test_ner_pipeline_initialization():
    """Test NER pipeline initializes without loading models."""
    pipeline = NERPipeline()
    assert pipeline.fast_nlp is None
    assert pipeline.precision_nlp is None


def test_stage_one_contextual():
    """Test stage 1 contextual NER."""
    pipeline = NERPipeline()
    
    text = "Apple Inc. CEO Tim Cook announced new products on January 15, 2025 in San Francisco."
    result = pipeline.stage_one_contextual(text)
    
    assert "entities" in result
    assert "entity_map" in result
    assert result["entity_count"] > 0
    
    # Should detect PERSON, ORG, GPE, DATE
    assert any(e["type"] == "ORG" for e in result["entities"])
    assert any(e["type"] == "PERSON" for e in result["entities"])


# Additional tests to be implemented as functionality is built
