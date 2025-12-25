"""
Layer 3: Synthesis & Conflict Detection - Semantic Judge.

Uses Qdrant vector database to detect semantic conflicts and
calculate omission scores.
"""
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.gemini_client import GeminiClient


class SemanticJudge:
    """Semantic conflict detection using Qdrant vector database."""
    
    def __init__(self, gemini_client: GeminiClient, collection_name: str = "audit_evidence"):
        """
        Initialize semantic judge with Qdrant and embeddings.
        
        Args:
            gemini_client: GeminiClient for generating embeddings
            collection_name: Name of Qdrant collection
        """
        self.gemini = gemini_client
        self.collection_name = collection_name
        
        # Initialize Qdrant in-memory mode (no Docker required for prototyping)
        self.qdrant = QdrantClient(":memory:")
        
        # Create collection for 768-dimensional vectors (Gemini text-embedding-004)
        self.qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    
    def ingest_search_results(self, portfolios: List[Dict[str, Any]]) -> int:
        """
        Ingest search results into Qdrant with embeddings.
        
        Args:
            portfolios: List of search portfolios from SearchPortfolio
            
        Returns:
            Number of vectors ingested
        """
        points = []
        point_id = 0
        
        for portfolio in portfolios:
            pillar_text = portfolio.get("pillar", "")
            
            for intent, results in portfolio.get("searches", {}).items():
                for result in results:
                    # Generate embedding for search result content
                    content = result.get("content", "")
                    if not content:
                        continue
                    
                    embedding_resp = self.gemini.generate_embedding(content)
                    if not embedding_resp.get("ok"):
                        print(f"Failed to embed: {embedding_resp.get('error')}")
                        continue
                    
                    # Create point with metadata
                    point = PointStruct(
                        id=point_id,
                        vector=embedding_resp["embedding"],
                        payload={
                            "pillar": pillar_text,
                            "content": content[:500],  # Truncate for storage
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "search_intent": intent,
                            "query": result.get("query", ""),
                            "score": result.get("score", 0.0)
                        }
                    )
                    points.append(point)
                    point_id += 1
        
        # Batch upsert
        if points:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        return len(points)
    
    def detect_conflicts(self, pillar_text: str, threshold: float = 0.85) -> List[Dict[str, Any]]:
        """
        Detect semantic conflicts for a narrative pillar.
        
        Args:
            pillar_text: Text of the narrative pillar
            threshold: Similarity threshold for conflict detection (0.0-1.0)
            
        Returns:
            List of detected conflicts with opposing evidence
        """
        # Generate embedding for pillar
        embedding_resp = self.gemini.generate_embedding(pillar_text)
        if not embedding_resp.get("ok"):
            return []
        
        # Search for similar content using query_points
        from qdrant_client.models import QueryRequest, ScoredPoint
        
        search_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=embedding_resp["embedding"],
            limit=20,
            score_threshold=threshold
        ).points
        
        # Group by search intent to find conflicts
        confirming = []
        adversarial = []
        
        for hit in search_results:
            intent = hit.payload.get("search_intent")
            if intent == "confirming":
                confirming.append(hit)
            elif intent == "adversarial":
                adversarial.append(hit)
        
        # If we have both confirming and adversarial with high similarity,
        # that indicates a conflict
        conflicts = []
        if confirming and adversarial:
            conflict = {
                "pillar": pillar_text[:100],
                "conflict_type": "opposing_evidence",
                "supporting_count": len(confirming),
                "opposing_count": len(adversarial),
                "supporting_sources": [h.payload.get("url") for h in confirming[:3]],
                "opposing_sources": [h.payload.get("url") for h in adversarial[:3]]
            }
            
            # Classify the conflict type
            classification = self.classify_conflict_type(pillar_text, conflict)
            conflict["classification"] = classification.get("classification", "source_disagreement")
            conflict["classification_reasoning"] = classification.get("reasoning", "")
            
            conflicts.append(conflict)
        
        return conflicts
    
    def calculate_omission_score(
        self,
        article_pillars: List[str],
        consensus_facts: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate omission score: consensus facts missing from article.
        
        Args:
            article_pillars: List of narrative pillar texts from article
            consensus_facts: List of consensus facts from authoritative sources
            
        Returns:
            Dict with omission score and missing facts
        """
        # Embed all article pillars
        article_embeddings = []
        for pillar in article_pillars:
            resp = self.gemini.generate_embedding(pillar)
            if resp.get("ok"):
                article_embeddings.append(resp["embedding"])
        
        if not article_embeddings:
            return {"omission_score": 0.0, "missing_facts": []}
        
        # Check which consensus facts are absent
        missing_facts = []
        
        for fact in consensus_facts:
            fact_resp = self.gemini.generate_embedding(fact)
            if not fact_resp.get("ok"):
                continue
            
            # Search article embeddings for this fact
            # (In production, we'd use Qdrant; for now, simple similarity check)
            # TODO: Implement proper vector similarity search
            missing_facts.append(fact)  # Placeholder
        
        omission_score = len(missing_facts) / len(consensus_facts) if consensus_facts else 0.0
        
        return {
            "omission_score": omission_score,
            "missing_facts": missing_facts,
            "total_consensus_facts": len(consensus_facts)
        }
    
    def classify_conflict_type(self, pillar_text: str, conflict: Dict[str, Any]) -> Dict[str, str]:
        """
        Classify a detected conflict into one of three types.
        
        Types:
        - factual_contradiction: Mutually exclusive objective facts
        - position_evolution: Entity changed stance over time (historical vs current)
        - source_disagreement: Competing claims about same timeframe
        
        Args:
            pillar_text: The narrative pillar being evaluated
            conflict: Conflict dict with supporting_count, opposing_count
            
        Returns:
            Dict with 'classification' and 'reasoning'
        """
        prompt = f"""Classify this narrative conflict.

NARRATIVE CLAIM:
{pillar_text}

EVIDENCE CONFLICT:
- {conflict['supporting_count']} sources support this claim
- {conflict['opposing_count']} sources contradict this claim

Classify as ONE of these types:

A) factual_contradiction
   - Mutually exclusive facts about the SAME event/timeframe
   - Example: "Treaty signed March 15" vs "Treaty signed March 20"
   - One source must be wrong

B) position_evolution  
   - Entity changed position/policy over time
   - Example: Historical stance X vs new/current stance Y
   - BOTH can be true if temporally separated
   - Key indicators: "previously supported", "now proposes", "shifted from"
   - Finding historical evidence that contradicts current claim actually VALIDATES a change story

C) source_disagreement
   - Different sources making different claims about SAME timeframe
   - Neither is clearly historical vs current
   - Requires evaluating source credibility

CRITICAL: If the claim describes a NEW or CHANGED position, and opposing evidence shows the OLD position, classify as B (position_evolution).

Return JSON: {{"classification": "factual_contradiction|position_evolution|source_disagreement", "reasoning": "2-3 sentence explanation"}}"""

        response = self.gemini.generate_json(
            prompt=prompt,
            timeout=30,
            temperature=0.1
        )
        
        if response.get("ok"):
            return response["data"]
        else:
            # Default to source_disagreement if classification fails
            return {
                "classification": "source_disagreement",
                "reasoning": f"Classification failed: {response.get('error')}"
            }
