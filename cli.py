#!/usr/bin/env python3
"""
News Audit Engine - Command Line Interface

Usage:
    python cli.py "paste article text here..."
    python cli.py --file article.txt
"""
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.gemini_client import GeminiClient
from src.ner_pipeline import NERPipeline
from src.search_portfolio import SearchPortfolio
from src.semantic_judge import SemanticJudge
from src.consensus_protocol import ConsensusProtocol
from src.utils import ProgressTracker, get_api_key


def extract_pillars(client: GeminiClient, text: str) -> list:
    """
    Extract narrative pillars from article text using Gemini.
    
    Args:
        client: GeminiClient instance
        text: Full article text
        
    Returns:
        List of narrative pillar dicts
    """
    prompt = (
        "Analyze this news article and extract 3-5 NARRATIVE PILLARS - the core causal "
        "arguments that form the backbone of the story. Each pillar should be a complete "
        "claim with subject, action, and consequence.\n\n"
        f"Article:\n{text[:8000]}\n\n"
        "Return JSON with structure: "
        '{"pillars": [{"text": "pillar statement", "importance": 1-5}]}'
    )
    
    response = client.generate_json(
        prompt=prompt,
        timeout=60,
        temperature=0.7
    )
    
    if response.get("ok"):
        return response["data"].get("pillars", [])
    else:
        print(f"Pillar extraction failed: {response.get('error')}")
        return []


def main():
    """Main CLI entry point."""
    tracker = ProgressTracker(verbose=True)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python cli.py <text> or python cli.py --file <path> [--test-mode]")
        sys.exit(1)
    
    # Check for test mode flag
    test_mode = "--test-mode" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--test-mode"]
    
    # Get article text
    if len(args) == 0:
        print("Error: No article text or file provided")
        sys.exit(1)
    
    if args[0] == "--file":
        if len(args) < 2:
            print("Error: --file requires a path argument")
            sys.exit(1)
        file_path = Path(args[1])
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        article_text = file_path.read_text()
    else:
        article_text = " ".join(args)
    
    print(f"\n{'='*60}")
    print("NEWS AUDIT ENGINE v4.0")
    if test_mode:
        print("[TEST MODE: 1 pillar, 2 queries]")
    print(f"{'='*60}")
    print(f"Article length: {len(article_text)} characters")
    
    try:
        # Initialize clients
        tracker.start_phase("Initialization")
        gemini_api_key = get_api_key("GEMINI_API_KEY")
        tavily_api_key = get_api_key("TAVILY_API_KEY")
        
        gemini = GeminiClient(api_key=gemini_api_key)
        ner = NERPipeline()
        search = SearchPortfolio(api_key=tavily_api_key, test_mode=test_mode)
        judge = SemanticJudge(gemini_client=gemini)
        consensus = ConsensusProtocol(gemini_client=gemini)
        tracker.complete_phase()
        
        # Layer 1: Extract pillars and entities
        tracker.start_phase("Layer 1: Decomposition")
        tracker.update("Extracting narrative pillars...")
        pillars = extract_pillars(gemini, article_text)
        print(f"  → Found {len(pillars)} narrative pillars")
        
        tracker.update("Running two-stage NER pipeline...")
        enriched_pillars = ner.extract_pillars_with_entities(article_text, pillars)
        
        # In test mode, only use the highest importance pillar
        if test_mode and enriched_pillars:
            highest_pillar = max(enriched_pillars, key=lambda p: p.get("importance", 0))
            print(f"  → Test mode: Using highest importance pillar (importance={highest_pillar.get('importance')})")
            enriched_pillars = [highest_pillar]
        
        total_entities = sum(len(p.get("entities", [])) for p in enriched_pillars)
        print(f"  → Extracted {total_entities} entities")
        tracker.complete_phase()
        
        # Layer 2: Search portfolio
        tracker.start_phase("Layer 2: Diverse Search")
        portfolios = search.search_all_pillars(enriched_pillars)
        total_results = sum(p.get("total_results", 0) for p in portfolios)
        print(f"  → Retrieved {total_results} search results")
        tracker.complete_phase()
        
        # Layer 3: Semantic analysis
        tracker.start_phase("Layer 3: Semantic Judge")
        tracker.update("Ingesting search results into Qdrant...")
        vector_count = judge.ingest_search_results(portfolios)
        print(f"  → Ingested {vector_count} vectors")
        
        tracker.update("Detecting semantic conflicts...")
        all_conflicts = []
        for pillar in enriched_pillars:
            conflicts = judge.detect_conflicts(pillar.get("text", ""))
            all_conflicts.extend(conflicts)
        print(f"  → Detected {len(all_conflicts)} conflicts")
        tracker.complete_phase()
        
        # Layer 4: Consensus protocol
        tracker.start_phase("Layer 4: Consensus Protocol")
        
        # Format evidence for agents
        evidence_summary = (
            f"NARRATIVE PILLARS:\n"
            + "\n".join([f"- {p.get('text', '')}" for p in enriched_pillars[:5]])
            + f"\n\nSEARCH RESULTS: {total_results} sources analyzed"
            + f"\n\nCONFLICTS DETECTED: {len(all_conflicts)}"
        )
        
        result = consensus.run_full_protocol(evidence_summary)
        tracker.complete_phase()
        
        # Print final verdict
        print(f"\n{'='*60}")
        print("FINAL VERDICT")
        print(f"{'='*60}")
        verdict = result["consensus"]
        print(f"Verdict: {verdict['verdict']}")
        print(f"Confidence: {verdict['confidence']:.1%}")
        print(f"Consensus: {verdict['consensus_percentage']:.0f}%")
        print(f"Reasoning: {verdict['reasoning']}")
        
        # Save detailed output
        output_path = Path("audit_result.json")
        output_data = {
            "article_length": len(article_text),
            "pillars": enriched_pillars,
            "search_results": total_results,
            "conflicts": all_conflicts,
            "consensus": result
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Detailed results saved to: {output_path}")
        
    except Exception as e:
        tracker.error(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
