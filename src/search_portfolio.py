"""
Layer 2: Diverse Search - Multi-Intent Search Portfolio.

Generates confirming, adversarial, contextual, and consensus searches
using Tavily API.
"""
from typing import Dict, Any, List, Optional
from tavily import TavilyClient


class SearchPortfolio:
    """Multi-intent search strategy using Tavily API."""
    
    def __init__(self, api_key: str, test_mode: bool = False):
        """Initialize Tavily client.
        
        Args:
            api_key: Tavily API key
            test_mode: If True, only use confirming + adversarial with 1 query each
        """
        self.client = TavilyClient(api_key=api_key)
        self.test_mode = test_mode
    
    def generate_search_queries(self, pillar: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate multi-intent search queries for a narrative pillar.
        
        Args:
            pillar: Narrative pillar with 'text' and 'entities'
            
        Returns:
            Dict with keys: confirming, adversarial, contextual, consensus
        """
        pillar_text = pillar.get("text", "")
        entities = pillar.get("entities", [])
        
        # Extract main entity/claim for query generation
        main_claim = pillar_text[:200]  # First 200 chars as claim
        
        # In test mode, only use confirming and adversarial with 1 query each
        if self.test_mode:
            queries = {
                "confirming": [f"{main_claim}"],
                "adversarial": [f"{main_claim} debunked"]
            }
        else:
            queries = {
                "confirming": [
                    f"{main_claim}",
                    f"{main_claim} evidence",
                    f"{main_claim} verification"
                ],
                "adversarial": [
                    f"{main_claim} debunked",
                    f"{main_claim} false",
                    f"{main_claim} criticism"
                ],
                "contextual": [
                    f"{main_claim} context",
                    f"{main_claim} background",
                    f"{main_claim} history"
                ],
                "consensus": [
                    f"site:reuters.com {main_claim}",
                    f"site:apnews.com {main_claim}",
                    f"site:bbc.com {main_claim}"
                ]
            }
        
        return queries
    
    def execute_search(
        self,
        query: str,
        search_type: str = "confirming",
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute a single search query via Tavily.
        
        Args:
            query: Search query string
            search_type: Type of search (for metadata)
            max_results: Maximum number of results
            
        Returns:
            List of search results with cleaned content
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_raw_content=False
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "search_type": search_type,
                    "query": query
                })
            
            return results
            
        except Exception as e:
            print(f"Search failed for '{query}': {e}")
            return []
    
    def execute_portfolio(self, pillar: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full multi-intent search portfolio for a pillar.
        
        Args:
            pillar: Narrative pillar dict
            
        Returns:
            Dict with all search results organized by intent
        """
        queries = self.generate_search_queries(pillar)
        
        portfolio = {
            "pillar": pillar.get("text", "")[:100],
            "searches": {}
        }
        
        for intent, query_list in queries.items():
            portfolio["searches"][intent] = []
            
            for query in query_list:
                results = self.execute_search(query, search_type=intent, max_results=3)
                portfolio["searches"][intent].extend(results)
        
        # Calculate total results
        portfolio["total_results"] = sum(
            len(results) for results in portfolio["searches"].values()
        )
        
        return portfolio
    
    def search_all_pillars(self, pillars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute search portfolios for all narrative pillars.
        
        Args:
            pillars: List of narrative pillars
            
        Returns:
            List of search portfolios (one per pillar)
        """
        portfolios = []
        
        for i, pillar in enumerate(pillars):
            print(f"Searching pillar {i+1}/{len(pillars)}...")
            portfolio = self.execute_portfolio(pillar)
            portfolios.append(portfolio)
        
        return portfolios
