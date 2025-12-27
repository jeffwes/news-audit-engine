"""
Layer 4: Consensus - Multi-Agent Stability Protocol.

Three agent personas debate the evidence and reach consensus
on narrative integrity verdict.
"""
from typing import Dict, Any, List
import json
import os
from src.gemini_client import GeminiClient


class ConsensusProtocol:
    """Multi-agent debate system for stable verdicts."""
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize consensus protocol with agent personas."""
        self.gemini = gemini_client
        
        # Load agent prompts from JSON file
        prompts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'agent_prompts.json')
        with open(prompts_path, 'r') as f:
            self.agents = json.load(f)
        # Load agent prompts from JSON file
        prompts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'agent_prompts.json')
        with open(prompts_path, 'r') as f:
            self.agents = json.load(f)
    
    def gather_agent_verdicts(self, evidence: str, current_date: str = None) -> List[Dict[str, Any]]:
        """
        Gather initial verdicts from all agents.
        
        Args:
            evidence: Formatted evidence string containing pillars, conflicts, searches
            current_date: Current date string for temporal context (e.g., "December 26, 2025")
            
        Returns:
            List of agent verdict dicts
        """
        from datetime import datetime
        
        # Use provided date or default to today
        if not current_date:
            current_date = datetime.now().strftime("%B %d, %Y")
        
        verdicts = []
        
        # Only iterate over the three main agents (skip executive)
        main_agents = {k: v for k, v in self.agents.items() if k != "executive"}
        
        for agent_id, agent_config in main_agents.items():
            prompt = agent_config["prompt_template"].format(
                evidence=evidence,
                current_date=current_date
            )
            
            response = self.gemini.generate_json(
                prompt=prompt,
                timeout=45,
                temperature=0.2
            )
            
            if response.get("ok"):
                verdict = response["data"]
                # Handle case where data might be a list
                if isinstance(verdict, list):
                    print(f"Warning: Agent {agent_id} returned list, using first item")
                    verdict = verdict[0] if verdict else {}
                verdict["agent"] = agent_config["name"]
                verdict["agent_id"] = agent_id
                verdicts.append(verdict)
            else:
                print(f"Agent {agent_id} failed: {response.get('error')}")
        
        return verdicts
    
    def conduct_debate(
        self,
        initial_verdicts: List[Dict[str, Any]],
        evidence: str,
        max_turns: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Conduct structured debate between agents.
        
        Args:
            initial_verdicts: Initial agent verdicts
            evidence: Evidence summary
            max_turns: Maximum debate turns
            
        Returns:
            Final agent verdicts after debate
        """
        current_verdicts = initial_verdicts
        
        for turn in range(max_turns):
            # Each agent reviews other agents' verdicts
            new_verdicts = []
            
            for agent_id, agent_config in self.agents.items():
                # Find this agent's current verdict
                current_verdict = next(
                    (v for v in current_verdicts if v.get("agent_id") == agent_id),
                    None
                )
                
                if not current_verdict:
                    continue
                
                # Show other agents' verdicts
                other_verdicts = [v for v in current_verdicts if v.get("agent_id") != agent_id]
                other_views = "\n".join([
                    f"{v.get('agent')}: {v.get('verdict')} (confidence: {v.get('confidence')}) - {v.get('reasoning')}"
                    for v in other_verdicts
                ])
                
                # Ask agent to reconsider
                debate_prompt = (
                    f"You are {agent_config['name']}. You previously assessed the evidence as: "
                    f"{current_verdict.get('verdict')} (confidence: {current_verdict.get('confidence')})\n\n"
                    f"Other agents have provided these assessments:\n{other_views}\n\n"
                    f"Given their perspectives, do you maintain your verdict or change it?\n\n"
                    f"IMPORTANT: Your verdict MUST be exactly one of these labels:\n"
                    f"- Accurate: Claims are well-supported and properly contextualized\n"
                    f"- Misleading: Claims contain factual errors or misleading framing\n"
                    f"- Biased: Claims show clear bias but facts are not necessarily wrong\n"
                    f"- Inconclusive: Evidence is insufficient to make a determination\n\n"
                    f"Provide updated assessment as JSON with keys: verdict (must be one of the above), confidence (0.0-1.0), reasoning, changed (boolean)"
                )
                
                response = self.gemini.generate_json(
                    prompt=debate_prompt,
                    timeout=45,
                    temperature=0.2  # Lower temp for deliberation
                )
                
                if response.get("ok"):
                    verdict = response["data"]
                    # Handle case where data might be a list
                    if isinstance(verdict, list):
                        print(f"Warning: Agent {agent_id} returned list, using first item")
                        verdict = verdict[0] if verdict else {}
                    verdict["agent"] = agent_config["name"]
                    verdict["agent_id"] = agent_id
                    verdict["turn"] = turn + 1
                    new_verdicts.append(verdict)
                else:
                    print(f"Agent {agent_id} debate turn {turn+1} failed: {response.get('error')}")
            
            current_verdicts = new_verdicts
        
        return current_verdicts
    
    def _normalize_verdict(self, verdict: str) -> str:
        """
        Normalize verdict labels to standard categories.
        
        Args:
            verdict: Raw verdict string from agent
            
        Returns:
            Normalized verdict label
        """
        verdict_lower = verdict.lower().strip()
        
        # Map common variations to standard labels
        if any(word in verdict_lower for word in ['accurate', 'valid', 'correct', 'sound', 'reliable']):
            return "Accurate"
        elif any(word in verdict_lower for word in ['misleading', 'inconsistent', 'false', 'incorrect', 'manipulative', 'deceptive']):
            return "Misleading"
        elif any(word in verdict_lower for word in ['biased', 'bias', 'slanted', 'partial']):
            return "Biased"
        elif any(word in verdict_lower for word in ['inconclusive', 'insufficient', 'unclear', 'unverified', 'speculative']):
            return "Inconclusive"
        else:
            # Default to Inconclusive if unclear
            return "Inconclusive"
    
    def calculate_consensus(self, final_verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate final consensus from agent verdicts.
        
        Args:
            final_verdicts: List of final agent verdicts
            
        Returns:
            Dict with consensus verdict and metadata
        """
        if not final_verdicts:
            return {
                "verdict": "Inconclusive",
                "confidence": 0.0,
                "reason": "No agent verdicts available"
            }
        
        # Count verdicts
        verdict_counts = {}
        total_confidence = 0.0
        
        for v in final_verdicts:
            raw_verdict = v.get("verdict", "Unknown")
            verdict_label = self._normalize_verdict(raw_verdict)
            
            # Handle confidence as either float or string
            confidence = v.get("confidence", 0.0)
            if isinstance(confidence, str):
                try:
                    # Try to parse percentage string like "70%" or "0.7"
                    confidence = confidence.strip().rstrip('%')
                    confidence = float(confidence)
                    # If it was a percentage, convert to decimal
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                except (ValueError, AttributeError):
                    confidence = 0.0
            
            if verdict_label not in verdict_counts:
                verdict_counts[verdict_label] = {"count": 0, "total_confidence": 0.0}
            
            verdict_counts[verdict_label]["count"] += 1
            verdict_counts[verdict_label]["total_confidence"] += confidence
            total_confidence += confidence
        
        # Find majority verdict
        majority = max(verdict_counts.items(), key=lambda x: x[1]["count"])
        majority_verdict = majority[0]
        majority_count = majority[1]["count"]
        
        # Calculate consensus strength
        consensus_percentage = (majority_count / len(final_verdicts)) * 100
        avg_confidence = majority[1]["total_confidence"] / majority_count
        
        # Require 80% consensus threshold
        if consensus_percentage >= 80 and avg_confidence >= 0.7:
            return {
                "verdict": majority_verdict,
                "confidence": avg_confidence,
                "consensus_percentage": consensus_percentage,
                "agent_count": len(final_verdicts),
                "reasoning": f"{majority_count}/{len(final_verdicts)} agents agreed"
            }
        else:
            return {
                "verdict": "Inconclusive",
                "confidence": 0.0,
                "consensus_percentage": consensus_percentage,
                "agent_count": len(final_verdicts),
                "reasoning": f"Insufficient consensus: {consensus_percentage:.0f}% agreement"
            }
    
    def executive_decision(self, agent_verdicts: List[Dict], evidence: str, current_date: str = None) -> Dict[str, Any]:
        """
        Layer 4.5: Executive Decision Agent reconciles conflicts and produces final verdict.
        
        Args:
            agent_verdicts: List of agent verdict dicts (initial or final)
            evidence: Formatted evidence string
            current_date: Current date string
            
        Returns:
            Dict with executive decision including final_verdict, decision_basis, override_reason, etc.
        """
        from datetime import datetime
        
        if not current_date:
            current_date = datetime.now().strftime("%B %d, %Y")
        
        # Format agent verdicts for executive review
        agent_summary = json.dumps(agent_verdicts, indent=2)
        
        # Get executive agent prompt
        exec_config = self.agents.get("executive")
        if not exec_config:
            # Fallback: no executive agent, use traditional consensus
            return self.calculate_consensus(agent_verdicts)
        
        prompt = exec_config["prompt_template"].format(
            agent_verdicts=agent_summary,
            evidence=evidence,
            current_date=current_date
        )
        
        try:
            response = self.gemini.generate_json(
                prompt=prompt,
                timeout=60,
                temperature=0.1  # Very low temp for consistent policy application
            )
            
            # Validate response structure
            if isinstance(response, list):
                response = response[0]
            
            # Extract data from Gemini response wrapper
            if isinstance(response, dict) and "data" in response:
                response = response["data"]
            
            # Ensure required fields exist
            if not response.get("final_verdict"):
                print(f"  ✗ WARNING: Executive agent missing final_verdict field")
                print(f"  Response keys: {list(response.keys()) if isinstance(response, dict) else 'not a dict'}")
                return self.calculate_consensus(agent_verdicts)
            
            # Add metadata
            response["agent"] = "Executive"
            response["agent_id"] = "executive"
            
            return response
            
        except Exception as e:
            print(f"  ✗ ERROR in Executive Decision: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to traditional consensus
            return self.calculate_consensus(agent_verdicts)
    
    def run_full_protocol(self, evidence: str, current_date: str = None) -> Dict[str, Any]:
        """
        Run complete consensus protocol: initial verdicts → debate → consensus.
        
        Args:
            evidence: Formatted evidence string
            current_date: Current date string for temporal context
            
        Returns:
            Dict with final consensus and full debate transcript
        """
        # Stage 1: Initial verdicts
        print("Gathering initial agent verdicts...")
        initial_verdicts = self.gather_agent_verdicts(evidence, current_date=current_date)
        
        # Stage 2: Structured debate
        print("Conducting agent debate...")
        final_verdicts = self.conduct_debate(initial_verdicts, evidence, max_turns=2)
        
        # Stage 3: Executive Decision (Layer 4.5)
        print("Executive agent reviewing verdicts...")
        executive_decision = self.executive_decision(final_verdicts, evidence, current_date=current_date)
        
        # Stage 4: Traditional consensus (for comparison/fallback)
        consensus = self.calculate_consensus(final_verdicts)
        
        return {
            "consensus": executive_decision,  # Use executive decision as primary consensus
            "traditional_consensus": consensus,  # Keep for reference
            "initial_verdicts": initial_verdicts,
            "final_verdicts": final_verdicts,
            "debate_turns": 2
        }
