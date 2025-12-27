"""
Layer 1.5: Semantic Enrichment - Linguistic Analysis for Machine-Readable Pillars.

Analyzes narrative pillars to detect:
- Change indicators (temporal shifts, position evolution)
- Claim types (factual, positional, evolutionary)
- Temporal contrasts (from X to Y patterns)
- Stance transitions
"""
import spacy
from typing import Dict, Any, List, Optional


class SemanticEnrichment:
    """Enriches narrative pillars with linguistic metadata."""
    
    # Verbs indicating position/policy changes
    CHANGE_VERBS = {
        'shift', 'reverse', 'abandon', 'adopt', 'change', 'switch', 
        'pivot', 'alter', 'modify', 'replace', 'revise', 'update',
        'withdraw', 'retract', 'backtrack', 'flip', 'U-turn'
    }
    
    # Temporal contrast patterns
    TEMPORAL_MARKERS = [
        'from', 'to', 'previously', 'now', 'used to', 'no longer',
        'formerly', 'currently', 'originally', 'initially', 'then'
    ]
    
    def __init__(self):
        """Initialize spaCy model for linguistic analysis."""
        self.nlp = None  # Load lazily
    
    def _load_model(self):
        """Load spaCy transformer model."""
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_trf")
            except OSError:
                raise RuntimeError(
                    "en_core_web_trf not found. "
                    "Install with: python -m spacy download en_core_web_trf"
                )
    
    def enrich_pillar(self, pillar: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single pillar with semantic metadata.
        
        Args:
            pillar: Dict with 'text' and optionally 'entities' from NER
            
        Returns:
            Pillar dict with added fields:
                - change_indicators: Dict with change detection metadata
                - claim_type: Classification of claim type
                - temporal_frame: Extracted old/new position if applicable
        """
        self._load_model()
        text = pillar.get('text', '')
        doc = self.nlp(text)
        
        # Detect change indicators
        change_indicators = self._detect_change_patterns(doc)
        
        # Classify claim type
        claim_type = self._classify_claim_type(doc, change_indicators)
        
        # Extract temporal frame if position evolution detected
        temporal_frame = None
        if change_indicators['has_temporal_shift']:
            temporal_frame = self._extract_temporal_frame(doc, change_indicators)
        
        # Add enrichment to pillar
        enriched = pillar.copy()
        enriched['change_indicators'] = change_indicators
        enriched['claim_type'] = claim_type
        if temporal_frame:
            enriched['temporal_frame'] = temporal_frame
        
        return enriched
    
    def _detect_change_patterns(self, doc) -> Dict[str, Any]:
        """
        Detect linguistic patterns indicating change/evolution.
        
        Returns:
            Dict with:
                - has_temporal_shift: bool
                - change_verbs: List of change verb lemmas found
                - temporal_markers: List of temporal markers found
                - negation_pattern: bool (e.g., "no longer")
        """
        change_verbs = []
        temporal_markers = []
        has_from_to = False
        has_negation = False
        
        # Find change verbs
        for token in doc:
            if token.pos_ == 'VERB' and token.lemma_ in self.CHANGE_VERBS:
                change_verbs.append(token.lemma_)
        
        # Find temporal markers
        text_lower = doc.text.lower()
        for marker in self.TEMPORAL_MARKERS:
            if marker in text_lower:
                temporal_markers.append(marker)
        
        # Detect "from X to Y" pattern
        if 'from' in text_lower and 'to' in text_lower:
            has_from_to = True
        
        # Detect negation patterns (no longer, stopped, ended)
        negation_words = ['no longer', 'stopped', 'ended', 'ceased', 'discontinued']
        for neg in negation_words:
            if neg in text_lower:
                has_negation = True
                break
        
        # Has temporal shift if any indicators present
        has_temporal_shift = (
            len(change_verbs) > 0 or 
            has_from_to or 
            has_negation or
            ('previously' in temporal_markers and 'now' in temporal_markers)
        )
        
        return {
            'has_temporal_shift': has_temporal_shift,
            'change_verbs': change_verbs,
            'temporal_markers': temporal_markers,
            'has_from_to_pattern': has_from_to,
            'has_negation': has_negation
        }
    
    def _classify_claim_type(self, doc, change_indicators: Dict) -> str:
        """
        Classify the type of claim in the pillar.
        
        Returns:
            One of: 'position_evolution', 'position_statement', 
                    'factual_event', 'quote_attribution'
        """
        text_lower = doc.text.lower()
        
        # Position evolution: change indicators present
        if change_indicators['has_temporal_shift']:
            return 'position_evolution'
        
        # Quote attribution: contains reporting verbs
        reporting_verbs = ['said', 'announced', 'stated', 'declared', 'claimed']
        for token in doc:
            if token.lemma_ in reporting_verbs:
                return 'quote_attribution'
        
        # Position statement: contains opinion/stance verbs
        stance_verbs = ['support', 'oppose', 'believe', 'advocate', 'promote']
        for token in doc:
            if token.lemma_ in stance_verbs:
                return 'position_statement'
        
        # Default: factual event
        return 'factual_event'
    
    def _extract_temporal_frame(self, doc, change_indicators: Dict) -> Optional[Dict[str, str]]:
        """
        Extract old vs new position from temporal shift patterns.
        
        Returns:
            Dict with 'old_position', 'new_position', 'transition_verb'
            or None if extraction fails
        """
        text = doc.text
        
        # Try to extract "from X to Y" pattern
        if change_indicators['has_from_to_pattern']:
            # Simple heuristic: find text between "from" and "to"
            try:
                from_idx = text.lower().find('from')
                to_idx = text.lower().find('to', from_idx)
                
                if from_idx != -1 and to_idx != -1:
                    old_position = text[from_idx+5:to_idx].strip()
                    new_position = text[to_idx+3:].strip()
                    
                    # Find transition verb
                    transition_verb = None
                    if change_indicators['change_verbs']:
                        transition_verb = change_indicators['change_verbs'][0]
                    
                    return {
                        'old_position': old_position,
                        'new_position': new_position,
                        'transition_verb': transition_verb
                    }
            except:
                pass
        
        # Try "previously X, now Y" pattern
        if 'previously' in change_indicators['temporal_markers'] and 'now' in change_indicators['temporal_markers']:
            try:
                prev_idx = text.lower().find('previously')
                now_idx = text.lower().find('now', prev_idx)
                
                if prev_idx != -1 and now_idx != -1:
                    old_position = text[prev_idx+11:now_idx].strip()
                    new_position = text[now_idx+4:].strip()
                    
                    return {
                        'old_position': old_position,
                        'new_position': new_position,
                        'transition_verb': 'changed'
                    }
            except:
                pass
        
        return None
    
    def enrich_pillars(self, pillars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich multiple pillars with semantic metadata.
        
        Args:
            pillars: List of pillar dicts
            
        Returns:
            List of enriched pillar dicts
        """
        return [self.enrich_pillar(p) for p in pillars]
