"""
Layer 1: Decomposition - Two-Stage NER Pipeline.

Stage 1: Fast contextual grounding with en_core_web_md
Stage 2: Precision extraction with en_core_web_trf on narrative pillars
"""
import spacy
from typing import Dict, Any, List, Optional
from dateparser import parse as parse_date
from datetime import datetime


class NERPipeline:
    """Two-stage NER pipeline for entity extraction and disambiguation."""
    
    def __init__(self):
        """Initialize spaCy models."""
        self.fast_nlp = None  # en_core_web_md - loaded lazily
        self.precision_nlp = None  # en_core_web_trf - loaded lazily
    
    def _load_fast_model(self):
        """Load fast model for stage 1."""
        if self.fast_nlp is None:
            try:
                self.fast_nlp = spacy.load("en_core_web_md")
            except OSError:
                raise RuntimeError(
                    "en_core_web_md not found. "
                    "Install with: python -m spacy download en_core_web_md"
                )
    
    def _load_precision_model(self):
        """Load transformer model for stage 2."""
        if self.precision_nlp is None:
            try:
                self.precision_nlp = spacy.load("en_core_web_trf")
            except OSError:
                raise RuntimeError(
                    "en_core_web_trf not found. "
                    "Install with: python -m spacy download en_core_web_trf"
                )
    
    def stage_one_contextual(self, text: str) -> Dict[str, Any]:
        """
        Stage 1: Fast NER pass for contextual grounding.
        
        Args:
            text: Full article text
            
        Returns:
            Dict with:
                - entities: List of {text, type, start, end}
                - entity_map: Dict of entity_type -> list of texts
                - tagged_text: Text with entity markers
        """
        self._load_fast_model()
        doc = self.fast_nlp(text)
        
        entities = []
        entity_map = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}
        
        for ent in doc.ents:
            if ent.label_ in entity_map:
                entity = {
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                entities.append(entity)
                entity_map[ent.label_].append(ent.text)
        
        return {
            "entities": entities,
            "entity_map": entity_map,
            "entity_count": len(entities)
        }
    
    def stage_two_precision(self, pillar_text: str, context_entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Stage 2: Precision NER on narrative pillar with entity linking.
        
        Args:
            pillar_text: Text of the narrative pillar
            context_entities: Entity map from stage 1 for disambiguation
            
        Returns:
            Dict with:
                - entities: List of refined entities with Wikidata IDs
                - temporal_anchors: List of resolved dates
                - quotes: List of extracted quotes
        """
        self._load_precision_model()
        doc = self.precision_nlp(pillar_text)
        
        entities = []
        temporal_anchors = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                entity = {
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                
                # TODO: Add Wikidata entity linking
                # For now, just check if entity was in context
                if ent.text in context_entities.get(ent.label_, []):
                    entity["context_match"] = True
                
                entities.append(entity)
            
            elif ent.label_ == "DATE":
                # Resolve relative dates to absolute
                resolved = self._resolve_temporal(ent.text)
                if resolved:
                    temporal_anchors.append({
                        "original": ent.text,
                        "resolved": resolved,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
        
        # TODO: Integrate SaysWho for quote extraction
        quotes = []
        
        return {
            "entities": entities,
            "temporal_anchors": temporal_anchors,
            "quotes": quotes
        }
    
    def _resolve_temporal(self, date_text: str, reference_date: Optional[datetime] = None) -> Optional[str]:
        """
        Resolve relative date expressions to ISO format.
        
        Args:
            date_text: Date expression (e.g., "last Tuesday")
            reference_date: Reference date for relative expressions
            
        Returns:
            ISO date string or None if parsing fails
        """
        try:
            parsed = parse_date(
                date_text,
                settings={
                    'RELATIVE_BASE': reference_date or datetime.now(),
                    'RETURN_AS_TIMEZONE_AWARE': False
                }
            )
            if parsed:
                return parsed.date().isoformat()
        except Exception:
            pass
        return None
    
    def extract_pillars_with_entities(
        self,
        full_text: str,
        pillars: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Complete two-stage pipeline: contextual grounding + precision extraction.
        
        Args:
            full_text: Full article text
            pillars: List of narrative pillars from LLM (with 'text' field)
            
        Returns:
            List of pillars enriched with entity information
        """
        # Stage 1: Contextual grounding on full article
        context = self.stage_one_contextual(full_text)
        
        # Stage 2: Precision extraction on each pillar
        enriched_pillars = []
        for pillar in pillars:
            pillar_entities = self.stage_two_precision(
                pillar.get("text", ""),
                context["entity_map"]
            )
            
            enriched_pillar = {
                **pillar,
                "entities": pillar_entities["entities"],
                "temporal_anchors": pillar_entities["temporal_anchors"],
                "quotes": pillar_entities["quotes"]
            }
            enriched_pillars.append(enriched_pillar)
        
        return enriched_pillars
