"""Shared utilities for News Audit Engine."""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_prompt(name: str, prompts_dir: str = "prompts") -> Dict[str, Any]:
    """
    Load a prompt configuration from JSON file.
    
    Args:
        name: Name of the prompt file (without .json extension)
        prompts_dir: Directory containing prompt files
        
    Returns:
        Dict with prompt configuration
    """
    prompt_path = Path(prompts_dir) / f"{name}.json"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r') as f:
        return json.load(f)


def get_api_key(key_name: str) -> str:
    """
    Get API key from environment with helpful error message.
    
    Args:
        key_name: Name of the environment variable
        
    Returns:
        API key string
        
    Raises:
        ValueError: If key not found in environment
    """
    key = os.environ.get(key_name)
    if not key:
        raise ValueError(
            f"{key_name} not found in environment. "
            f"Please set it in your .env file or export it."
        )
    return key


def truncate_text(text: str, max_chars: int = 8000, suffix: str = "...") -> str:
    """
    Truncate text to max characters with ellipsis.
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters to keep
        suffix: Suffix to append if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def format_entity(entity_text: str, entity_type: str, wikidata_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Format entity information into structured dict.
    
    Args:
        entity_text: The entity text (e.g., "Apple Inc.")
        entity_type: Entity type (PERSON, ORG, GPE, DATE)
        wikidata_id: Optional Wikidata ID (e.g., "Q312")
        
    Returns:
        Formatted entity dict
    """
    entity = {
        "text": entity_text,
        "type": entity_type
    }
    
    if wikidata_id:
        entity["wikidata_id"] = wikidata_id
        entity["wikidata_url"] = f"https://www.wikidata.org/wiki/{wikidata_id}"
    
    return entity


class ProgressTracker:
    """Simple progress tracker for CLI output."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_phase = None
    
    def start_phase(self, phase: str):
        """Start a new phase."""
        self.current_phase = phase
        if self.verbose:
            print(f"\n[{phase}] Starting...")
    
    def update(self, message: str):
        """Update current phase with message."""
        if self.verbose and self.current_phase:
            print(f"  → {message}")
    
    def complete_phase(self, message: str = "Complete"):
        """Mark current phase as complete."""
        if self.verbose and self.current_phase:
            print(f"[{self.current_phase}] ✓ {message}")
        self.current_phase = None
    
    def error(self, message: str):
        """Log error message."""
        if self.verbose:
            print(f"  ✗ ERROR: {message}")
