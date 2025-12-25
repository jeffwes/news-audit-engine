"""Gemini API client for News Audit Engine."""
import os
import json
import requests
from typing import Dict, Any, Optional, List


class GeminiClient:
    """Client for Google Gemini API with JSON mode support."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize client with API key from environment or parameter."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment or passed to constructor")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.default_model = "gemini-3-flash-preview"
    
    def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        thinking_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON response from Gemini API.
        
        Args:
            prompt: The input text prompt
            schema: Optional JSON schema to constrain output
            timeout: Request timeout in seconds
            system_instruction: Optional system instruction
            model: Model name (defaults to gemini-3-flash-preview)
            temperature: Sampling temperature (0.0-2.0)
            thinking_level: Thinking level for Gemini 3 models (minimal, low, medium, high)
            
        Returns:
            Dict with 'ok' (bool), 'data' (dict if successful), 'error' (str if failed)
        """
        model_name = model or self.default_model
        url = f"{self.base_url}/{model_name}:generateContent?key={self.api_key}"
        
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json"
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        if schema:
            payload["generationConfig"]["responseSchema"] = schema
        
        if thinking_level:
            payload["thinkingConfig"] = {"thinkingLevel": thinking_level}
        
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    text = candidate["content"]["parts"][0].get("text", "")
                    
                    # Parse JSON
                    try:
                        data = json.loads(text)
                        return {"ok": True, "data": data, "raw": result}
                    except json.JSONDecodeError as e:
                        return {"ok": False, "error": f"Invalid JSON response: {e}", "raw": result}
            
            return {"ok": False, "error": "No valid response from API", "raw": result}
            
        except requests.exceptions.Timeout:
            return {"ok": False, "error": f"Request timed out after {timeout}s"}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"ok": False, "error": f"Unexpected error: {str(e)}"}
    
    def generate_embedding(
        self,
        text: str,
        model: str = "text-embedding-004"
    ) -> Dict[str, Any]:
        """
        Generate text embedding using Gemini API.
        
        Args:
            text: Text to embed
            model: Embedding model name
            
        Returns:
            Dict with 'ok' (bool), 'embedding' (list of floats), 'error' (str if failed)
        """
        url = f"{self.base_url}/{model}:embedContent?key={self.api_key}"
        
        payload = {
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if "embedding" in result and "values" in result["embedding"]:
                return {
                    "ok": True,
                    "embedding": result["embedding"]["values"],
                    "dimensions": len(result["embedding"]["values"])
                }
            
            return {"ok": False, "error": "No embedding in response", "raw": result}
            
        except Exception as e:
            return {"ok": False, "error": f"Embedding generation failed: {str(e)}"}
