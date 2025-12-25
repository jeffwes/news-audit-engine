# News Audit Engine v4.0

Advanced narrative integrity analysis system using multi-stage NER, semantic conflict detection, and multi-agent consensus protocols.

## Architecture

### Layer 1: Decomposition (Narrative & Entity Architect)
- Two-stage NER: Fast contextual grounding (en_core_web_md) → Precision extraction (en_core_web_trf)
- Entity linking to Wikidata for disambiguation
- Temporal anchoring with absolute date resolution
- Quote attribution via SaysWho package

### Layer 2: Diverse Search (Investigatory Team)
- Multi-intent search portfolio via Tavily API:
  - Confirming searches (supporting evidence)
  - Adversarial searches (debunking/contradictions)
  - Contextual searches (background/history)
  - Consensus searches (authoritative wire services)

### Layer 3: Synthesis & Conflict Detection (Semantic Judge)
- Qdrant vector database for semantic similarity
- Embedding via Gemini text-embedding-004 (1536 dimensions)
- Knowledge conflict detection (opposing views on same topic)
- Omission scoring (consensus facts absent from article)

### Layer 4: Consensus (Stability Protocol)
- Multi-agent debate:
  - The Auditor (logical consistency, emotional framing)
  - The Contextualist (temporal logic, historical continuity)
  - The Skeptic (evidence quality, devil's advocate)
- 80% confidence threshold for verdicts
- Outputs: Accurate / Misleading / Biased / Inconclusive

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_md
python -m spacy download en_core_web_trf

# Set environment variables
export GEMINI_API_KEY="your-key-here"
export TAVILY_API_KEY="your-key-here"
```

## Usage

```bash
# CLI mode
python cli.py "paste your article here..."

# Or from file
python cli.py --file article.txt
```

## Project Structure

```
News-Audit-Engine/
├── src/
│   ├── ner_pipeline.py         # Two-stage NER + entity linking
│   ├── search_portfolio.py     # Tavily multi-intent search
│   ├── semantic_judge.py       # Qdrant conflict detection
│   ├── consensus_protocol.py   # Multi-agent debate
│   ├── gemini_client.py        # Shared API client
│   └── utils.py                # Shared utilities
├── prompts/
│   ├── pillar_extraction.json  # Narrative pillar prompt
│   ├── agent_personas.json     # Multi-agent definitions
│   └── search_templates.json   # Search query templates
├── tests/
│   └── test_*.py               # Unit tests per layer
├── cli.py                      # Command-line interface
└── requirements.txt
```

## Development Roadmap

- [x] Project scaffolding
- [ ] Phase 1: NER pipeline + pillar extraction
- [ ] Phase 2: Search portfolio + result aggregation
- [ ] Phase 3: Qdrant integration + conflict detection
- [ ] Phase 4: Multi-agent consensus protocol
- [ ] Phase 5: Integration with Project Truth UI

## License

Private research project.
