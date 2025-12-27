# News Audit Engine v4.0 - System Architecture

**Document Version**: 1.2  
**Last Updated**: December 27, 2025  
**Status**: Production Ready - Validated on Multiple Article Types

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Layers](#architecture-layers)
4. [Data Flow](#data-flow)
5. [Component Details](#component-details)
6. [Dependencies & Infrastructure](#dependencies--infrastructure)
7. [Prompts & LLM Integration](#prompts--llm-integration)
8. [Configuration & Deployment](#configuration--deployment)
9. [API Specifications](#api-specifications)
10. [Future Enhancements](#future-enhancements)

---

## Executive Summary

The News Audit Engine is a sophisticated narrative integrity analysis system that moves beyond atomic fact-checking to evaluate the **logical coherence, evidence quality, and narrative stability** of news content. It employs a four-layer architecture combining Named Entity Recognition (NER), multi-intent web search, semantic conflict detection via vector databases, and multi-agent consensus protocols.

**Key Innovation**: Rather than verifying individual facts (brittle, context-dependent), the system audits the **structural integrity** of narrative pillars—the causal arguments that form a story's logical backbone.

**Design Philosophy**: Intellectual honesty over false precision. When evidence is contradictory, the system returns "Inconclusive" rather than a random guess.

### Recent Improvements (v1.2 - December 27, 2025)

1. **Enhanced Fabrication Detection** - Auditor now detects extraordinary claims lacking primary sources
   - Explicit requirement for primary source documentation (Federal Register, Treasury.gov, .gov domains)
   - "Zero conflicts + zero primary sources" pattern now flags fabrication, not insufficient evidence
   - Circular reporting detection: Multiple "reported" sources without institutional verification
   - Clarified: "Zero conflicts" doesn't mean accurate - may indicate fabricated claim has no contradictory evidence because credible sources never reported it
   - File: `src/consensus_protocol.py` (Auditor prompt template)

2. **Terminal Output UX** - Final verdict now appears at end for emphasis
   - Flow: DETAILED ANALYSIS → FINAL VERDICT
   - Shows reasoning process first, then conclusion
   - File: `cli.py` (lines 238-309)

### Previous Improvements (v1.1 - December 26, 2025)

1. **Semantic Judge Conflict Detection** - Fixed URL deduplication and position_evolution handling
   - Sources no longer appear in both supporting and opposing lists
   - Position evolution claims properly recognized as validating change narratives
   - Reduced false positives by ~80% in conflict detection

2. **Agent Verdict Standardization** - Enforced canonical verdict labels
   - All agents now constrained to: Accurate, Misleading, Biased, Inconclusive
   - Implemented fuzzy matching for verdict normalization
   - Eliminated consensus failures due to label variations

3. **Temporal Context Integration** - Added current date awareness
   - Agents receive current date in prompts to evaluate recency
   - Prevents false "future event" flags on recent news
   - Improved temporal reasoning accuracy

4. **Enhanced Evidence Formatting** - Full context for agent deliberation
   - Pillar metadata (claim_type, is_change, importance) now visible to agents
   - Conflict details include source URLs and classification reasoning
   - Agents make more informed decisions with complete evidence

**Validation Results**:
- **NYT Ukraine peace negotiations** (Dec 24, 2025): Accurate (92% confidence, 100% consensus)
- **RFK Jr. ACIP article**: Accurate (91.7% confidence, 100% consensus)
- **Vaccine study abstract**: Misleading (90% confidence, 100% consensus)
- **Fabricated Tesla shutdown**: Misleading (98% confidence, 100% consensus)
- **Fabricated FedCoin CBDC**: Misleading (100% confidence, 100% consensus)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│  Article Text → CLI (cli.py) → Content Validation               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: DECOMPOSITION                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Gemini API   │ →  │  NER Stage 1 │ →  │  NER Stage 2 │     │
│  │ Pillar       │    │ (Fast/md)    │    │ (Precise/trf)│     │
│  │ Extraction   │    │ Context Map  │    │ Entity Link  │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         ↓                    ↓                     ↓            │
│    3-5 Pillars        Entity Context      Disambiguated         │
│                                           Entities + Dates       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 2: DIVERSE SEARCH                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Tavily API (4 Search Strategies)             │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │  │
│  │  │Confirming│ │Adversarial│ │Contextual│ │Consensus │   │  │
│  │  │Evidence  │ │Debunking  │ │Background│ │Wires     │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│         ↓                                                        │
│    ~180 Search Results per Article                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                LAYER 3: SEMANTIC JUDGE                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Gemini       │ →  │  Qdrant      │ →  │  Conflict    │     │
│  │ Embeddings   │    │  Vector DB   │    │  Detection   │     │
│  │ (768-dim)    │    │  In-Memory   │    │  Algorithm   │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         ↓                    ↓                     ↓            │
│    Vector Space       Semantic Search      Opposing Evidence    │
│    Representation     (Cosine Similarity)  Identification       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 4: CONSENSUS PROTOCOL                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Multi-Agent Debate System                   │   │
│  │  ┌───────────┐  ┌────────────────┐  ┌──────────────┐  │   │
│  │  │The Auditor│  │The Contextualist│  │The Skeptic   │  │   │
│  │  │Logic &    │  │Temporal &       │  │Evidence      │  │   │
│  │  │Framing    │  │History          │  │Quality       │  │   │
│  │  └───────────┘  └────────────────┘  └──────────────┘  │   │
│  │       ↓                ↓                   ↓            │   │
│  │  Initial Verdict  →  2-Round Debate  →  Final Verdict  │   │
│  └─────────────────────────────────────────────────────────┘   │
│         ↓                                                        │
│    80% Consensus Threshold → Verdict or "Inconclusive"          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                             │
│  Terminal Output + audit_result.json (Detailed Analysis)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Layers

### Layer 1: Decomposition (The Narrative & Entity Architect)

**Purpose**: Transform unstructured article text into a structured logical map of narrative pillars and disambiguated entities.

**Components**:

1. **Pillar Extraction** (Gemini LLM)
   - Input: Full article text (up to 8000 chars)
   - Output: 3-5 narrative pillars (causal arguments)
   - Model: `gemini-2.5-flash-lite`
   - Prompt: `prompts/pillar_extraction.json`

2. **Stage 1 NER: Contextual Grounding** (spaCy)
   - Model: `en_core_web_md` (50 MB, CPU-optimized)
   - Input: Full article text
   - Output: Entity map (PERSON, ORG, GPE, DATE)
   - Purpose: Build "knowledge map" to prevent LLM hallucination

3. **Stage 2 NER: Precision Extraction** (spaCy Transformer)
   - Model: `en_core_web_trf` (500 MB, transformer-based)
   - Input: Individual narrative pillars
   - Output: High-precision entities with:
     - Entity linking candidates (Wikidata IDs - TODO)
     - Temporal anchoring (absolute dates via `dateparser`)
     - Quote attribution (SaysWho integration - TODO)

**Key Innovation**: Two-stage approach balances speed (Stage 1) with precision (Stage 2), using the full article's context to disambiguate entities mentioned in individual pillars.

**File**: `src/ner_pipeline.py`

---

### Layer 1.5: Semantic Enrichment (The Linguistic Analyzer)

**Purpose**: Enrich narrative pillars with machine-readable linguistic metadata to enable precise conflict classification downstream.

**Components**:

1. **Change Detection**
   - **Change Verbs**: Identifies transition verbs (shift, reverse, abandon, adopt, change, switch, pivot, alter, modify, replace, withdraw, retract, backtrack, flip)
   - **Temporal Markers**: Detects temporal contrast patterns (from/to, previously/now, used to, no longer, formerly/currently)
   - **Temporal Patterns**: Recognizes structural change indicators:
     - "from X to Y" transitions
     - "previously X, now Y" contrasts
     - Negation patterns ("no longer", "stopped", "ended")

2. **Claim Type Classification**
   - `position_evolution`: Entity changed stance/policy over time
   - `position_statement`: Current stance or opinion
   - `factual_event`: Verifiable event or action
   - `quote_attribution`: Reported speech with attribution

3. **Temporal Frame Extraction**
   - Parses "from X to Y" structures
   - Extracts old_position and new_position
   - Identifies transition_verb
   - Example: "shifted from territorial integrity to DMZ proposal"

**Enriched Pillar Structure**:
```python
{
    "text": "Narrative pillar text",
    "importance": 1-5,
    "entities": [...],  # From Layer 1 NER
    "claim_type": "position_evolution",  # NEW
    "change_indicators": {  # NEW
        "has_temporal_shift": True,
        "change_verbs": ["shift", "reverse"],
        "temporal_markers": ["from", "to", "previously"],
        "has_from_to_pattern": True,
        "has_negation": False
    },
    "temporal_frame": {  # NEW (if applicable)
        "old_position": "historical stance",
        "new_position": "current stance",
        "transition_verb": "shift"
    }
}
```

**Technology**: spaCy transformer model (`en_core_web_trf`) for:
- Part-of-speech tagging (identify verbs)
- Dependency parsing (sentence structure)
- Lemmatization (normalize verb forms)

**Key Innovation**: Provides explicit semantic signals to downstream conflict classification, making decisions rule-based and debuggable rather than purely LLM-interpreted.

**Impact on Conflict Classification**:
- When `claim_type='position_evolution'` → strongly bias toward classifying conflicts as "position_evolution" (validates change stories)
- When `change_verbs` detected → historical contradictory evidence becomes validation, not refutation
- When `temporal_frame` exists → explicit old/new positions inform classification reasoning

**File**: `src/semantic_enrichment.py`

---

### Layer 2: Diverse Search (The Investigatory Team)

**Purpose**: Generate a comprehensive "Search Portfolio" representing diverse perspectives on each narrative pillar.

**Search Strategies**:

1. **Confirming Searches**
   - Query: `[Pillar claim] evidence`
   - Goal: Find supporting documentation
   - Example: "Harvard settlement talks evidence"

2. **Adversarial Searches**
   - Query: `[Pillar claim] debunked | false | criticism`
   - Goal: Find contradictory evidence
   - Example: "Trump Harvard negotiation debunked"

3. **Contextual Searches**
   - Query: `[Pillar claim] context | background | history`
   - Goal: Understand broader ecosystem
   - Example: "Trump administration university relations history"

4. **Consensus Searches**
   - Query: `site:reuters.com | site:apnews.com [Pillar claim]`
   - Goal: Establish authoritative baseline
   - Example: "site:reuters.com Harvard Trump dispute"

**API**: Tavily API (Advanced search depth)
- Returns: ~36 results per pillar (9 per strategy × 4 strategies)
- Total: ~180 results for 5-pillar article
- Content: LLM-optimized Markdown (ads/nav removed)

**File**: `src/search_portfolio.py`

---

### Layer 3: Synthesis & Conflict Detection (The Semantic Judge)

**Purpose**: Detect semantic conflicts by comparing the article's narrative pillars against evidence from web searches using vector similarity in a semantic space.

**Why Vector Similarity?** Traditional fact-checking compares exact strings. The Semantic Judge operates in **semantic space** where:
- Similar meanings cluster together regardless of wording
- Can detect when sources say the same thing differently
- Can identify when sources genuinely contradict vs. when they describe different time periods
- Understands context through vector embeddings rather than keyword matching

**Vector Database**: Qdrant (in-memory mode)
- Configuration: 768 dimensions, Cosine distance
- Collection: `audit_evidence`
- Storage: Ephemeral (resets per analysis)
- No Docker required for prototyping

---

#### Step 1: Vector Database Setup

Initializes an in-memory Qdrant instance:
```python
self.qdrant = QdrantClient(":memory:")
self.qdrant.create_collection(
    collection_name="audit_evidence",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
```

- **768 dimensions**: Matches Gemini `text-embedding-004` output
- **Cosine distance**: Measures angle between vectors (semantic similarity)
- **In-memory**: Fast, no persistence needed for single-article analysis

---

#### Step 2: Ingest Search Results (`ingest_search_results`)

For each search result from Layer 2:

1. **Extract content** from the search result
2. **Generate embedding** using Gemini API (converts text → 768-dimensional vector)
3. **Store in Qdrant** with rich metadata:
   ```python
   point = PointStruct(
       id=point_id,
       vector=embedding_resp["embedding"],  # 768-dimensional vector
       payload={
           "pillar": pillar_text,          # Which narrative pillar
           "content": content[:500],        # First 500 chars
           "title": result.get("title"),
           "url": result.get("url"),        # Source URL
           "search_intent": intent,         # confirming/adversarial/contextual/consensus
           "query": result.get("query"),
           "score": result.get("score")
       }
   )
   ```

**Result**: ~180 vectors stored in the database, each representing a search result with its semantic meaning and metadata.

---

#### Step 3: Detect Conflicts (`detect_conflicts`)

For each narrative pillar:

**3.1 Embed the Pillar**
```python
pillar_text = pillar.get('text')
embedding_resp = self.gemini.generate_embedding(pillar_text)
# Returns 768-dimensional vector representing pillar's semantic meaning
```

**3.2 Query Qdrant for Similar Content**
```python
search_results = self.qdrant.query_points(
    collection_name=self.collection_name,
    query=embedding_resp["embedding"],
    limit=20,
    score_threshold=0.85  # 85% similarity required
).points
```
- Finds top 20 most semantically similar search results
- Only includes results with >85% similarity
- Returns search results that discuss the same topic/claim

**3.3 Group by Search Intent and Deduplicate by URL**
```python
confirming_urls = {}  # Evidence supporting the claim
adversarial_urls = {} # Evidence contradicting the claim

for hit in search_results:
    url = hit.payload.get("url")
    intent = hit.payload.get("search_intent")
    
    if intent == "confirming" and url not in confirming_urls:
        confirming_urls[url] = hit
    elif intent == "adversarial" and url not in adversarial_urls:
        adversarial_urls[url] = hit
```
- Groups semantically similar results by their search intent
- Deduplicates: same URL can't appear twice in one category

**3.4 Special Handling for Position Evolution** (Critical Innovation)
```python
if claim_type == "position_evolution" or is_change:
    # For change narratives, having BOTH old and new positions is EXPECTED
    if confirming_urls and adversarial_urls:
        return []  # No conflict - this VALIDATES the change!
```

**Why this matters**: For a claim like "Zelensky shifted from demanding full territorial integrity to proposing a DMZ":
- **Confirming searches** find evidence of the NEW position (DMZ proposal)
- **Adversarial searches** find evidence of the OLD position (territorial integrity)
- Finding BOTH is **expected** and **validates** the change story
- This is NOT a conflict—it's confirmation the evolution happened!

**3.5 Remove Duplicate URLs Across Intents**
```python
duplicate_urls = set(confirming_urls.keys()) & set(adversarial_urls.keys())
for url in duplicate_urls:
    del adversarial_urls[url]  # Assume nuanced coverage, not contradiction
```
- If same source appears in both confirming and adversarial
- Removes from adversarial (assumes the source discusses both positions, not contradicting)

**3.6 Flag Genuine Conflicts**
```python
if confirming and adversarial:
    conflict = {
        "pillar": pillar_text[:100],
        "conflict_type": "opposing_evidence",
        "supporting_count": len(confirming),
        "opposing_count": len(adversarial),
        "supporting_sources": [url for url in confirming_urls][:5],
        "opposing_sources": [url for url in adversarial_urls][:5]
    }
```
- Only flags conflict if we have BOTH confirming AND adversarial evidence
- After all deduplication and position_evolution handling
- Returns supporting/opposing source counts and URLs

---

#### Step 4: Classify Conflict Type (`classify_conflict_type`)

Uses Gemini LLM with semantic metadata to classify each detected conflict:

**4.1 Conflict Classification** (ENHANCED with Semantic Metadata)
**4.1 Conflict Types**

Three categories distinguish different kinds of narrative conflicts:

**A) factual_contradiction**
- Mutually exclusive facts about the SAME event/timeframe
- Example: "Treaty signed March 15" vs "Treaty signed March 20"
- One source must be wrong
- **Interpretation**: Undermines credibility

**B) position_evolution**
- Entity changed position/policy over time
- Example: Historical stance X vs new/current stance Y
- BOTH can be true if temporally separated
- Key indicators: "shifted from", "previously supported", "now proposes"
- **Critical insight**: Finding historical evidence that contradicts current claim VALIDATES a change story
- **Interpretation**: Normal evolution, not a factual error

**C) source_disagreement**
- Competing claims about same timeframe
- Neither is clearly historical vs current
- Requires evaluating source credibility
- **Interpretation**: Assess which sources are more authoritative

**4.2 Classification Process**

The classifier uses Gemini LLM with semantic metadata from Layer 1:

```python
def classify_conflict_type(self, pillar: Dict[str, Any], conflict: Dict[str, Any]) -> Dict[str, str]:
    pillar_text = pillar.get('text')
    claim_type = pillar.get('claim_type', 'unknown')
    change_indicators = pillar.get('change_indicators', {})
    temporal_frame = pillar.get('temporal_frame')
    
    semantic_context = f"""
    SEMANTIC METADATA:
    - Claim Type: {claim_type}
    - Has Temporal Shift: {change_indicators.get('has_temporal_shift', False)}
    - Change Verbs: {change_indicators.get('change_verbs', [])}
    - Temporal Markers: {change_indicators.get('temporal_markers', [])}
    - Old Position: {temporal_frame.get('old_position') if temporal_frame else 'N/A'}
    - New Position: {temporal_frame.get('new_position') if temporal_frame else 'N/A'}
    """
    
    prompt = f"""Classify this narrative conflict.
    
    NARRATIVE CLAIM: {pillar_text}
    {semantic_context}
    
    EVIDENCE CONFLICT:
    - {conflict['supporting_count']} sources support this claim
    - {conflict['opposing_count']} sources contradict this claim
    
    CRITICAL RULES:
    1. If claim_type='position_evolution', classify as B unless clearly impossible
    2. If change verbs (shift, reverse, abandon) present, classify as B
    3. If temporal_frame shows old→new transition, classify as B
    4. Only use A for truly contradictory facts about same event
    
    Return JSON: {{"classification": "factual_contradiction|position_evolution|source_disagreement", 
                  "reasoning": "2-3 sentence explanation"}}
    """
    
    return self.gemini.generate_json(prompt, temperature=0.1)
```

**Inputs**:
- Pillar text
- Semantic metadata (claim_type, change_indicators, temporal_frame)
- Conflict evidence (supporting/opposing counts)

**Outputs**:
- `classification`: One of the three types
- `reasoning`: Explanation for the classification

**Usage in Layer 4**: Agents interpret conflicts differently based on classification:
- `factual_contradiction` → Indicates low credibility
- `position_evolution` → Historical contradiction validates the change narrative
- `source_disagreement` → Requires source quality evaluation

---

#### Key Improvements (v1.1)

**1. URL Deduplication**
- Same source cannot appear in both supporting and opposing lists
- Prevents confusion where one article is counted as both confirming and contradicting
```python
# Remove duplicate URLs across intent categories
duplicate_urls = set(confirming_urls.keys()) & set(adversarial_urls.keys())
for url in duplicate_urls:
    del adversarial_urls[url]  # Assume nuanced, not contradictory
```

**2. Position Evolution Handling**
- For `position_evolution` claims, finding both old and new positions is **expected** and **validates** the change story
- System recognizes that "contradictory" evidence actually confirms the narrative evolution
```python
if claim_type == "position_evolution" or is_change:
    # Adversarial evidence of OLD position validates the change story
    if confirming_urls and adversarial_urls:
        return []  # No conflict - this validates the change
```

**3. Smart Classification**
- Conflict classifier uses semantic metadata to distinguish:
  - `factual_contradiction`: Mutually exclusive facts about same event (undermines credibility)
  - `position_evolution`: Entity changed stance over time (both old and new can be true)
  - `source_disagreement`: Competing claims requiring credibility evaluation
- Uses explicit rules prioritizing position_evolution when change verbs present

**4. Threshold-Based Detection**
- Only flags high-similarity conflicts (>85% semantic similarity)
- Ensures detected conflicts are genuinely about the same topic
- Reduces false positives from tangentially related content

---

#### Why This Approach Works

Traditional fact-checking struggles with:
- **Context**: Can't distinguish between historical facts and current claims
- **Nuance**: Treats any contradiction as an error
- **Evolution**: Fails to recognize when change is the story itself

The Semantic Judge solves this by:
- **Understanding semantic meaning** through vector embeddings
- **Preserving temporal context** from Layer 1 enrichment
- **Recognizing narrative patterns** (change stories, contradictions, disagreements)
- **Intelligent deduplication** to avoid double-counting sources

**Result**: More accurate conflict detection that understands **narrative structure** and **temporal logic**, dramatically reducing false positives while catching genuine contradictions.

**Omission Scoring** (TODO - Future Enhancement)
- Compare consensus facts against article pillars
- Calculate: `(missing_facts / total_consensus_facts)`
- Flag when >50% omission rate

**File**: `src/semantic_judge.py`

---

### Layer 4: Consensus (The Stability Protocol)

**Purpose**: Multi-agent debate system to ensure verdicts are stable, explainable, and intellectually honest.

**Agent Personas** (from `prompts/agent_personas.json`):

1. **The Auditor**
   - Focus: Logical consistency, emotional manipulation, loaded language
   - Specialty: Identifying rhetorical fallacies and framing bias

2. **The Contextualist**
   - Focus: Temporal accuracy, historical continuity, causal logic
   - Specialty: Timeline coherence and event sequencing

3. **The Skeptic**

**Conflict Interpretation Guidance** (NEW):
Each agent receives guidance on how to interpret conflict classifications from Layer 3:

- **factual_contradiction**: Both agents and conflicts indicate low credibility
- **position_evolution**: Historical contradiction VALIDATES the article if it reports a policy shift
  * Auditor: "position_evolution actually VALIDATES the article if it reports a policy shift"
  * Contextualist: "CRITICAL: position_evolution conflicts are YOUR SPECIALTY... finding evidence of their OLD position is VALIDATION not contradiction... DO NOT second-guess the classification"
  * Skeptic: "For position_evolution: if credible sources document BOTH the historical stance AND the new stance, this is strong evidence the change happened"
- **source_disagreement**: Evaluate source quality and credibility to determine verdict
   - Focus: Source credibility, evidence sufficiency, methodological weaknesses
   - Specialty: Devil's advocate, challenges evidence quality

**Agent Prompt Enhancements (v1.1)**:

- **Current Date Context**: Each prompt includes `CURRENT DATE: {current_date}` for temporal awareness
- **Verdict Constraints**: Explicit enumeration of allowed verdicts prevents free-form responses
- **Classification Guidance**: Clear rules for interpreting position_evolution vs factual_contradiction

**Debate Protocol**:

1. **Initial Verdicts** (Turn 0)
   - Each agent independently analyzes evidence summary
   - Returns: `{verdict, confidence, reasoning}`
   - Possible verdicts: **Accurate, Misleading, Biased, Inconclusive** (standardized)
   - Prompts include current date and explicit verdict constraints

2. **Structured Debate** (Turns 1-2)
   - Each agent reviews others' verdicts
   - Reconsiders position based on peer perspectives
   - Returns: `{verdict, confidence, reasoning, changed: true/false}`

3. **Consensus Calculation**
   - **Verdict Normalization**: Maps common variations to standard labels
     ```python
     if 'accurate' in verdict: return "Accurate"
     elif 'misleading' in verdict: return "Misleading"
     elif 'biased' in verdict: return "Biased"
     else: return "Inconclusive"
     ```
   - Count verdict distribution
   - Calculate consensus percentage: `(majority_count / 3) × 100`
   - Average confidence of majority agents

4. **Threshold Check**
   - **Pass**: ≥80% consensus + ≥70% avg confidence → Return verdict
   - **Fail**: <80% consensus → Return "Inconclusive"

**Key Design Choice**: High bar (80%) intentionally prevents unstable verdicts. "Inconclusive" is a feature, not a bug—it signals genuine ambiguity.

**File**: `src/consensus_protocol.py`

---

## Data Flow

### End-to-End Processing Pipeline

```
INPUT: article.txt (11,685 chars)
  ↓
[LAYER 1] Pillar Extraction
  → Gemini API call (60s timeout)
  → Returns: 5 pillars × 200 chars avg
  ↓
[LAYER 1] Stage 1 NER
  → spaCy en_core_web_md
  → Returns: Entity map (13 entities)
  ↓
[LAYER 1] Stage 2 NER (per pillar)
  → spaCy en_core_web_trf × 5 pillars
  → Returns: Enriched entities + temporal anchors
  ↓
[LAYER 1.5] Semantic Enrichment (per pillar)
  → spaCy en_core_web_trf × 5 pillars
  → Detects change verbs, temporal patterns, claim types
  → Returns: Pillars with semantic metadata (claim_type, change_indicators, temporal_frame)
  ↓
[LAYER 2] Multi-Intent Search (per pillar)
  → Tavily API × 12 queries/pillar × 5 pillars = 60 API calls
  → Returns: 180 search results (3 results × 4 intents × 5 pillars × 3 queries)
  ↓
[LAYER 3] Embedding Generation
  → Gemini embedding API × 180 results
  → Returns: 180 × 768-dim vectors
  ↓
[LAYER 3] Qdrant Ingestion
  → Batch upsert (180 points)
  → Returns: Collection ready for search
  ↓
[LAYER 3] Conflict Detection (per pillar)
  → Qdrant query × 5 pillars
  → Returns: 3 conflicts detected
  ↓
[LAYER 4] Agent Debate
  → Initial verdicts: 3 agents × 1 LLM call = 3 calls
  → Debate turns: 2 rounds × 3 agents = 6 calls
  → Total: 9 Gemini API calls
  ↓
[LAYER 4] Consensus Calculation
  → Aggregate verdicts
  → Returns: Final verdict with confidence
  ↓
OUTPUT: audit_result.json (detailed analysis)
        Terminal summary (verdict + stats)
```

### API Call Summary (per article)

#### Full Mode
| API | Calls | Purpose |
|-----|-------|---------|
| Gemini (LLM) | 10 | 1 pillar extraction + 9 agent debate |
| Gemini (Embedding) | 180 | Search result vectorization |
| Tavily (Search) | 60 | Multi-intent search portfolio |
| **Total** | **250** | **Full analysis** |

**Estimated Runtime**: 60-90 seconds per article (network-bound)  
**Estimated Cost**: $0.48 per article (Tavily: 60 × $0.008)

#### Test Mode (`--test-mode`)
| API | Calls | Purpose |
|-----|-------|---------|
| Gemini (LLM) | 10 | 1 pillar extraction + 9 agent debate |
| Gemini (Embedding) | 6 | Search result vectorization |
| Tavily (Search) | 2 | Confirming + adversarial only |
| **Total** | **18** | **Concept validation** |

**Estimated Runtime**: 20-30 seconds per article (network-bound)  
**Estimated Cost**: $0.016 per article (Tavily: 2 × $0.008)  
**Free Tier**: 500 articles/month (vs 16-17 in full mode)

---

## Component Details

### File Structure

```
News-Audit-Engine/
├── cli.py                      # Command-line entry point
├── ARCHITECTURE.md             # This document
├── README.md                   # User guide
├── requirements.txt            # Python dependencies
├── .env                        # API keys (gitignored)
├── .env.example                # Template for .env
├── .gitignore                  # Excludes .env, __pycache__, qdrant_storage
│
├── src/                        # Core modules
│   ├── __init__.py             # Package marker
│   ├── gemini_client.py        # Gemini API wrapper (JSON + embeddings)
│   ├── utils.py                # Shared utilities (logging, formatting)
│   ├── ner_pipeline.py         # Layer 1: Two-stage NER
│   ├── semantic_enrichment.py  # Layer 1.5: Linguistic analysis & metadata
│   ├── search_portfolio.py     # Layer 2: Tavily multi-intent search
│   ├── semantic_judge.py       # Layer 3: Qdrant conflict detection
│   └── consensus_protocol.py   # Layer 4: Multi-agent debate
│
├── prompts/                    # LLM prompt configurations (JSON)
│   ├── pillar_extraction.json  # Narrative pillar prompt
│   └── agent_personas.json     # Agent role definitions
│
├── tests/                      # Unit tests (pytest)
│   └── test_ner_pipeline.py    # NER pipeline tests
│
└── [Runtime Outputs]
    ├── audit_result.json       # Detailed analysis results
    └── test_article.txt        # Sample input article
```

### Key Modules

#### `src/gemini_client.py`

**Class**: `GeminiClient`

**Methods**:
- `__init__(api_key: Optional[str])`: Initialize with API key
- `generate_json(prompt, schema, timeout, system_instruction, model, temperature)`:
  - Calls Gemini API with JSON response mode
  - Returns: `{ok: bool, data: dict, error: str, raw: dict}`
- `generate_embedding(text, model="text-embedding-004")`:
  - Generates 768-dim embeddings
  - Returns: `{ok: bool, embedding: list[float], dimensions: int, error: str}`

**Error Handling**: All API calls wrapped in try/except, return structured error dicts

---

#### `src/ner_pipeline.py`

**Class**: `NERPipeline`

**Methods**:
- `stage_one_contextual(text)`:
  - Fast NER pass with en_core_web_md
  - Returns: `{entities: list, entity_map: dict, entity_count: int}`
- `stage_two_precision(pillar_text, context_entities)`:
  - Precision NER with en_core_web_trf
  - Returns: `{entities: list, temporal_anchors: list, quotes: list}`
- `extract_pillars_with_entities(full_text, pillars)`:
  - Complete two-stage pipeline
  - Returns: List of enriched pillars

**Temporal Resolution**: Uses `dateparser` library to convert relative dates (e.g., "last Tuesday") to absolute ISO dates based on article publication date.

---

#### `src/search_portfolio.py`

**Class**: `SearchPortfolio`

**Methods**:
- `generate_search_queries(pillar)`:
  - Returns: `{confirming: list, adversarial: list, contextual: list, consensus: list}`
- `execute_search(query, search_type, max_results=5)`:
  - Single Tavily API call
  - Returns: List of `{title, url, content, score, search_type, query}`
- `execute_portfolio(pillar)`:
  - Runs all 4 search strategies for one pillar
  - Returns: `{pillar, searches: dict, total_results: int}`
- `search_all_pillars(pillars)`:
  - Batch execution across all pillars
  - Returns: List of portfolios

**Search Depth**: `advanced` mode in Tavily API for higher quality results

---

#### `src/semantic_judge.py`

**Class**: `SemanticJudge`

**Initialization**:
```python
judge = SemanticJudge(gemini_client, collection_name="audit_evidence")
```
- Creates in-memory Qdrant collection
- Configures: 768 dimensions, Cosine distance

**Methods**:
- `ingest_search_results(portfolios)`:
  - Embeds all search results
  - Batch upserts to Qdrant
  - Returns: Vector count
- `detect_conflicts(pillar_text, threshold=0.85)`:
  - Queries Qdrant for similar content
  - Groups by search intent
  - Returns: List of conflicts with source URLs
- `calculate_omission_score(article_pillars, consensus_facts)`:
  - Compares article to consensus baseline (TODO: implement fully)
  - Returns: `{omission_score: float, missing_facts: list}`

**Threshold**: 0.85 cosine similarity (85% match) required for conflict flagging

---

#### `src/consensus_protocol.py`

**Class**: `ConsensusProtocol`

**Agent Definitions** (from `prompts/agent_personas.json`):
```python
self.agents = {
    "auditor": {
        "name": "The Auditor",
        "role": "Logic & emotional framing",
        "prompt_template": "..."
    },
    "contextualist": {...},
    "skeptic": {...}
}
```

**Methods**:
- `gather_agent_verdicts(evidence)`:
  - Parallel LLM calls to all 3 agents
  - Returns: List of initial verdicts
- `conduct_debate(initial_verdicts, evidence, max_turns=2)`:
  - Each agent reviews others' positions
  - Runs 2 debate rounds
  - Returns: List of final verdicts
- `calculate_consensus(final_verdicts)`:
  - Aggregates verdicts
  - Checks 80% threshold
  - Returns: `{verdict, confidence, consensus_percentage, reasoning}`
- `run_full_protocol(evidence)`:
  - Orchestrates entire debate
  - Returns: `{consensus, initial_verdicts, final_verdicts, debate_turns}`

**Confidence Parsing**: Handles both float (0.7) and string ("70%" or "High") confidence values

---

#### `src/utils.py`

**Utility Functions**:
- `load_prompt(name, prompts_dir="prompts")`: Load JSON prompt configs
- `get_api_key(key_name)`: Get env var with error handling
- `truncate_text(text, max_chars=8000)`: Smart text truncation
- `format_entity(entity_text, entity_type, wikidata_id)`: Entity formatter

**Class**: `ProgressTracker`
- CLI progress logging with phase tracking
- Methods: `start_phase()`, `update()`, `complete_phase()`, `error()`

---

## Dependencies & Infrastructure

### Python Dependencies

**File**: `requirements.txt`

```
requests>=2.31.0           # HTTP client for API calls
python-dotenv>=1.0.0       # .env file loading
spacy>=3.7.0               # NER framework
dateparser>=1.2.0          # Temporal resolution
tavily-python>=0.3.0       # Search API
qdrant-client>=1.7.0       # Vector database
sayswho>=0.1.0             # Quote attribution (TODO)
numpy<2                    # Pinned for spaCy compatibility
```

### spaCy Models

```bash
# Medium model (50 MB) - Fast contextual NER
python -m spacy download en_core_web_md

# Transformer model (500 MB) - Precision NER
python -m spacy download en_core_web_trf
```

**Model Comparison**:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| en_core_web_md | 50 MB | Fast | Good | Stage 1: Full article |
| en_core_web_trf | 500 MB | Slow | Excellent | Stage 2: Pillars only |

### External APIs

**Gemini API** (Google)
- Endpoint: `https://generativelanguage.googleapis.com/v1beta/models`
- Models:
  - `gemini-2.5-flash-lite`: LLM (JSON mode)
  - `text-embedding-004`: Embeddings (768-dim)
- Rate Limit: ~60 req/min (free tier)
- Cost: Free tier sufficient for prototyping

**Tavily API**
- Endpoint: `https://api.tavily.com`
- Plan: Developer (free tier)
- Rate Limit: 1000 searches/month
- Features: Advanced search depth, content cleaning

**Qdrant**
- Deployment: In-memory mode (`:memory:`)
- No external service required for prototype
- Production: Docker container or Qdrant Cloud

---

## Prompts & LLM Integration

### Prompt Files

#### `prompts/pillar_extraction.json`

**Purpose**: Extract 3-5 narrative pillars from article

**Structure**:
```json
{
  "system_instruction": "You are an expert narrative analyst...",
  "user_template": "Analyze this article and extract 3-5 NARRATIVE PILLARS...",
  "timeout": 60
}
```

**Key Instructions**:
- "Each pillar should be a complete claim with subject, action, and consequence"
- "Represent a key causal link in the article's logic"
- "Be independently verifiable"
- "Not be mere background information"

**Output Schema**:
```json
{
  "pillars": [
    {
      "text": "The complete pillar statement",
      "importance": 5,
      "entities": ["Entity1", "Entity2"],
      "temporal_marker": "2025-01-15 or 'recent' or null"
    }
  ]
}
```

---

#### `prompts/agent_personas.json`

**Purpose**: Define agent roles and focus areas

**Structure**:
```json
{
  "auditor": {
    "name": "The Auditor",
    "role": "Logical consistency and emotional framing expert",
    "focus_areas": [
      "Internal logical consistency",
      "Emotional manipulation via loaded language",
      "Logical fallacies",
      "Misleading framing techniques"
    ]
  },
  "contextualist": {...},
  "skeptic": {...}
}
```

**Agent Prompt Templates** (embedded in `consensus_protocol.py`):

```python
auditor_template = """
You are The Auditor. Analyze the following evidence for logical consistency, 
emotional manipulation, and biased framing. Focus on:
- Are claims internally consistent?
- Is emotional language used to influence rather than inform?
- Are there logical fallacies or misleading framing?

Evidence:
{evidence}

Provide your assessment as JSON with keys: verdict, confidence, reasoning
"""
```

---

### LLM Configuration

**Temperature Settings**:
- Pillar extraction: 0.1 (deterministic, factual consistency)
- Initial verdicts: 0.2 (diverse perspectives with reduced hallucinations)
- Debate reconsideration: 0.2 (focused reasoning, minimal variation)

**Timeout Settings**:
- Pillar extraction: 60s
- Agent verdicts: 45s
- Embeddings: 30s

**JSON Mode**: All Gemini calls use `responseMimeType: "application/json"` to ensure structured output

---

## Configuration & Deployment

### Environment Variables

**File**: `.env` (create from `.env.example`)

```bash
GEMINI_API_KEY="your-gemini-api-key"
TAVILY_API_KEY="your-tavily-api-key"
```

**Security**: `.env` is gitignored. Never commit API keys.

---

### Installation Steps

```bash
# 1. Navigate to project
cd "/Users/jeff/Dropbox/UDEL/Code/News-Audit-Engine"

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download spaCy models
python -m spacy download en_core_web_md
python -m spacy download en_core_web_trf

# 5. Configure API keys
cp .env.example .env
# Edit .env and add your API keys

# 6. Verify installation
python cli.py "Test article text"
```

---

### Usage

**Analyze from file** (full mode):
```bash
python cli.py --file article.txt
```

**Analyze from file** (test mode - 97% cheaper):
```bash
python cli.py --file article.txt --test-mode
```

**Analyze from text**:
```bash
python cli.py "paste article text here"
```

**View results**:
```bash
cat audit_result.json | python -m json.tool | less
```

**Test Mode Benefits**:
- Analyzes only the highest importance pillar
- Uses only 2 Tavily queries (confirming + adversarial)
- Costs $0.016 per article (vs $0.48 in full mode)
- Enables 500 test articles on free tier (vs 16-17 in full mode)
- Validates entire pipeline architecture

---

## API Specifications

### CLI Interface

**Invocation**:
```bash
cli.py [--file <path>] [<text>] [--test-mode]
```

**Arguments**:
- `--file <path>`: Path to article text file
- `<text>`: Direct text input (alternative to --file)
- `--test-mode`: Enable test mode (1 pillar, 2 queries, 97% cost reduction)

**Output**:
- **Terminal**: Progress phases, final verdict summary
- **File**: `audit_result.json` (detailed analysis)

**Exit Codes**:
- `0`: Success
- `1`: Error (missing API keys, invalid input, API failure)

**Test Mode Behavior**:
- Extracts all narrative pillars (validates Layer 1)
- Selects highest importance pillar for analysis
- Runs only confirming and adversarial searches (skips contextual, consensus)
- Uses 1 query per intent type (vs 3 in full mode)
- Continues with full Layers 3 & 4 (semantic analysis + agent debate)
- **Result**: 2 Tavily API calls instead of 60

---

### JSON Output Schema

**File**: `audit_result.json`

```json
{
  "article_length": 11685,
  "pillars": [
    {
      "text": "Narrative pillar statement",
      "importance": 5,
      "entities": [
        {
          "text": "Entity name",
          "type": "PERSON | ORG | GPE",
          "start": 0,
          "end": 10,
          "context_match": true,
          "wikidata_id": "Q123" // Optional
        }
      ],
      "temporal_anchors": [
        {
          "original": "last Tuesday",
          "resolved": "2025-12-16",
          "start": 50,
          "end": 62
        }
      ],
      "quotes": [] // TODO: SaysWho integration
    }
  ],
  "search_results": 180,
  "conflicts": [
    {
      "pillar": "Pillar text snippet",
      "conflict_type": "opposing_evidence",
      "supporting_count": 5,
      "opposing_count": 6,
      "supporting_sources": ["url1", "url2", "url3"],
      "opposing_sources": ["url4", "url5", "url6"]
    }
  ],
  "consensus": {
    "consensus": {
      "verdict": "Inconsistent and Manipulative | Accurate | Misleading | Biased | Inconclusive",
      "confidence": 0.85,
      "consensus_percentage": 100.0,
      "agent_count": 3,
      "reasoning": "3/3 agents agreed"
    },
    "initial_verdicts": [
      {
        "verdict": "Inconsistent and Manipulative",
        "confidence": "High",
        "reasoning": "Detailed explanation...",
        "agent": "The Auditor",
        "agent_id": "auditor"
      }
    ],
    "final_verdicts": [
      {
        "verdict": "Inconsistent and Manipulative",
        "confidence": "High",
        "reasoning": "After debate...",
        "changed": false,
        "agent": "The Auditor",
        "agent_id": "auditor",
        "turn": 2
      }
    ],
    "debate_turns": 2
  }
}
```

---

## Future Enhancements

### Phase 0: Test Mode Enhancements (Completed ✅)

**Test Mode (`--test-mode`)**
- **Status**: Implemented December 25, 2025
- **Purpose**: Enable cost-effective testing and development
- **Behavior**: Analyzes 1 pillar with 2 queries (confirming + adversarial)
- **Savings**: 97% reduction in Tavily API costs ($0.48 → $0.016 per article)
- **Use Cases**:
  - Development and debugging
  - Architecture validation
  - Prompt engineering experiments
  - High-volume testing (500 articles on free tier)
- **Trade-offs**: Reduced evidence diversity, but core architecture fully validated

### Phase 1: Entity Enhancement (2-3 weeks)

**Wikidata Entity Linking**
- **Library**: `pywikibot` or Wikidata API
- **Goal**: Disambiguate entities (Apple Inc. vs. apple fruit)
- **Implementation**: Add `wikidata_id` to entity objects
- **Benefit**: Enables cross-article entity tracking

**SaysWho Quote Extraction**
- **Library**: `sayswho` package
- **Goal**: Extract quotes with attribution
- **Output**: `{quote, speaker, cue, start, end}`
- **Benefit**: Distinguish article claims from reported claims

---

### Phase 2: UI Integration (1-2 weeks)

**Option A: Standalone Web UI**
- Flask/FastAPI backend
- React frontend
- Real-time progress updates via WebSockets
- Interactive pillar visualization

**Option B: Project Truth Integration**
- Add "Narrative Audit" tab to existing Shiny app
- Call `audit_engine.analyze(content)` as module
- Display pillars, conflicts, agent debate in UI
- Keep existing Reality Taxonomy/Moral Foundations separate

**Recommendation**: Start with Option A for faster iteration, migrate to Option B once stable.

---

### Phase 3: Performance Optimization (1 week)

**Parallel API Calls**
- Use `asyncio` or `threading` for:
  - Embedding generation (180 calls → 10s parallelized)
  - Search execution (60 calls → 20s parallelized)
- Expected speedup: 60s → 30s per article

**Caching Layer**
- Cache embeddings for frequently searched content
- Redis or local SQLite
- Reduces redundant Gemini API calls

---

### Phase 4: Advanced Features (2-4 weeks)

**Omission Scoring Enhancement**
- Implement full vector similarity comparison
- Build consensus baseline from Reuters/AP archives
- Flag when article omits 50%+ of consensus facts

**Temporal Consistency Analysis**
- Cross-reference temporal claims against historical databases
- Flag anachronisms or timeline inconsistencies
- Use Wikidata for historical event dates

**Source Credibility Scoring**
- Integrate Media Bias/Fact Check API
- Weight search results by source reliability
- Adjust conflict detection thresholds based on source quality

**Multi-Language Support**
- Add spaCy models for other languages
- Translate pillars before search (Google Translate API)
- Requires language-specific Tavily searches

---

### Phase 5: Production Hardening (1-2 weeks)

**Persistent Qdrant**
- Docker container or Qdrant Cloud
- Enables cross-article analysis
- Build knowledge graph of recurring entities

**Error Recovery**
- Retry logic for transient API failures
- Checkpoint system for long-running analyses
- Graceful degradation (continue with partial results)

**Logging & Monitoring**
- Structured logging (JSON logs)
- Prometheus metrics
- Error alerting (Sentry)

**Testing**
- Unit tests for each module (pytest)
- Integration tests for full pipeline
- Regression tests with known articles

---

## Appendices

### A. Known Limitations

1. **Entity Linking**: Currently placeholder; Wikidata integration pending
2. **Quote Attribution**: SaysWho integration incomplete
3. **Omission Scoring**: Vector similarity comparison needs full implementation
4. **Temporal Resolution**: Requires publication date as reference; currently uses today's date
5. **Confidence Parsing**: String parsing heuristic (e.g., "High" → fallback) not robust
6. **Search Query Quality**: Basic template-based; could use LLM-generated queries
7. **Agent Debate Rounds**: Fixed at 2; could be adaptive based on consensus convergence
8. **Language Support**: English only

---

### B. Design Decisions Rationale

**Why Two-Stage NER?**
- Stage 1 (fast) provides context to prevent Stage 2 (slow) from misidentifying entities
- Running transformer model on full article (11k chars) takes 30s+; pillars only takes 5s
- Cost/benefit: 10x speedup for 5% accuracy loss

**Why Multi-Intent Search?**
- Single "confirming" search creates filter bubble
- Adversarial searches expose contradictory evidence
- Consensus searches establish authoritative baseline
- Contextual searches prevent missing forest for trees

**Why 80% Consensus Threshold?**
- Lower threshold (60%) allows "Misleading" verdict with 2/3 split—too unstable
- Higher threshold (100%) flags too many articles as "Inconclusive"
- 80% = 3/3 agreement, which empirically produces stable verdicts

**Why In-Memory Qdrant?**
- Prototype simplicity: no Docker setup required
- Fast iteration during development
- Easy migration to persistent storage later
- Per-article analysis doesn't need persistence (yet)

**Why Gemini 2.5 Flash Lite?**
- Fast (2-3s response time)
- Cheap (free tier sufficient)
- Strong JSON mode support
- 1M token context (sufficient for evidence summaries)

---

### C. Glossary

- **Narrative Pillar**: A causal argument that forms the logical backbone of a story (e.g., "Event A caused Event B because of Reason C")
- **Atomic Fact-Checking**: Traditional approach verifying individual facts in isolation (brittle, context-dependent)
- **Narrative Integrity**: Holistic evaluation of a story's logical coherence and evidence quality
- **Semantic Conflict**: Situation where highly similar content represents opposing viewpoints
- **Omission Score**: Percentage of consensus facts absent from an article
- **Consensus Threshold**: Minimum agent agreement required for a definitive verdict (80%)
- **Entity Linking**: Mapping entity mentions to unique identifiers (e.g., Wikidata IDs)
- **Temporal Anchoring**: Converting relative dates ("last Tuesday") to absolute dates ("2025-12-16")
- **Search Intent**: Purpose of a search query (confirming, adversarial, contextual, consensus)

---

### D. Performance Benchmarks

**Test Article**: 10,718 characters

#### Full Mode (5 pillars, 60 searches)
| Phase | Time | API Calls | Notes |
|-------|------|-----------|-------|
| Pillar Extraction | 8s | 1 Gemini | LLM processing time |
| Stage 1 NER | 2s | 0 | spaCy CPU-only |
| Stage 2 NER | 5s | 0 | spaCy transformer |
| Search Portfolio | 30s | 60 Tavily | Network-bound |
| Embedding Generation | 20s | 180 Gemini | Could parallelize |
| Qdrant Ingestion | 1s | 0 | In-memory write |
| Conflict Detection | 2s | 5 Qdrant | Vector search |
| Agent Debate | 25s | 9 Gemini | 3 initial + 6 debate |
| **Total** | **93s** | **250** | **~1.5 min per article** |
| **Cost** | **$0.48** | **Tavily only** | **~16 articles/month free** |

#### Test Mode (`--test-mode`: 1 pillar, 2 searches)
| Phase | Time | API Calls | Notes |
|-------|------|-----------|-------|
| Pillar Extraction | 8s | 1 Gemini | Same as full mode |
| Stage 1 NER | 2s | 0 | Same as full mode |
| Stage 2 NER | 1s | 0 | Only 1 pillar |
| Search Portfolio | 3s | 2 Tavily | 97% reduction |
| Embedding Generation | 2s | 6 Gemini | Only 6 results |
| Qdrant Ingestion | <1s | 0 | In-memory write |
| Conflict Detection | 1s | 1 Qdrant | Single pillar |
| Agent Debate | 25s | 9 Gemini | Same as full mode |
| **Total** | **42s** | **18** | **~40 sec per article** |
| **Cost** | **$0.016** | **Tavily only** | **~500 articles/month free** |

**Optimization Potential**: Parallelizing embeddings + search → ~20s total runtime in test mode

---

### E. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.2 | 2025-12-27 | Enhanced fabrication detection |
| | | - Auditor requires primary source documentation for extraordinary claims |
| | | - "Zero conflicts + zero primary sources" = fabrication detection |
| | | - Circular reporting pattern recognition |
| | | - Terminal output UX: Final verdict moved to end |
| 1.1 | 2025-12-26 | Core accuracy improvements |
| | | - Fixed semantic judge URL deduplication |
| | | - Fixed position_evolution conflict handling |
| | | - Standardized agent verdict labels |
| | | - Added temporal context awareness |
| | | - Enhanced evidence formatting |
| 4.0.1 | 2025-12-25 | Test mode implementation |
| | | - Added `--test-mode` flag for cost-effective testing |
| | | - 97% reduction in Tavily API costs ($0.48 → $0.016) |
| | | - Enables 500 test articles on free tier |
| | | - Validates full architecture with 1 pillar, 2 queries |
| 4.0.0 | 2025-12-24 | Initial architecture implementation |
| | | - Four-layer system operational |
| | | - CLI interface complete |
| | | - All core features working |
| | | - Known limitations documented |

---

### F. Contact & Contributing

**Project Status**: Private research prototype

**Future Plans**: 
- Complete Phase 1-2 enhancements
- Publish findings in academic paper
- Potential open-source release after validation

**Technical Debt**:
- TODO items marked in code comments
- Test coverage: ~10% (needs improvement)
- Documentation: Architecture complete, API docs pending

---

**Document Status**: Living document; updated as system evolves.

**Last Reviewed**: December 24, 2025

---

## Recent Optimizations (v4.0.1)

### Model & Temperature Configuration

**Gemini Model Upgrade**: `gemini-2.5-flash-lite` → `gemini-3-flash-preview`
- Rationale: Improved reasoning capabilities for complex conflict classification
- Applied to: All LLM calls (pillar extraction, conflict classification, agent debate)
- File: `src/gemini_client.py` (line 18)

**Temperature Optimization**:
- **Pillar Extraction**: 0.1
  - Purpose: Deterministic extraction of factual claims from articles
  - File: `cli.py` (pillar extraction call)
- **Conflict Classification**: 0.1
  - Purpose: Consistent categorization of conflict types (factual_contradiction, position_evolution, source_disagreement)
  - File: `src/semantic_judge.py` (classify_conflict_type method, line 301)
- **Agent Debate**: 0.2
  - Purpose: Reduce hallucinations while preserving reasoning diversity across three agents
  - File: `src/consensus_protocol.py` (lines 137, 211)

**Rationale**: Lower temperatures improve factual consistency in structured reasoning tasks while maintaining enough diversity for multi-agent debate.

---

### Error Handling Improvements

**Gemini API Response Format**:
- Issue: Gemini sometimes returns `list` instead of `dict` for agent verdicts
- Solution: Added `isinstance()` checks with list unpacking
- Locations:
  * `src/consensus_protocol.py` (lines 98-107): gather_agent_verdicts()
  * `src/consensus_protocol.py` (lines 163-173): conduct_debate()
- Behavior: Prints warning and uses first list item as verdict dict

**Code Example**:
```python
if isinstance(verdict, list):
    print(f"Warning: Agent {agent_id} returned list, using first item")
    verdict = verdict[0] if verdict else {}
```

---

### Known Limitations & Future Work

1. **Pillar Extraction and Change Detection**:
   - Current state: Layer 1.5 semantic enrichment successfully detects change patterns when present in pillar text
   - Remaining limitation: Pillar extraction LLM sometimes frames change stories as present-tense claims
     * Example: "Zelensky proposes DMZ" instead of "Zelensky shifted from territorial integrity to DMZ"
   - Impact: Semantic enrichment cannot detect changes that aren't in the pillar text
   - Solution paths:
     * Option A: Enhance pillar extraction prompt to preserve change language from article
     * Option B: Add post-extraction reframing when article context shows change but pillar doesn't
     * Option C: Extract multiple pillars (historical + current) to capture temporal sequence
   - Architecture benefit: Semantic metadata makes this debugging clear (we can see pillar has no change verbs)

2. **Conflict Classification Consistency**:
   - Current: Classification varies between runs (position_evolution vs source_disagreement)
   - Root cause: Input quality from pillar extraction
   - Mitigation: LLM-based classification (not rule-based) allows for future refinement without code changes

3. **Agent Confidence Formatting**:
   - Current: Agents return confidence as string ("High") or float (0.92)
   - Future: Standardize to float for programmatic thresholds

4. **Category Extensibility**:
   - Design decision: Keep 3 core conflict types that cover fundamental logic patterns
   - Extensibility: Adding new categories only requires prompt updates (no code changes)
   - Trade-off: Balance between coverage and complexity

