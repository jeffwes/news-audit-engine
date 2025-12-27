# Semantic Enrichment Implementation Summary

## What Was Built

### New Module: `src/semantic_enrichment.py`
A linguistic analysis pipeline that enriches narrative pillars with machine-readable metadata:

**Detected Features:**
1. **Change Verbs**: shift, reverse, abandon, adopt, change, switch, pivot, etc.
2. **Temporal Markers**: from/to, previously/now, used to, no longer, etc.
3. **Temporal Patterns**: "from X to Y", "previously X, now Y", negations
4. **Claim Classification**: 
   - `position_evolution`: Entity changed stance over time
   - `position_statement`: Current stance/opinion
   - `factual_event`: Verifiable event
   - `quote_attribution`: Reported speech
5. **Temporal Frame Extraction**: old_position → new_position + transition_verb

### Pipeline Integration

**Layer 1.5 - Semantic Enrichment** (added between NER and search):
```
Pillar Extraction → NER → SEMANTIC ENRICHMENT → Search → Conflict Detection
```

**Enriched Pillar Structure:**
```python
{
    "text": "Zelensky shifted from X to Y",
    "importance": 5,
    "entities": [...],  # from NER
    "claim_type": "position_evolution",  # NEW
    "change_indicators": {  # NEW
        "has_temporal_shift": True,
        "change_verbs": ["shift"],
        "temporal_markers": ["from", "to"],
        "has_from_to_pattern": True
    },
    "temporal_frame": {  # NEW
        "old_position": "territorial integrity",
        "new_position": "DMZ proposal",
        "transition_verb": "shift"
    }
}
```

### Enhanced Conflict Classification

**Before:**
- Only had pillar text + conflict counts
- Classification based on text analysis alone

**After:**
- Has explicit semantic signals
- Classification prompt includes:
  ```
  SEMANTIC METADATA:
  - Claim Type: position_evolution
  - Has Temporal Shift: True
  - Change Verbs Detected: ['shift']
  - Temporal Markers: ['from', 'to']
  - Old Position: territorial integrity
  - New Position: DMZ proposal
  ```
- **Critical Rules** added:
  1. If `claim_type='position_evolution'` → classify as position_evolution
  2. If change verbs present → classify as position_evolution  
  3. If temporal_frame exists → classify as position_evolution

## Test Results

### Enrichment Test (Successful ✓)
```
Pillar: "Zelensky shifted his position from total territorial integrity to..."
  Claim Type: position_evolution  ✓
  Has Temporal Shift: True  ✓
  Change Verbs: ['shift']  ✓
```

### Full Pipeline Test (Reveals Upstream Issue)
```
Extracted Pillar: "President Zelensky proposes converting territories..."
  Claim Type: factual_event  ← PROBLEM HERE
  Has Temporal Shift: False
  Change Verbs: []
  
Classification: factual_contradiction  ← Correct based on pillar
Verdict: factual_contradiction (100% consensus)
```

## Root Cause Identified

**The semantic enrichment is working correctly** - it's accurately analyzing what's in the pillar.

**The problem is upstream**: The pillar extraction LLM is framing the story as:
- ❌ "Zelensky proposes DMZ" (present-tense factual claim)
  
Instead of:
- ✓ "Zelensky shifted from territorial integrity to DMZ proposal" (change story)

## Next Steps

### Option 1: Enhance Pillar Extraction Prompt
Add instructions to capture change/transition language:
```
"If the article describes an entity CHANGING position, policy, or stance:
- Frame the pillar to explicitly mention the transition
- Use change verbs: shifted, reversed, abandoned, adopted
- Structure: '[Entity] shifted from [old position] to [new position]'
- Example: 'Zelensky shifted from demanding territorial integrity to proposing DMZ'"
```

### Option 2: Add Post-Extraction Reframing
After enrichment detects change indicators in the article context but NOT in the pillar:
- Use LLM to reframe the pillar to capture the change
- Input: original pillar + article context + detected change signals
- Output: reframed pillar that explicitly states the transition

### Option 3: Multi-Pillar Synthesis
Extract both the change claim AND the current position as separate pillars:
- Pillar 1: "Zelensky previously demanded territorial integrity"
- Pillar 2: "Zelensky now proposes DMZ in Donetsk"
- Let the semantic enrichment link them as a temporal sequence

## Architecture Improvement

The semantic enrichment layer successfully:
- ✅ Provides machine-readable metadata to downstream components
- ✅ Detects change patterns accurately when they exist in text
- ✅ Classifies claim types correctly
- ✅ Gives explicit signals to conflict classification
- ✅ Makes the system debuggable (we can see WHY classification happened)

**Impact**: The system now has explicit linguistic features instead of relying on LLM interpretation alone. This makes it:
- More consistent (rule-based change detection)
- More explainable (semantic metadata in output)
- More extensible (add new change patterns without retraining)
- More debuggable (can see what features were detected)

## Files Modified

1. **Created**: `src/semantic_enrichment.py` (230 lines)
2. **Modified**: `cli.py` - Added enrichment step in Layer 1
3. **Modified**: `src/semantic_judge.py` - Enhanced conflict classification with metadata
