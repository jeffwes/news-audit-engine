from src.semantic_enrichment import SemanticEnrichment
import json

enricher = SemanticEnrichment()

test_pillars = [
    {'text': 'Zelensky shifted his position from total territorial integrity to now proposing a demilitarized zone', 'importance': 5},
    {'text': 'Biden announced the signing of the infrastructure bill on November 15', 'importance': 4},
    {'text': 'Experts disagree on whether the policy will reduce inflation', 'importance': 3}
]

print('Testing Semantic Enrichment:')
print('=' * 60)

for pillar in test_pillars:
    enriched = enricher.enrich_pillar(pillar)
    print(f'\nPillar: {pillar["text"][:60]}...')
    print(f'  Claim Type: {enriched["claim_type"]}')
    print(f'  Has Temporal Shift: {enriched["change_indicators"]["has_temporal_shift"]}')
    print(f'  Change Verbs: {enriched["change_indicators"]["change_verbs"]}')
    if enriched.get('temporal_frame'):
        print(f'  Temporal Frame:')
        for k, v in enriched["temporal_frame"].items():
            print(f'    {k}: {v}')
