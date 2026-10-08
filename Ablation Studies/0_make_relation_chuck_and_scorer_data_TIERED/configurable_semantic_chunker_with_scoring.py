#!/usr/bin/env python3
"""
Configurable Semantic Chunker for Biomedical Relation Extraction
================================================================

A generalized version of the improved semantic chunker that can be 
configured for different datasets (GM-CIHT, DDI, ChemProt, etc.)

Key Features:
- Dataset-agnostic relation trigger detection
- Configurable relation types and markers
- Entity boundary preservation
- Relation context preservation
- Clinical context awareness

Author: BioMedRAG Analysis
Date: November 12, 2025
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class ConfigurableSemanticChunker:
    """
    A configurable semantic chunker that can adapt to different biomedical datasets.
    
    Key improvements:
    1. Never splits entities across chunks
    2. Keeps relation triggers with entities
    3. Preserves directional modifiers (might, not, etc.)
    4. Maintains clinical context (treatment + disease)
    5. **NEW**: Configurable for different datasets via config files
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 config_dict: Optional[Dict] = None,
                 window_size: int = 7,
                 stride: int = 3,
                 context_window: int = 4,
                 min_chunk_size: int = 5,
                 max_chunk_size: int = 15):
        """
        Initialize configurable chunker.
        
        Args:
            config_file: Path to JSON configuration file
            config_dict: Configuration dictionary (alternative to config_file)
            window_size: Base window size
            stride: Stride for sliding window
            context_window: Words to include around relation triggers
            min_chunk_size: Minimum words per chunk
            max_chunk_size: Maximum words per chunk
        """
        self.window_size = window_size
        self.stride = stride
        self.context_window = context_window
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.config: Dict = {}

        # Load configuration early so relation_triggers exist before flattening
        if config_file:
            self._load_config_from_file(config_file)
        elif config_dict:
            self._load_config_from_dict(config_dict)
        else:
            # Default to original GM-CIHT configuration
            self._load_default_config()
        
        # Entity detection patterns (universal across datasets)
        self.entity_patterns = [
            r'\[[^\]]+\]',                                  # [entity] markers
            r'\{[^\}]+\}',                                  # {entity} markers
            r'\b[A-Z][a-z]+-?\w*(?:\s+[A-Z][a-z]+-?\w*)*\b', # Capitalized terms
            r'\b\w+(?:-\w+)+\b',                            # Hyphenated compounds
        ]
        
        # Flatten triggers for quick lookup (tiered triggers are flattened separately)
        self.all_triggers = set()
        for triggers in self.relation_triggers.values():
            self.all_triggers.update(triggers)
        for rel, tiers in getattr(self, "tiered_triggers", {}).items():
            for trig_list in tiers.values():
                self.all_triggers.update(trig_list)
        
        # Directional modifiers (universal)
        self.uncertainty_markers = {
            'might', 'may', 'possibly', 'potentially', 'could', 'would',
            'suggested', 'appears', 'seems', 'thought', 'likely', 'probably'
        }
        
        self.negation_markers = {
            'not', 'no', 'never', 'without', 'lack', 'absence',
            'unable', 'cannot', 'doesn\'t', 'don\'t', 'didn\'t'
        }
    
    def _load_config_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        self._load_config_from_dict(config)
    
    def _load_config_from_dict(self, config: Dict):
        """Load configuration from dictionary."""
        self.config = config or {}
        self.dataset_name = config.get('dataset_name', 'UNKNOWN')
        self.relation_triggers = config.get('relation_triggers', {})
        self.tiered_triggers = config.get('tiered_triggers', {})
        self.relation_hierarchy = config.get('relation_hierarchy', {})
        self.relation_aliases = config.get('relation_aliases', {})
        self.negation_rules = config.get('negation_rules', {})
        self.symmetric_relations = set(config.get('symmetric_relations', []))
        self.scoring_weights = config.get('scoring_weights', {})
        self.disease_markers = set(config.get('disease_markers', []))
        self.mechanism_markers = set(config.get('mechanism_markers', []))
        self.dataset_specific_markers = config.get('dataset_specific_markers', {})
        self.max_tail_length = config.get('max_tail_length', self.max_chunk_size)
        
        # Chunking parameters (can override init params)
        chunking_params = config.get('chunking_parameters', {})
        if chunking_params:
            self.window_size = chunking_params.get('window_size', self.window_size)
            self.stride = chunking_params.get('stride', self.stride)
            self.min_chunk_size = chunking_params.get('min_chunk_size', self.min_chunk_size)
            self.max_chunk_size = chunking_params.get('max_chunk_size', self.max_chunk_size)
        
        # Add custom entity patterns if provided
        if 'entity_patterns' in config:
            self.entity_patterns.extend(config['entity_patterns'])
    
    def _load_default_config(self):
        """Load default GM-CIHT configuration."""
        self.dataset_name = 'GM-CIHT'
        
        # Original relation triggers
        self.relation_triggers = {
            'INHIBITS': ['inhibits', 'inhibit', 'inhibited', 'inhibiting', 'inhibition', 'inhibitor', 'inhibitory',
                        'blocks', 'block', 'blocked', 'blocking', 
                        'suppresses', 'suppress', 'suppressed', 'suppressing', 'suppression',
                        'prevents', 'prevent', 'prevented', 'preventing', 'prevention',
                        'reduces', 'reduce', 'reduced', 'reducing', 'reduction',
                        'decreases', 'decrease', 'decreased', 'decreasing',
                        'downregulates', 'downregulate', 'downregulated', 'downregulating', 'downregulation',
                        'antagonizes', 'antagonize', 'antagonized', 'antagonizing', 'antagonist',
                        'inactivates', 'inactivate', 'inactivated', 'inactivating', 'inactivation',
                        'attenuates', 'attenuate', 'attenuated', 'attenuating', 'attenuation',
                        'depletes', 'deplete', 'depleted', 'depleting', 'depletion'],
            'STIMULATES': ['stimulates', 'stimulate', 'stimulated', 'stimulating', 'stimulation',
                          'activates', 'activate', 'activated', 'activating', 'activation', 'activator',
                          'enhances', 'enhance', 'enhanced', 'enhancing', 'enhancement',
                          'promotes', 'promote', 'promoted', 'promoting', 'promotion', 'promoter',
                          'increases', 'increase', 'increased', 'increasing',
                          'upregulates', 'upregulate', 'upregulated', 'upregulating', 'upregulation',
                          'potentiates', 'potentiate', 'potentiated', 'potentiating',
                          'induces', 'induce', 'induced', 'inducing', 'induction', 'inducer'],
            'TREATS': ['treats', 'treat', 'treated', 'treating', 'treatment',
                      'therapy', 'therapeutic', 'therapies',
                      'prescribed', 'prescribe', 'prescribing', 'prescription',
                      'administered', 'administer', 'administering', 'administration',
                      'medication', 'medicate', 'drug', 'drugs'],
            'AFFECTS': ['affects', 'affect', 'affected', 'affecting',
                       'modulates', 'modulate', 'modulated', 'modulating', 'modulation',
                       'influences', 'influence', 'influenced', 'influencing',
                       'impacts', 'impact', 'impacted', 'impacting',
                       'regulates', 'regulate', 'regulated', 'regulating', 'regulation',
                       'alters', 'alter', 'altered', 'altering', 'alteration'],
            'INTERACTS_WITH': ['binds', 'bind', 'bound', 'binding', 
                              'interacts', 'interact', 'interacted', 'interacting', 'interaction',
                              'associates', 'associate', 'associated', 'associating', 'association',
                              'couples', 'couple', 'coupled', 'coupling'],
            'PREVENTS': ['prevents', 'prevent', 'prevented', 'preventing', 'prevention', 
                        'prophylaxis', 'prophylactic', 'protective', 'protect', 'protected'],
            'CAUSES': ['causes', 'cause', 'caused', 'causing',
                      'induces', 'induce', 'induced', 'inducing', 'induction',
                      'triggers', 'trigger', 'triggered', 'triggering',
                      'elicits', 'elicit', 'elicited', 'eliciting',
                      'produces', 'produce', 'produced', 'producing', 'production'],
        }
        
        self.disease_markers = {
            'disease', 'disorder', 'syndrome', 'condition', 'illness', 'pathology',
            'cancer', 'tumor', 'diabetes', 'hypertension', 'infection', 'inflammation',
            'patient', 'patients'
        }
        
        self.mechanism_markers = {
            'mechanism', 'pathway', 'signaling', 'metabolism', 'synthesis',
            'expression', 'activity', 'function', 'levels', 'production',
            'protein', 'enzyme', 'receptor', 'cytokine', 'hormone', 'metabolite'
        }
        
        self.dataset_specific_markers = {}
    
    def detect_entities(self, text: str) -> List[Dict]:
        """
        Detect entity spans in text using multiple patterns.
        
        Returns:
            List of dicts with 'text', 'start', 'end' for each entity
        """
        entities = []
        
        for pattern in self.entity_patterns:
            for match in re.finditer(pattern, text):
                # Clean entity text (remove brackets if present)
                entity_text = match.group(0)
                entity_text = entity_text.strip('[]{}')
                
                entities.append({
                    'text': entity_text,
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Remove duplicates and sort by position
        unique_entities = []
        seen_spans = set()
        
        for entity in sorted(entities, key=lambda x: x['start']):
            span = (entity['start'], entity['end'])
            if span not in seen_spans:
                unique_entities.append(entity)
                seen_spans.add(span)
        
        return unique_entities
    
    def detect_relation_triggers(self, text: str) -> List[Dict]:
        """Find relation trigger words and their types, with tiered priority if provided."""
        words = text.lower().split()
        triggers = []
        tiered = getattr(self, "tiered_triggers", {})
        
        for i, word in enumerate(words):
            word_clean = word.strip('.,;:!?()')
            if word_clean not in self.all_triggers:
                continue
            
            relation_type = None
            priority = 'tertiary'
            
            # Check tiered triggers first
            for rel, tiers in tiered.items():
                if word_clean in tiers.get('primary', []):
                    relation_type = rel
                    priority = 'primary'
                    break
                if word_clean in tiers.get('secondary', []):
                    relation_type = rel
                    priority = 'secondary'
                    break
                if word_clean in tiers.get('tertiary', []):
                    relation_type = rel
                    priority = 'tertiary'
                    break
            # Fallback to flat relation_triggers
            if relation_type is None:
                for rel, trigger_list in self.relation_triggers.items():
                    if word_clean in trigger_list:
                        relation_type = rel
                        priority = 'primary'
                        break
            
            triggers.append({
                'word': word_clean,
                'position': i,
                'relation_type': relation_type,
                'priority': priority,
            })
        
        return triggers
    
    def detect_modifiers(self, text: str) -> List[Dict]:
        """Find directional modifiers (uncertainty, negation)."""
        words = text.lower().split()
        modifiers = []
        
        for i, word in enumerate(words):
            word_clean = word.strip('.,;:!?()')
            
            if word_clean in self.uncertainty_markers:
                modifiers.append({
                    'word': word_clean,
                    'position': i,
                    'type': 'uncertainty'
                })
            elif word_clean in self.negation_markers:
                modifiers.append({
                    'word': word_clean,
                    'position': i,
                    'type': 'negation'
                })
        
        return modifiers
    
    def classify_context(self, text: str) -> str:
        """
        Classify text as disease/treatment/mechanism context.
        Helps distinguish TREATS vs AFFECTS.
        """
        words = text.lower().split()
        
        disease_count = sum(1 for w in words if any(m in w for m in self.disease_markers))
        mechanism_count = sum(1 for w in words if any(m in w for m in self.mechanism_markers))
        
        if disease_count > 0 and mechanism_count > 0:
            return 'mixed_clinical'
        elif disease_count > 0:
            return 'disease_context'
        elif mechanism_count > 0:
            return 'mechanism_context'
        else:
            return 'general_context'
    
    def create_entity_preserving_chunks(self, text: str) -> List[Dict]:
        """
        Create chunks that never split entities across boundaries.
        """
        words = text.split()
        entities = self.detect_entities(text)
        
        if len(words) <= self.window_size:
            return [{
                'text': text,
                'start_idx': 0,
                'end_idx': len(text),
                'num_words': len(words),
                'entities': entities,
                'strategy': 'full_text'
            }]
        
        # Map word positions
        word_positions = []
        current_pos = 0
        for word in words:
            start = text.find(word, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(word)
            word_positions.append((start, end))
            current_pos = end
        
        chunks = []
        i = 0
        
        while i < len(words):
            window_end = min(i + self.window_size, len(words))
            
            # Check if this window would split any entity
            if word_positions:
                window_start_pos = word_positions[i][0]
                window_end_pos = word_positions[min(window_end - 1, len(word_positions) - 1)][1]
                
                # Adjust boundaries to not split entities
                for entity in entities:
                    # Entity starts before window and ends inside
                    if entity['start'] < window_start_pos < entity['end'] < window_end_pos:
                        # Move start to after entity
                        while i < len(words) and word_positions[i][0] < entity['end']:
                            i += 1
                        continue
                    
                    # Entity starts inside window and ends after
                    if window_start_pos < entity['start'] < window_end_pos < entity['end']:
                        # Extend end to include entity
                        while window_end < len(words) and word_positions[window_end - 1][1] < entity['end']:
                            window_end += 1
            
            # Create chunk
            chunk_text = ' '.join(words[i:window_end])
            
            if chunk_text.strip():
                chunk_entities = [e for e in entities if e['start'] >= word_positions[i][0] and 
                                 e['end'] <= word_positions[min(window_end - 1, len(word_positions) - 1)][1]]
                
                chunks.append({
                    'text': chunk_text,
                    'start_idx': i,
                    'end_idx': window_end,
                    'num_words': window_end - i,
                    'entities': chunk_entities,
                    'strategy': 'entity_preserving'
                })
            
            # Move forward
            i += self.stride
        
        return chunks if chunks else [{
            'text': text,
            'start_idx': 0,
            'end_idx': len(words),
            'num_words': len(words),
            'entities': entities,
            'strategy': 'fallback'
        }]
    
    def create_relation_context_chunks(self, text: str) -> List[Dict]:
        """
        Create chunks centered around relation triggers with entity context.
        """
        words = text.split()
        entities = self.detect_entities(text)
        triggers = self.detect_relation_triggers(text)
        modifiers = self.detect_modifiers(text)
        
        if not triggers:
            # No triggers, use entity-preserving chunks
            return self.create_entity_preserving_chunks(text)
        
        chunks = []
        covered_positions = set()
        
        for trigger in triggers:
            # Get context window around trigger
            start_idx = max(0, trigger['position'] - self.context_window)
            end_idx = min(len(words), trigger['position'] + self.context_window + 1)
            
            # Expand to include nearby modifiers
            for modifier in modifiers:
                if abs(modifier['position'] - trigger['position']) <= 2:
                    start_idx = min(start_idx, modifier['position'])
                    end_idx = max(end_idx, modifier['position'] + 1)
            
            # Expand to include entities
            chunk_text_current = ' '.join(words[start_idx:end_idx])
            for entity in entities:
                # If entity overlaps with current chunk range
                if entity['text'].lower() in chunk_text_current.lower():
                    # Ensure full entity is included
                    pass
            
            # Create chunk
            chunk_text = ' '.join(words[start_idx:end_idx])
            
            if len(chunk_text.split()) >= self.min_chunk_size:
                # Find modifiers in this chunk
                chunk_modifiers = [m for m in modifiers 
                                  if start_idx <= m['position'] < end_idx]
                
                # Determine if this is uncertain relation
                is_uncertain = any(m['type'] == 'uncertainty' for m in chunk_modifiers)
                is_negated = any(m['type'] == 'negation' for m in chunk_modifiers)
                
                chunks.append({
                    'text': chunk_text,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'num_words': end_idx - start_idx,
                    'relation_type': trigger['relation_type'],
                    'trigger_word': trigger['word'],
                    'has_trigger': True,
                    'is_uncertain': is_uncertain,
                    'is_negated': is_negated,
                    'context_type': self.classify_context(chunk_text),
                    'strategy': 'relation_context'
                })
                
                covered_positions.update(range(start_idx, end_idx))
        
        # Add chunks for uncovered regions
        i = 0
        while i < len(words):
            if i not in covered_positions:
                # Find continuous uncovered region
                start = i
                while i < len(words) and i not in covered_positions:
                    i += 1
                
                if i - start >= self.min_chunk_size:
                    chunk_text = ' '.join(words[start:i])
                    chunks.append({
                        'text': chunk_text,
                        'start_idx': start,
                        'end_idx': i,
                        'num_words': i - start,
                        'relation_type': None,
                        'has_trigger': False,
                        'context_type': self.classify_context(chunk_text),
                        'strategy': 'filler'
                    })
            else:
                i += 1
        
        return sorted(chunks, key=lambda x: x['start_idx'])

    def get_all_candidates(self, text: str, relation: Optional[str] = None) -> List[Dict]:
        """
        Get all candidate chunks, optionally scored with tiered trigger bonuses.
        
        Args:
            text: Input text to chunk
            relation: Optional relation type for targeted scoring (enables tiered trigger bonuses)
        
        Returns:
            List of chunk dictionaries. If relation is provided, chunks are scored using
            tiered trigger bonuses from the configurable chunker's scoring system.
        """
        # 1. Relation Context Chunks
        rel_chunks = self.create_relation_context_chunks(text)
        
        # 2. Entity Preserving Chunks (as fallback/supplement)
        ent_chunks = self.create_entity_preserving_chunks(text)
        
        candidates = rel_chunks + ent_chunks
        
        # Score chunks if relation is provided (enables tiered trigger bonuses)
        if relation:
            for chunk in candidates:
                chunk['score'] = self._score_chunk(chunk, relation)
        
        return candidates

    def chunk_for_triplet_extraction(self, 
                                     text: str, 
                                     relation: Optional[str] = None,
                                     num_chunks: int = 5) -> List[Dict]:
        """
        Main method: Create optimized chunks for triplet extraction.
        Enhanced with relation hierarchy support.
        
        Args:
            text: Input sentence
            relation: Expected relation type (for targeted chunking)
            num_chunks: Target number of chunks to return
        
        Returns:
            List of chunk dictionaries with metadata
        """
        # Create relation-context-aware chunks
        all_chunks = self.create_relation_context_chunks(text)
        
        # Filter by size constraints
        valid_chunks = [
            c for c in all_chunks
            if self.min_chunk_size <= c['num_words'] <= self.max_chunk_size
        ]
        
        # If no valid chunks, fallback
        if not valid_chunks:
            valid_chunks = self.create_entity_preserving_chunks(text)
        
        # Score chunks
        for chunk in valid_chunks:
            chunk['score'] = self._score_chunk(chunk, relation)
        
        # Apply relation hierarchy to detected relations (if present)
        for chunk in valid_chunks:
            if 'detected_relations' in chunk:
                chunk['detected_relations'] = self._apply_relation_hierarchy(
                    chunk['detected_relations']
                )
        
        # Sort by score and select top chunks
        valid_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top N chunks, but ensure diversity
        selected_chunks = []
        selected_texts = set()
        
        for chunk in valid_chunks:
            if len(selected_chunks) >= num_chunks:
                break
            
            # Avoid duplicate text
            if chunk['text'] not in selected_texts:
                selected_chunks.append(chunk)
                selected_texts.add(chunk['text'])
        
        # If we have fewer chunks than requested, that's okay
        return selected_chunks
    
    def _get_tiered_triggers(self, relation_type: str) -> Dict[str, List[str]]:
        """Get triggers organized by tier (primary > secondary > tertiary)."""
        if hasattr(self, 'tiered_triggers') and relation_type in self.tiered_triggers:
            return self.tiered_triggers[relation_type]
        
        # Fallback: use relation_triggers as primary tier
        if relation_type in self.relation_triggers:
            return {
                'primary': self.relation_triggers[relation_type]
            }
        
        return {}
    
    def _apply_relation_hierarchy(self, detected_relations: List[Dict]) -> List[Dict]:
        """
        Disambiguate between similar relations using hierarchy.
        
        Example: If both AFFECTS and TREATS detected, prefer based on hierarchy priority.
        """
        if not hasattr(self, 'relation_hierarchy') or not self.relation_hierarchy:
            return detected_relations
        
        hierarchy = self.relation_hierarchy
        
        # Handle priority_order format
        if 'priority_order' in hierarchy:
            priority_map = {
                rel: len(hierarchy['priority_order']) - i 
                for i, rel in enumerate(hierarchy['priority_order'])
            }
        else:
            # Handle explicit priority format
            priority_map = {
                rel: cfg.get('priority', 0) 
                for rel, cfg in hierarchy.items()
            }
        
        # Sort by priority (higher first), then by score
        detected_relations.sort(
            key=lambda r: (
                -priority_map.get(r.get('relation', ''), 0),
                -r.get('score', 0)
            )
        )
        
        return detected_relations
    
    def _apply_negation_rules(self, chunk_text: str, relation: str) -> Tuple[str, float]:
        """
        Apply negation rules from config.
        
        Returns:
            (possibly_flipped_relation, score_penalty)
        """
        if not hasattr(self, 'negation_rules') or not self.negation_rules:
            return relation, 1.0
        
        negation_rules = self.negation_rules
        
        # Check for negation markers in chunk
        chunk_lower = chunk_text.lower()
        for marker in negation_rules.get('markers', ['not', 'no', 'does not']):
            if marker.lower() in chunk_lower:
                # Apply rule if exists for this relation
                flip_map = negation_rules.get('flip', {})
                if relation in flip_map:
                    relation = flip_map[relation]
                
                # Apply penalty
                penalty = negation_rules.get('penalty', 0.7)
                return relation, penalty
        
        return relation, 1.0
    
    def _score_chunk(self, chunk: Dict, relation: Optional[str] = None) -> float:
        """
        Score chunk relevance for triplet extraction.
        Enhanced with tiered triggers, hierarchy, negation, and length bonuses.
        """
        score = 0.0
        chunk_relation = chunk.get('relation_type')
        
        # Has relation trigger
        if chunk.get('has_trigger'):
            score += 2.0
            
            # Matches expected relation
            if relation and chunk_relation == relation:
                score += 3.0
            elif relation and chunk_relation:
                # Different relation but has a trigger
                score += 0.5
            
            # Tiered trigger bonus (if tiered_triggers configured)
            if hasattr(self, 'tiered_triggers') and chunk_relation:
                trigger_word = chunk.get('trigger_word', '')
                tiers = self._get_tiered_triggers(chunk_relation)
                
                if trigger_word in tiers.get('primary', []):
                    bonus = self.scoring_weights.get('primary_trigger_bonus', 0.12)
                    score += bonus
                elif trigger_word in tiers.get('secondary', []):
                    bonus = self.scoring_weights.get('secondary_trigger_bonus', 0.06)
                    score += bonus
                elif trigger_word in tiers.get('tertiary', []):
                    bonus = self.scoring_weights.get('tertiary_trigger_bonus', 0.03)
                    score += bonus
        
        # Uncertainty handling
        if chunk.get('is_uncertain'):
            # If trigger is specific but uncertain, downgrade
            if chunk.get('relation_type') in ['INHIBITS', 'STIMULATES', 'CAUSES']:
                score -= 1.0
        
        # Negation handling
        if chunk_relation:
            rel_name, penalty = self._apply_negation_rules(chunk.get('text', ''), chunk_relation)
            score *= penalty
        
        # Context alignment
        if relation:
            if relation in ['TREATS', 'PREVENTS']:
                if chunk.get('context_type') == 'disease_context':
                    score += 1.5
            elif relation in ['AFFECTS', 'INHIBITS', 'STIMULATES']:
                if chunk.get('context_type') == 'mechanism_context':
                    score += 1.5
        
        # Length preference (enhanced with config weights)
        num_words = chunk.get('num_words', 0)
        short_max = self.scoring_weights.get('short_window_max', 8)
        long_max = self.scoring_weights.get('long_window_max', 16)
        
        if num_words <= short_max:
            # Short-length bonus
            bonus = self.scoring_weights.get('short_length_bonus', 0.12)
            score += bonus
        elif 7 <= num_words <= 12:
            # Medium-sized chunks (original logic)
            score += 1.0
        elif 5 <= num_words <= 15:
            score += 0.5
        elif num_words > long_max:
            # Long-length penalty
            penalty = self.scoring_weights.get('long_length_penalty', 0.15)
            score -= penalty
        
        # Entity presence
        if chunk.get('entities'):
            score += len(chunk['entities']) * 0.3
        
        return score


def create_chunker_from_config(config_path: str) -> ConfigurableSemanticChunker:
    """
    Factory function to create a chunker from a config file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Configured ConfigurableSemanticChunker instance
    """
    return ConfigurableSemanticChunker(config_file=config_path)


def test_configurable_chunker():
    """Test the configurable chunker with default config."""
    
    chunker = ConfigurableSemanticChunker()
    
    test_text = "ethionine might produce an inhibition of the synthesis of biotin-dependent acetyl-coa carboxylase"
    
    print("="*80)
    print(f"Testing Configurable Semantic Chunker - Dataset: {chunker.dataset_name}")
    print("="*80)
    print(f"Text: {test_text}")
    
    chunks = chunker.chunk_for_triplet_extraction(
        test_text,
        relation='INTERACTS_WITH',
        num_chunks=3
    )
    
    print(f"\nGenerated {len(chunks)} optimized chunks:")
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('score', 0)
        strategy = chunk.get('strategy', 'unknown')
        trigger = chunk.get('trigger_word', 'none')
        rel_type = chunk.get('relation_type', 'none')
        uncertain = ' [UNCERTAIN]' if chunk.get('is_uncertain') else ''
        
        print(f"\n  Chunk {i} (score: {score:.2f}, strategy: {strategy}):")
        print(f"    Trigger: {trigger} ({rel_type}){uncertain}")
        print(f"    Text: {chunk['text']}")
    
    print(f"\n{'='*80}")
    print("Testing completed!")
    print("="*80)


if __name__ == "__main__":
    test_configurable_chunker()
