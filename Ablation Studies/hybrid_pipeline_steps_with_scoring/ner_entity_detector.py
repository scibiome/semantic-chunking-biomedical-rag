"""
NER-based Entity Detector for Configurable Semantic Chunker

This module provides a wrapper to use pre-trained NER systems (spaCy, transformers, etc.)
with the configurable semantic chunker, replacing the regex-based entity detection.

Usage:
    from ner_entity_detector import NEREntityDetector
    
    # Option 1: Use HuggingFace transformers NER model
    ner_detector = NEREntityDetector(ner_model='d4data/biomedical-ner-all')
    
    # Option 2: Use spaCy model
    ner_detector = NEREntityDetector(ner_model='en_core_sci_sm', use_spacy=True)
    
    # Use with configurable chunker
    chunker = ConfigurableSemanticChunker(...)
    chunker.detect_entities = ner_detector.detect_entities  # Replace method
"""

from typing import List, Dict, Optional, Callable
import re


class NEREntityDetector:
    """
    Wrapper for pre-trained NER systems to use with ConfigurableSemanticChunker.
    
    Supports:
    - HuggingFace transformers NER models (e.g., 'd4data/biomedical-ner-all')
    - spaCy models (e.g., 'en_core_sci_sm', 'en_core_web_sm')
    - Custom NER functions
    """
    
    def __init__(
        self,
        ner_model: Optional[str] = None,
        use_spacy: bool = False,
        custom_ner_function: Optional[Callable[[str], List[Dict]]] = None,
        confidence_threshold: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize NER entity detector.
        
        Args:
            ner_model: Model identifier (HuggingFace model ID or spaCy model name)
            use_spacy: If True, use spaCy instead of transformers
            custom_ner_function: Custom function(text) -> List[Dict] with 'text', 'start', 'end'
            confidence_threshold: Minimum confidence score for entities
            device: Device for transformers models ('cpu' or 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.ner_pipeline = None
        self.custom_ner = custom_ner_function
        
        if custom_ner_function:
            print(f"✅ Using custom NER function")
        elif use_spacy:
            self._init_spacy(ner_model)
        else:
            self._init_transformers(ner_model)
    
    def _init_spacy(self, model_name: Optional[str]):
        """Initialize spaCy NER model."""
        try:
            import spacy
            if model_name:
                print(f"Loading spaCy model: {model_name}...")
                self.ner_pipeline = spacy.load(model_name)
            else:
                # Try default biomedical model
                try:
                    self.ner_pipeline = spacy.load('en_core_sci_sm')
                    print("✅ Loaded default spaCy biomedical model: en_core_sci_sm")
                except:
                    self.ner_pipeline = spacy.load('en_core_web_sm')
                    print("✅ Loaded default spaCy model: en_core_web_sm")
            print("✅ spaCy NER ready!")
        except ImportError:
            print("❌ Error: spaCy not installed. Install with: pip install spacy")
            print("   Then download model: python -m spacy download en_core_sci_sm")
            self.ner_pipeline = None
        except Exception as e:
            print(f"❌ Error loading spaCy model: {e}")
            self.ner_pipeline = None
    
    def _init_transformers(self, model_name: Optional[str]):
        """Initialize HuggingFace transformers NER pipeline."""
        try:
            from transformers import pipeline
            import time
            
            if not model_name:
                model_name = 'd4data/biomedical-ner-all'  # Default biomedical NER
            
            print(f"Loading transformers NER model: {model_name}...")
            
            # Try loading with retries for network issues
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    # Set environment to prefer local cache
                    import os
                    os.environ['HF_HUB_OFFLINE'] = '0'  # Allow online but prefer cache
                    
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=model_name,
                        aggregation_strategy="simple",
                        device=0 if self.device == 'cuda' else -1,
                        # Use local_files_only=False to allow download if needed
                        # but transformers will use cache if available
                    )
                    print(f"✅ Transformers NER ready! (model: {model_name})")
                    return  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if 'network' in error_str or 'resolve' in error_str or 'connection' in error_str:
                        if attempt < max_retries - 1:
                            print(f"⚠️  Network error (attempt {attempt + 1}/{max_retries}): {e}")
                            print(f"   Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"❌ Failed to download model after {max_retries} attempts")
                            print(f"   Error: {e}")
                            print("   Will use cached model if available, otherwise fallback to regex")
                            # Try one more time with local_files_only=True to use cache only
                            try:
                                self.ner_pipeline = pipeline(
                                    "ner",
                                    model=model_name,
                                    aggregation_strategy="simple",
                                    device=0 if self.device == 'cuda' else -1,
                                    local_files_only=True  # Use cache only
                                )
                                print(f"✅ Loaded NER model from cache!")
                                return
                            except:
                                self.ner_pipeline = None
                                return
                    else:
                        # Non-network error, don't retry
                        raise
            
        except ImportError:
            print("❌ Error: transformers not installed. Install with: pip install transformers")
            self.ner_pipeline = None
        except Exception as e:
            print(f"❌ Error loading transformers model: {e}")
            self.ner_pipeline = None
    
    def detect_entities(self, text: str) -> List[Dict]:
        """
        Detect entities in text using the configured NER system.
        
        Returns:
            List of dicts with 'text', 'start', 'end' for each entity
            (compatible with ConfigurableSemanticChunker interface)
        """
        if self.custom_ner:
            return self.custom_ner(text)
        
        if not self.ner_pipeline:
            # Fallback to regex patterns if NER not available
            return self._fallback_regex_detection(text)
        
        try:
            if hasattr(self.ner_pipeline, 'pipe'):  # spaCy
                return self._detect_entities_spacy(text)
            else:  # transformers pipeline
                return self._detect_entities_transformers(text)
        except Exception as e:
            print(f"Warning: NER failed for text, using fallback: {e}")
            return self._fallback_regex_detection(text)
    
    def _detect_entities_spacy(self, text: str) -> List[Dict]:
        """Detect entities using spaCy."""
        doc = self.ner_pipeline(text)
        entities = []
        
        for ent in doc.ents:
            # Filter by confidence if available (spaCy doesn't always provide scores)
            entities.append({
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'type': ent.label_,  # Optional: entity type
                'score': getattr(ent, 'score', 1.0)  # May not be available
            })
        
        return entities
    
    def _detect_entities_transformers(self, text: str) -> List[Dict]:
        """Detect entities using transformers pipeline."""
        entities = self.ner_pipeline(text)
        
        result = []
        for entity in entities:
            score = entity.get('score', 1.0)
            if score >= self.confidence_threshold:
                result.append({
                    'text': entity['word'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'type': entity.get('entity_group', entity.get('label', 'ENTITY')),
                    'score': score
                })
        
        return result
    
    def _fallback_regex_detection(self, text: str) -> List[Dict]:
        """Fallback to basic regex patterns if NER unavailable."""
        entities = []
        patterns = [
            (r'\[[^\]]+\]', True),  # [entity] markers
            (r'\{[^\}]+\}', True),  # {entity} markers
            (r'\b[A-Z][a-z]+-?\w*(?:\s+[A-Z][a-z]+-?\w*)*\b', False),  # Capitalized terms
        ]
        
        for pattern, strip_brackets in patterns:
            for match in re.finditer(pattern, text):
                entity_text = match.group(0)
                if strip_brackets:
                    entity_text = entity_text.strip('[]{}')
                
                entities.append({
                    'text': entity_text,
                    'start': match.start(),
                    'end': match.end()
                })
        
        # Remove duplicates
        unique_entities = []
        seen_spans = set()
        for entity in sorted(entities, key=lambda x: x['start']):
            span = (entity['start'], entity['end'])
            if span not in seen_spans:
                unique_entities.append(entity)
                seen_spans.add(span)
        
        return unique_entities


# Convenience function for common biomedical NER models
def create_biomedical_ner_detector(model_name: Optional[str] = None) -> NEREntityDetector:
    """
    Create a NER detector optimized for biomedical text.
    
    Args:
        model_name: Optional model name. Defaults to 'd4data/biomedical-ner-all'
    
    Returns:
        NEREntityDetector instance
    """
    if model_name is None:
        model_name = 'd4data/biomedical-ner-all'
    
    return NEREntityDetector(ner_model=model_name, confidence_threshold=0.5)


def create_spacy_ner_detector(model_name: Optional[str] = None) -> NEREntityDetector:
    """
    Create a spaCy-based NER detector.
    
    Args:
        model_name: spaCy model name (e.g., 'en_core_sci_sm', 'en_core_web_sm')
                   Defaults to 'en_core_sci_sm' if available
    
    Returns:
        NEREntityDetector instance
    """
    return NEREntityDetector(ner_model=model_name, use_spacy=True)
