"""
Lightweight proposition extractor to support proposition-first chunking.

This module surfaces a minimal, dependency-free way to split a sentence into
relation-bearing propositions before feeding them to the semantic chunker.

Heuristics:
- Uses relation triggers from config (tiered_triggers or relation_triggers).
- Finds nearest entity before and after a trigger as (subject, object).
- Also supports "Prefix" triggers (Trigger followed by two entities).
- Builds a span between start of first element and end of last element; skips overly long spans.
- Ranks propositions with hierarchy + trigger tier priority.

Usage:
    from proposition_chunker import PropositionChunker
    pc = PropositionChunker(config, entity_detector=chunker.detect_entities)
    props = pc.extract_propositions(text)
"""

from dataclasses import dataclass
import re
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class Proposition:
    text: str  # text used for chunking/scoring (cleaned)
    span_start: int
    span_end: int
    subject: Optional[str]
    obj: Optional[str]
    trigger: str
    relation_type: str
    priority: int
    canonical_text: Optional[str] = None  # canonical "SUBJECT trigger OBJECT"
    original_text: Optional[str] = None  # raw span text before recontextualization


class PropositionChunker:
    def __init__(
        self,
        config: Dict,
        entity_detector: Optional[Callable[[str], List[Dict]]] = None,
        max_prop_tokens: int = 25,  # Increased slightly to capture prefix patterns
        min_prop_words: int = 3,    # Lowered slightly
        context_window: int = 3,
    ):
        """
        Args:
            config: Chunker config dict (supports tiered_triggers/relation_triggers,
                    relation_hierarchy).
            entity_detector: Optional callable text -> [{'text','start','end',...}]
            max_prop_tokens: Skip propositions longer than this token count.
            min_prop_words: Skip propositions shorter than this word count.
            context_window: Tokens to include before/after the subject-object span.
        """
        self.config = config or {}
        self.entity_detector = entity_detector
        self.max_prop_tokens = max_prop_tokens
        self.min_prop_words = min_prop_words
        self.context_window = context_window
        self.trigger_map = self._build_trigger_map()
        self.relation_priority = self._build_relation_priority()

        # Triggers that imply the Subject and Object are swapped in the text
        # (e.g. "A is treated by B" -> Text order: A treated B. Logic: B treats A)
        self.passive_markers = {'by', 'with', 'from', 'in'} # Simple check for word after trigger
        self.inverted_triggers = {
            'treated', 'induced', 'caused', 'produced', 'stimulated', 'inhibited', 
            'mediated', 'triggered', 'elicited', 'generated', 'initiated', 'promoted',
            'activated', 'associated'
        }

    def _build_trigger_map(self) -> Dict[str, List[str]]:
        """Flatten tiered_triggers or relation_triggers into {relation: [triggers]}."""
        # Start with relation_triggers if available
        flat = self.config.get("relation_triggers", {}).copy()
        
        # Merge/Overwrite with tiered_triggers if available
        if "tiered_triggers" in self.config:
            for rel, tiers in self.config["tiered_triggers"].items():
                if rel not in flat:
                    flat[rel] = []
                # Add tiered triggers if not already present
                for trig_list in tiers.values():
                    for t in trig_list:
                        if t not in flat[rel]:
                            flat[rel].append(t)
        return flat

    def _build_relation_priority(self) -> Dict[str, int]:
        """Get relation priority from relation_hierarchy if provided."""
        hierarchy = self.config.get("relation_hierarchy", {})
        # relation_hierarchy may be {"priority_order":[...]} or {"REL":{"priority":..}}
        if "priority_order" in hierarchy:
            return {rel: len(hierarchy["priority_order"]) - i for i, rel in enumerate(hierarchy["priority_order"])}
        return {rel: cfg.get("priority", 0) for rel, cfg in hierarchy.items()}

    def _tier_bonus(self, relation: str, trigger: str) -> int:
        tiers = self.config.get("tiered_triggers", {}).get(relation, {})
        if trigger in tiers.get("primary", []):
            return 3
        if trigger in tiers.get("secondary", []):
            return 2
        if trigger in tiers.get("tertiary", []):
            return 1
        return 0

    def _priority(self, relation: str, trigger: str) -> int:
        base = self.relation_priority.get(relation, 0)
        return base + self._tier_bonus(relation, trigger)

    @staticmethod
    def _deduplicate(props: Sequence[Proposition]) -> List[Proposition]:
        """Remove overlapping propositions, keeping highest priority first."""
        props_sorted = sorted(props, key=lambda p: (-p.priority, p.span_start, p.span_end))
        kept: List[Proposition] = []
        for prop in props_sorted:
            overlap = any(
                not (prop.span_end <= k.span_start or prop.span_start >= k.span_end)
                for k in kept
            )
            if not overlap:
                kept.append(prop)
        return kept

    def _find_entities(self, text: str) -> List[Dict]:
        if self.entity_detector:
            try:
                return self.entity_detector(text) or []
            except Exception:
                return []
        return []

    def extract_propositions(
        self,
        text: str,
        entities: Optional[List[Dict]] = None,
        expected_relation: Optional[str] = None,
    ) -> List[Proposition]:
        """
        Extract propositions from text using triggers + nearest entities.
        Supports both Infix (A trigger B) and Prefix (Trigger A ... B) patterns.

        Args:
            text: input sentence
            entities: optional precomputed entities [{'text','start','end',...}]
            expected_relation: optional relation hint; propositions with a different
                                relation_type are skipped

        Returns:
            List[Proposition] sorted by priority (highest first), deduplicated.
        """
        if not text:
            return []

        ents = entities if entities is not None else self._find_entities(text)
        ents = sorted(ents, key=lambda e: e["start"])
        if len(ents) < 2: # Need at least two entities for a relation typically
            # Exception: if we want to capture unary relations, but typical RE needs 2.
            # Actually, let's allow 1 entity if the trigger is very strong, but for now keep 2 constraint.
            if not ents: return [] 
            pass

        expected_rel_upper = expected_relation.upper() if expected_relation else None

        # Precompute token spans for lightweight context expansion
        token_spans = list(re.finditer(r"\S+", text))

        def expand_with_context(start: int, end: int) -> (int, int):
            """Expand a char span by context_window tokens on each side."""
            if not token_spans:
                return start, end
            start_idx = 0
            end_idx = len(token_spans) - 1
            for i, m in enumerate(token_spans):
                if m.start() <= start < m.end():
                    start_idx = i
                    break
                if start < m.start():
                    start_idx = max(0, i - 1)
                    break
            for i, m in enumerate(token_spans):
                if m.start() < end <= m.end():
                    end_idx = i
                    break
                if end < m.start():
                    end_idx = max(0, i - 1)
                    break
            start_idx = max(0, start_idx - self.context_window)
            end_idx = min(len(token_spans) - 1, end_idx + self.context_window)
            return token_spans[start_idx].start(), token_spans[end_idx].end()

        def noisy(text_val: str) -> bool:
            """Filter out spans that are purely numeric/markers/brackets."""
            t = text_val.strip().lower()
            if not t:
                return True
            if re.match(r"^[\\[\\{]?[0-9][0-9\\-\\.]*", t):
                return True
            if len(t) <= 2:
                return True
            return False

        propositions: List[Proposition] = []
        for relation, triggers in self.trigger_map.items():
            # If a relation hint is provided, skip other relations early
            if expected_rel_upper and relation.upper() != expected_rel_upper:
                continue

            for trig in triggers:
                if not trig:
                    continue

                # Enforce word boundaries to avoid partial matches (e.g. "treat" in "treatment")
                pattern = r"\b" + re.escape(trig) + r"\b"
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    t_start, t_end = match.start(), match.end()
                    
                    # Identify entities before and after trigger
                    before = [e for e in ents if e["end"] <= t_start]
                    after = [e for e in ents if e["start"] >= t_end]

                    subj = None
                    obj = None
                    pattern_type = "infix"

                    # Strategy 1: Infix (Standard) - Entity TRIGGER Entity
                    if before and after:
                        subj = before[-1]
                        obj = after[0]
                        pattern_type = "infix"
                    
                    # Strategy 2: Prefix - TRIGGER Entity ... Entity (e.g. "Treatment of X with Y")
                    elif not before and len(after) >= 2:
                        subj = after[0]
                        obj = after[1]
                        pattern_type = "prefix"
                    
                    # Strategy 3: Postfix - Entity ... Entity TRIGGER (Rare in English, e.g. "X and Y interaction")
                    elif not after and len(before) >= 2:
                        subj = before[-2]
                        obj = before[-1]
                        pattern_type = "postfix"
                    
                    if not subj or not obj:
                        continue

                    # Determine span boundaries
                    # We want the span to cover the Trigger and both Entities
                    
                    elements_start = min(t_start, subj["start"], obj["start"])
                    elements_end = max(t_end, subj["end"], obj["end"])

                    span_start = elements_start
                    span_end = elements_end

                    # Expand span with a bit of local context
                    expanded_start, expanded_end = expand_with_context(span_start, span_end)
                    span_text = text[expanded_start:expanded_end].strip()
                    
                    if not span_text:
                        continue
                    
                    span_words = span_text.split()
                    # Prefix patterns are often longer ("Treatment of ... with ..."), allow more tokens
                    limit = self.max_prop_tokens + 5 if pattern_type == "prefix" else self.max_prop_tokens
                    
                    if len(span_words) > limit:
                        continue

                    subj_text = subj.get("text")
                    obj_text = obj.get("text")

                    if not subj_text or not obj_text:
                        continue

                    # Check for passive construction (Entity TRIGGER by/with Entity)
                    is_passive = False
                    if pattern_type == "infix" and trig.lower() in self.inverted_triggers:
                        # Look at the word immediately following the trigger in the text
                        # We need the text span after the trigger
                        after_trig_start = match.end()
                        after_trig_text = text[after_trig_start:after_trig_start+10].strip().lower()
                        first_word = after_trig_text.split()[0] if after_trig_text else ""
                        if first_word in self.passive_markers:
                            is_passive = True

                    # Build canonical recontextualization
                    canonical = self._recontextualize(
                        subj_text, trig, obj_text, pattern_type, is_passive
                    )
                    canonical = self._clean_chunk_text(canonical)

                    chunk_text = self._clean_chunk_text(span_text)

                    # Prefer canonical if it is high quality, otherwise use span
                    # For passive/inverted, we STRONGLY prefer canonical to fix the order
                    canonical_text_val = None
                    
                    if is_passive:
                         chunk_text = canonical
                         canonical_text_val = canonical
                    elif pattern_type == "infix" and canonical and len(canonical.split()) >= max(len(chunk_text.split()), self.min_prop_words):
                        chunk_text = canonical
                        canonical_text_val = canonical
                    elif pattern_type == "prefix" and canonical:
                         # Use canonical for prefix if it seems reasonable (e.g. "Treatment of A with B")
                        chunk_text = canonical
                        canonical_text_val = canonical
                    else:
                        # For postfix, span is usually safer unless we are sure
                        # But if we have a smart prefix recontextualizer, we can use it
                        pass

                        # Skip if too short or noisy
                        if len(chunk_text.split()) < self.min_prop_words or noisy(chunk_text):
                            continue

                    propositions.append(
                        Proposition(
                            text=chunk_text,
                            span_start=expanded_start,
                            span_end=expanded_end,
                            subject=subj_text,
                            obj=obj_text,
                            trigger=trig,
                            relation_type=relation,
                            priority=self._priority(relation, trig),
                            canonical_text=canonical_text_val,
                            original_text=span_text
                        )
                    )

        return self._deduplicate(propositions)

    @staticmethod
    def _recontextualize(
        subject: Optional[str], 
        trigger: str, 
        obj: Optional[str], 
        pattern: str = "infix",
        is_passive: bool = False
    ) -> str:
        """
        Rewrite a proposition into a short, canonical string.
        Handles passive voice swapping and preposition insertion.
        """
        subject = subject.strip() if subject else ""
        trigger = trigger.strip() if trigger else ""
        obj = obj.strip() if obj else ""
        
        if is_passive:
            # "A is treated by B" -> Logical: "B treats A"
            # We construct: Object Trigger Subject (swapped)
            # But we need to fix the verb tense/form roughly?
            # "treated" -> "treats"? Hard to do without lemma.
            # "B treated A" is fine.
            return f"{obj} {trigger} {subject}"
        
        if pattern == "prefix":
            # "Treatment of A with B"
            # Try to reconstruct a natural sentence
            return f"{trigger} of {subject} with {obj}"
        elif pattern == "postfix":
            return f"{subject} and {obj} {trigger}"
        else:
            # Infix
            # Try to add "of" if it sounds like a noun phrase trigger ("inhibition", "production")
            if trigger.endswith("tion") or trigger.endswith("sis") or trigger.endswith("ment"):
                 return f"{subject} {trigger} of {obj}"
            return f"{subject} {trigger} {obj}"

    @staticmethod
    def _clean_chunk_text(text: str) -> str:
        """Strip leading punctuation/brackets and trim whitespace."""
        cleaned = re.sub(r'^[\\s,;:\\-\\(\\[\\]]+', '', text)
        cleaned = cleaned.strip()
        return cleaned if cleaned else text.strip()
