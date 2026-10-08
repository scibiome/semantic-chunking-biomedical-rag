#!/usr/bin/env python3
"""
Lightweight chunk re-ranking without neural model training.
This demonstrates how to improve chunk selection using simple heuristics.
"""

# Relation-triggering keywords for each relation
RELATION_TRIGGERS = {
    'ADMINISTERED_TO': ['administered', 'given', 'injected', 'delivered', 'dose', 'dosage'],
    'TREATS': ['treats', 'therapy', 'treatment', 'alleviates', 'cures', 'relieves'],
    'INHIBITS': ['inhibits', 'blocks', 'suppresses', 'prevents', 'reduces'],
    'STIMULATES': ['stimulates', 'activates', 'enhances', 'increases', 'promotes'],
    'PREVENTS': ['prevents', 'avoids', 'reduces risk', 'protects'],
    'PRODUCES': ['produces', 'generates', 'creates', 'synthesizes'],
    'INTERACTS_WITH': ['interacts', 'binds', 'associates', 'combines'],
    'CAUSES': ['causes', 'induces', 'leads to', 'triggers'],
    'DIAGNOSES': ['diagnoses', 'detects', 'identifies', 'reveals'],
    'PROCESS_OF': ['process', 'condition', 'disease', 'disorder'],
    # Add more as needed
}

def score_chunk_lightweight(chunk_text, relation, context, embedding_sim):
    """
    Lightweight scoring function combining multiple signals.
    
    Args:
        chunk_text: The chunk text to score
        relation: Target relation (e.g., 'TREATS')
        context: Original context sentence
        embedding_sim: Embedding similarity score (0-1)
    
    Returns:
        Combined score (0-1)
    """
    score = 0.0
    chunk_lower = chunk_text.lower()
    context_lower = context.lower()
    
    # 1. Embedding similarity (40% weight)
    score += 0.4 * embedding_sim
    
    # 2. Relation keyword matching (30% weight)
    keywords = RELATION_TRIGGERS.get(relation, [])
    if keywords:
        keyword_matches = sum(1 for kw in keywords if kw.lower() in chunk_lower)
        keyword_score = keyword_matches / len(keywords)
        score += 0.3 * keyword_score
    else:
        # If no keywords defined, give neutral score
        score += 0.15
    
    # 3. Entity overlap (20% weight) - simple word overlap
    chunk_words = set(chunk_lower.split())
    context_words = set(context_lower.split())
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    chunk_words = chunk_words - stop_words
    context_words = context_words - stop_words
    
    if context_words:
        overlap = len(chunk_words & context_words) / len(context_words)
        score += 0.2 * min(overlap, 1.0)
    
    # 4. Chunk length preference (10% weight) - prefer 10-20 words
    word_count = len(chunk_text.split())
    if 10 <= word_count <= 20:
        length_score = 1.0
    elif word_count < 10:
        length_score = word_count / 10.0
    else:
        length_score = max(0, 1.0 - (word_count - 20) / 30.0)
    score += 0.1 * length_score
    
    return min(score, 1.0)  # Cap at 1.0


def rerank_chunks_lightweight(chunks_with_scores, relation, context, top_k=3):
    """
    Re-rank chunks using lightweight scoring.
    
    Args:
        chunks_with_scores: List of (chunk_text, relation, embedding_sim) tuples
        relation: Target relation
        context: Original context sentence
        top_k: Number of top chunks to return
    
    Returns:
        List of (chunk_text, relation, final_score) tuples, sorted by score
    """
    scored_chunks = []
    for chunk_text, chunk_relation, embedding_sim in chunks_with_scores:
        # Skip chunks with wrong relation
        if chunk_relation != relation:
            continue
        
        # Compute lightweight score
        final_score = score_chunk_lightweight(chunk_text, chunk_relation, context, embedding_sim)
        scored_chunks.append((chunk_text, chunk_relation, final_score))
    
    # Sort by final score (descending)
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    
    return scored_chunks[:top_k]


# Example usage
if __name__ == "__main__":
    # Example: chunks from retrieval
    chunks = [
        ("aspirin treats headache and reduces inflammation", "TREATS", 0.85),
        ("the effect of bicarbonate on liver alcohol dehydrogenase", "INTERACTS_WITH", 0.82),
        ("chemotherapy treatment for cancer patients", "TREATS", 0.78),
        ("inhibition of enzyme activity", "INHIBITS", 0.75),
    ]
    
    relation = "TREATS"
    context = "aspirin is commonly used to treat headaches in patients"
    
    # Re-rank chunks
    reranked = rerank_chunks_lightweight(chunks, relation, context, top_k=3)
    
    print("Original chunks (by embedding similarity):")
    for chunk, rel, score in chunks:
        print(f"  {chunk[:50]}... | {rel} | {score:.3f}")
    
    print("\nRe-ranked chunks (by lightweight scoring):")
    for chunk, rel, score in reranked:
        print(f"  {chunk[:50]}... | {rel} | {score:.3f}")






"""
Lightweight chunk re-ranking without neural model training.
This demonstrates how to improve chunk selection using simple heuristics.
"""

# Relation-triggering keywords for each relation
RELATION_TRIGGERS = {
    'ADMINISTERED_TO': ['administered', 'given', 'injected', 'delivered', 'dose', 'dosage'],
    'TREATS': ['treats', 'therapy', 'treatment', 'alleviates', 'cures', 'relieves'],
    'INHIBITS': ['inhibits', 'blocks', 'suppresses', 'prevents', 'reduces'],
    'STIMULATES': ['stimulates', 'activates', 'enhances', 'increases', 'promotes'],
    'PREVENTS': ['prevents', 'avoids', 'reduces risk', 'protects'],
    'PRODUCES': ['produces', 'generates', 'creates', 'synthesizes'],
    'INTERACTS_WITH': ['interacts', 'binds', 'associates', 'combines'],
    'CAUSES': ['causes', 'induces', 'leads to', 'triggers'],
    'DIAGNOSES': ['diagnoses', 'detects', 'identifies', 'reveals'],
    'PROCESS_OF': ['process', 'condition', 'disease', 'disorder'],
    # Add more as needed
}

def score_chunk_lightweight(chunk_text, relation, context, embedding_sim):
    """
    Lightweight scoring function combining multiple signals.
    
    Args:
        chunk_text: The chunk text to score
        relation: Target relation (e.g., 'TREATS')
        context: Original context sentence
        embedding_sim: Embedding similarity score (0-1)
    
    Returns:
        Combined score (0-1)
    """
    score = 0.0
    chunk_lower = chunk_text.lower()
    context_lower = context.lower()
    
    # 1. Embedding similarity (40% weight)
    score += 0.4 * embedding_sim
    
    # 2. Relation keyword matching (30% weight)
    keywords = RELATION_TRIGGERS.get(relation, [])
    if keywords:
        keyword_matches = sum(1 for kw in keywords if kw.lower() in chunk_lower)
        keyword_score = keyword_matches / len(keywords)
        score += 0.3 * keyword_score
    else:
        # If no keywords defined, give neutral score
        score += 0.15
    
    # 3. Entity overlap (20% weight) - simple word overlap
    chunk_words = set(chunk_lower.split())
    context_words = set(context_lower.split())
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    chunk_words = chunk_words - stop_words
    context_words = context_words - stop_words
    
    if context_words:
        overlap = len(chunk_words & context_words) / len(context_words)
        score += 0.2 * min(overlap, 1.0)
    
    # 4. Chunk length preference (10% weight) - prefer 10-20 words
    word_count = len(chunk_text.split())
    if 10 <= word_count <= 20:
        length_score = 1.0
    elif word_count < 10:
        length_score = word_count / 10.0
    else:
        length_score = max(0, 1.0 - (word_count - 20) / 30.0)
    score += 0.1 * length_score
    
    return min(score, 1.0)  # Cap at 1.0


def rerank_chunks_lightweight(chunks_with_scores, relation, context, top_k=3):
    """
    Re-rank chunks using lightweight scoring.
    
    Args:
        chunks_with_scores: List of (chunk_text, relation, embedding_sim) tuples
        relation: Target relation
        context: Original context sentence
        top_k: Number of top chunks to return
    
    Returns:
        List of (chunk_text, relation, final_score) tuples, sorted by score
    """
    scored_chunks = []
    for chunk_text, chunk_relation, embedding_sim in chunks_with_scores:
        # Skip chunks with wrong relation
        if chunk_relation != relation:
            continue
        
        # Compute lightweight score
        final_score = score_chunk_lightweight(chunk_text, chunk_relation, context, embedding_sim)
        scored_chunks.append((chunk_text, chunk_relation, final_score))
    
    # Sort by final score (descending)
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    
    return scored_chunks[:top_k]


# Example usage
if __name__ == "__main__":
    # Example: chunks from retrieval
    chunks = [
        ("aspirin treats headache and reduces inflammation", "TREATS", 0.85),
        ("the effect of bicarbonate on liver alcohol dehydrogenase", "INTERACTS_WITH", 0.82),
        ("chemotherapy treatment for cancer patients", "TREATS", 0.78),
        ("inhibition of enzyme activity", "INHIBITS", 0.75),
    ]
    
    relation = "TREATS"
    context = "aspirin is commonly used to treat headaches in patients"
    
    # Re-rank chunks
    reranked = rerank_chunks_lightweight(chunks, relation, context, top_k=3)
    
    print("Original chunks (by embedding similarity):")
    for chunk, rel, score in chunks:
        print(f"  {chunk[:50]}... | {rel} | {score:.3f}")
    
    print("\nRe-ranked chunks (by lightweight scoring):")
    for chunk, rel, score in reranked:
        print(f"  {chunk[:50]}... | {rel} | {score:.3f}")











