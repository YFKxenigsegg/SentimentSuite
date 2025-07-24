"""
Advanced semantic text chunking using sentence embeddings and clustering.

This module implements intelligent text segmentation that goes beyond simple token counting
by understanding semantic relationships between sentences. It uses state-of-the-art 
sentence transformers and K-means clustering to create coherent, meaningful chunks.

Key Features:
- Sentence embedding-based similarity analysis
- K-means clustering for semantic grouping  
- Adaptive chunk sizing based on content complexity
- Intelligent overlap for context preservation
- Content complexity scoring for processing optimization

Algorithm Overview:
1. Split text into sentences with abbreviation handling
2. Generate embeddings for each sentence using SentenceTransformers
3. Determine optimal number of clusters based on content length and diversity
4. Use K-means to group semantically similar sentences
5. Build chunks from clusters with intelligent overlap
6. Calculate complexity scores for adaptive processing
"""

import re
import asyncio
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from core.interfaces import ITextChunker, TextChunk, ChunkMetadata
from core.config import settings

class SemanticChunker(ITextChunker):
    """
    Advanced semantic chunker using sentence embeddings and clustering.
    
    This implementation goes beyond traditional word-based chunking by understanding
    the semantic relationships between sentences and grouping related content together.
    This leads to more coherent summaries and better context preservation.
    
    Architecture:
    - Uses SentenceTransformer for high-quality sentence embeddings
    - Applies K-means clustering to group semantically similar sentences
    - Implements intelligent overlap between chunks for context continuity
    - Calculates complexity scores to guide adaptive summarization parameters
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic chunker with specified embedding model.
        
        Args:
            embedding_model: HuggingFace model name for sentence embeddings.
                            'all-MiniLM-L6-v2' provides good balance of speed/quality.
        """
        # Load the sentence transformer model for semantic embeddings
        # This model converts sentences to dense vector representations
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Compile regex pattern for efficient sentence boundary detection
        # Matches periods, exclamation marks, and question marks followed by whitespace
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        
    async def chunk_text(self, text: str, max_tokens: int = 512) -> List[TextChunk]:
        """
        Split text into semantic chunks using sentence embeddings and clustering.
        
        This is the main entry point for semantic chunking. The algorithm:
        1. Splits text into sentences with intelligent boundary detection
        2. For short texts, returns a single chunk
        3. For longer texts, generates sentence embeddings
        4. Clusters sentences by semantic similarity
        5. Builds chunks from clusters with overlap for context
        
        Args:
            text: Input text to be chunked
            max_tokens: Advisory maximum tokens per chunk
            
        Returns:
            List of TextChunk objects with rich metadata
        """
        # Step 1: Split text into sentences with intelligent boundary detection
        sentences = await self._split_into_sentences(text)
        
        # Short texts don't benefit from semantic chunking
        if len(sentences) <= 3:
            return [await self._create_single_chunk(text, sentences)]
        
        # Step 2: Generate sentence embeddings for semantic analysis
        embeddings = await self._get_sentence_embeddings(sentences)
        
        # Step 3: Determine optimal number of chunks based on content analysis
        optimal_chunks = await self._estimate_chunk_count(sentences, max_tokens)
        
        if optimal_chunks <= 1:
            return [await self._create_single_chunk(text, sentences)]
        
        # Step 4: Cluster sentences semantically using K-means
        clusters = await self._cluster_sentences(embeddings, optimal_chunks)
        
        # Step 5: Build final chunks from clusters with intelligent overlap
        chunks = await self._build_chunks_from_clusters(
            sentences, clusters, text, max_tokens
        )
        
        return chunks
    
    def estimate_optimal_chunk_size(self, text: str) -> int:
        """
        Analyze content complexity to determine optimal chunk size.
        
        Different types of content benefit from different chunk sizes:
        - Technical content: Smaller chunks for precision
        - Narrative content: Larger chunks for context
        - Simple content: Larger chunks for efficiency
        
        Args:
            text: Text to analyze for complexity
            
        Returns:
            Recommended chunk size in tokens
        """
        sentences = self.sentence_pattern.split(text.strip())
        
        # Calculate average sentence length as complexity indicator
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return settings.max_chunk_size
            
        avg_sentence_length = np.mean(sentence_lengths)
        
        # Adapt chunk size based on sentence complexity
        if avg_sentence_length > 25:  # Complex, technical sentences
            # Smaller chunks preserve precision for complex content
            return min(400, settings.max_chunk_size)
        elif avg_sentence_length < 10:  # Simple, conversational sentences  
            # Larger chunks for simple content improve efficiency
            return min(600, settings.max_chunk_size)
        else:
            # Use configured default for medium complexity
            return settings.max_chunk_size
    
    async def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with improved boundary detection.
        
        This method handles common edge cases that simple regex splitting misses:
        - Abbreviations that shouldn't end sentences (Dr., Mr., etc.)
        - Minimum sentence length requirements
        - Whitespace normalization
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentence strings
        """
        sentences = []
        current = ""
        
        # Common abbreviations that shouldn't trigger sentence boundaries
        # This prevents splitting on "Dr. Smith" or "e.g. example"
        abbreviations = {
            'Mr.', 'Mrs.', 'Dr.', 'Prof.', 'vs.', 'etc.', 
            'e.g.', 'i.e.', 'Co.', 'Inc.', 'Ltd.', 'Corp.'
        }
        
        # Process each potential sentence boundary
        for part in self.sentence_pattern.split(text):
            part = part.strip()
            if not part:
                continue
                
            current += part + " "
            
            # Check if this looks like a genuine sentence ending
            words = current.strip().split()
            if words and not any(current.strip().endswith(abbr) for abbr in abbreviations):
                # Require minimum viable sentence length (3+ words)
                if len(words) >= 3:
                    sentences.append(current.strip())
                    current = ""
        
        # Handle any remaining text
        if current.strip():
            sentences.append(current.strip())
            
        # Filter out very short fragments that aren't meaningful
        return [s for s in sentences if len(s.split()) >= 2]
    
    async def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate semantic embeddings for sentences asynchronously.
        
        Uses SentenceTransformer to convert sentences into dense vector representations
        that capture semantic meaning. These embeddings enable similarity analysis.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            NumPy array of sentence embeddings (shape: [n_sentences, embedding_dim])
        """
        # Run embedding generation in thread pool to avoid blocking
        # This is CPU-intensive work that benefits from being off the main thread
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.embedding_model.encode, sentences
        )
        return embeddings
    
    async def _estimate_chunk_count(self, sentences: List[str], max_tokens: int) -> int:
        """
        Determine optimal number of chunks based on content analysis.
        
        Balances chunk size constraints with content coherence requirements.
        Considers both token limits and content diversity.
        
        Args:
            sentences: List of sentences to analyze
            max_tokens: Target maximum tokens per chunk
            
        Returns:
            Optimal number of chunks for this content
        """
        # Calculate base chunk count from token constraints
        total_tokens = sum(len(s.split()) for s in sentences)
        base_chunks = max(1, (total_tokens + max_tokens - 1) // max_tokens)
        
        # Adjust for content diversity - longer texts benefit from more chunks
        if len(sentences) > 20:
            # For longer texts, prefer smaller chunks for better coherence
            # But don't exceed reasonable limits
            base_chunks = min(base_chunks + 1, len(sentences) // 3)
        
        # Cap at 8 chunks to maintain manageable processing
        return min(base_chunks, 8)
    
    async def _cluster_sentences(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Cluster sentences using K-means on their semantic embeddings.
        
        Groups semantically similar sentences together using K-means clustering
        on the high-dimensional embedding space.
        
        Args:
            embeddings: Sentence embeddings array
            n_clusters: Number of clusters to create
            
        Returns:
            Cluster assignment array (shape: [n_sentences])
        """
        # Handle edge case where we have fewer sentences than desired clusters
        if n_clusters >= len(embeddings):
            # Each sentence gets its own cluster
            return np.arange(len(embeddings))
            
        # Run K-means clustering in thread pool for CPU-intensive work
        loop = asyncio.get_event_loop()
        
        # Configure K-means for reproducible results
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42,  # Reproducible results
            n_init=10  # Multiple initializations for stability
        )
        
        clusters = await loop.run_in_executor(None, kmeans.fit_predict, embeddings)
        return clusters
    
    async def _build_chunks_from_clusters(
        self, 
        sentences: List[str], 
        clusters: np.ndarray, 
        original_text: str,
        max_tokens: int
    ) -> List[TextChunk]:
        """
        Build final text chunks from sentence clusters with intelligent overlap.
        
        This method creates the final chunks by:
        1. Grouping sentences by cluster assignment
        2. Ordering clusters by original text position
        3. Adding intelligent overlap between adjacent chunks
        4. Calculating rich metadata for each chunk
        
        Args:
            sentences: Original sentences
            clusters: Cluster assignments from K-means
            original_text: Original full text for position tracking
            max_tokens: Token limit for validation
            
        Returns:
            List of TextChunk objects with metadata
        """
        chunks = []
        
        # Group sentences by their cluster assignment
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append((i, sentences[i]))
        
        # Sort clusters by first sentence position to maintain text order
        # This ensures chunks follow the logical flow of the original text
        sorted_clusters = sorted(
            cluster_groups.items(), 
            key=lambda x: min(pos for pos, _ in x[1])
        )
        
        # Build each chunk with overlap for context preservation
        for chunk_id, (cluster_id, sentence_data) in enumerate(sorted_clusters):
            # Sort sentences within cluster by original position
            sentence_data.sort(key=lambda x: x[0])
            
            # Extract sentence content
            chunk_sentences = [sent for _, sent in sentence_data]
            chunk_text = " ".join(chunk_sentences)
            
            # Add intelligent overlap from previous chunk for context
            if chunk_id > 0:
                prev_chunk = sorted_clusters[chunk_id - 1][1]
                if prev_chunk:
                    # Take last sentence from previous chunk for context
                    overlap_sentences = [sent for _, sent in prev_chunk[-1:]]
                    chunk_text = " ".join(overlap_sentences + chunk_sentences)
            
            # Add overlap from next chunk for forward context
            if chunk_id < len(sorted_clusters) - 1:
                next_chunk = sorted_clusters[chunk_id + 1][1]
                if next_chunk:
                    # Take first sentence from next chunk
                    overlap_sentences = [sent for _, sent in next_chunk[:1]]
                    chunk_text = " ".join([chunk_text] + overlap_sentences)
            
            # Calculate position metadata for tracking
            start_pos = original_text.find(chunk_sentences[0]) if chunk_sentences else 0
            end_pos = start_pos + len(" ".join(chunk_sentences))
            
            # Create rich metadata for adaptive processing
            metadata = ChunkMetadata(
                id=chunk_id,
                start_pos=max(0, start_pos),
                end_pos=min(len(original_text), end_pos),
                token_count=len(chunk_text.split()),
                sentence_count=len(chunk_sentences),
                complexity_score=await self._calculate_complexity_score(chunk_text),
                priority=1  # Normal priority, could be adjusted based on content
            )
            
            chunks.append(TextChunk(content=chunk_text, metadata=metadata))
        
        return chunks
    
    async def _create_single_chunk(self, text: str, sentences: List[str]) -> TextChunk:
        """
        Create a single chunk for texts too short for semantic chunking.
        
        Used when the input text is short enough that chunking would not
        provide benefits and might actually hurt coherence.
        
        Args:
            text: Full input text
            sentences: Parsed sentences
            
        Returns:
            Single TextChunk containing all content
        """
        metadata = ChunkMetadata(
            id=0,  # Single chunk gets ID 0
            start_pos=0,
            end_pos=len(text),
            token_count=len(text.split()),
            sentence_count=len(sentences),
            complexity_score=await self._calculate_complexity_score(text),
            priority=1
        )
        
        return TextChunk(content=text, metadata=metadata)
    
    async def _calculate_complexity_score(self, text: str) -> float:
        """
        Calculate content complexity score for adaptive processing.
        
        Analyzes multiple factors to determine text complexity:
        - Average word length (vocabulary sophistication)
        - Average sentence length (structural complexity)
        - Vocabulary diversity (conceptual richness)
        
        This score is used to adapt summarization parameters for optimal results.
        
        Args:
            text: Text to analyze
            
        Returns:
            Complexity score from 0.0 (simple) to 1.0 (complex)
        """
        words = text.split()
        sentences = self.sentence_pattern.split(text)
        
        # Handle edge cases
        if not words or not sentences:
            return 0.0
        
        # Factor 1: Average word length (vocabulary sophistication)
        # Longer words often indicate technical or academic content
        avg_word_length = np.mean([len(word) for word in words])
        
        # Factor 2: Average sentence length (structural complexity)
        # Longer sentences suggest more complex grammar and ideas
        avg_sentence_length = len(words) / len(sentences)
        
        # Factor 3: Vocabulary diversity (conceptual richness)
        # Higher diversity suggests more varied concepts and complexity
        unique_words = len(set(word.lower() for word in words))
        vocab_diversity = unique_words / len(words)
        
        # Normalize factors to 0-1 range using reasonable upper bounds
        word_complexity = min(avg_word_length / 8.0, 1.0)  # 8 chars = complex
        sentence_complexity = min(avg_sentence_length / 25.0, 1.0)  # 25 words = complex
        
        # Weighted combination emphasizing sentence structure
        complexity = (
            0.3 * word_complexity +      # Word sophistication
            0.4 * sentence_complexity +  # Structural complexity (most important)
            0.3 * vocab_diversity        # Conceptual diversity
        )
        
        # Ensure score stays in valid range
        return min(max(complexity, 0.0), 1.0) 