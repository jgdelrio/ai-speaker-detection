import torchaudio
from speechbrain.inference import SpeakerRecognition
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
from typing import Dict, List, Tuple, Optional
import os
import re


# Initialize model (downloads pretrained weights locally)
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model",  # stores model locally
    run_opts={"device": "cpu"}  # forces CPU
)


def extract_embeddings_with_overlap(audio_path: str, chunk_size: float = 2.0, overlap: float = 0.5) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Extracts speaker embeddings from overlapping chunks of audio."""
    sig, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        sig = resampler(sig)

    chunk_samples = int(chunk_size * 16000)
    step_samples = int(chunk_samples * (1 - overlap))
    embeddings = []
    timestamps = []

    # Split into overlapping chunks
    for i in range(0, sig.shape[1] - chunk_samples + 1, step_samples):
        chunk = sig[:, i:i + chunk_samples]
        if chunk.shape[1] < 1000:  # Skip tiny chunks
            continue
        
        # Extract embedding
        emb = model.encode_batch(chunk).squeeze(0).numpy()
        embeddings.append(emb)
        
        # Calculate timestamp
        start_time = i / 16000
        end_time = (i + chunk_samples) / 16000
        timestamps.append((start_time, end_time))

    return embeddings, timestamps


def determine_optimal_speakers(embeddings: List[np.ndarray], max_speakers: int = 5) -> int:
    """Use silhouette score to determine optimal number of speakers."""
    if len(embeddings) < 2:
        return 1
    
    # Reshape embeddings to 2D if needed
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim > 2:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    n_samples = len(embeddings_array)
    
    # Silhouette score requires at least 2 samples and n_clusters must be < n_samples
    max_clusters = min(max_speakers, n_samples - 1)
    if max_clusters < 2:
        return 1
    
    best_score = -1
    best_n_speakers = 2  # Start with minimum valid clusters
    
    for n_speakers in range(2, max_clusters + 1):
        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_speakers,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings_array)
            
            # Check if we actually have the expected number of clusters
            unique_labels = len(set(labels))
            if unique_labels > 1 and unique_labels == n_speakers:
                score = silhouette_score(embeddings_array, labels, metric='cosine')
                if score > best_score:
                    best_score = score
                    best_n_speakers = n_speakers
        except ValueError as e:
            # Skip this number of clusters if it causes issues
            continue
    
    return best_n_speakers


def cluster_speakers(embeddings: List[np.ndarray], n_speakers: Optional[int] = None) -> List[int]:
    """Cluster embeddings using AgglomerativeClustering."""
    if len(embeddings) == 0:
        return []
    
    if len(embeddings) == 1:
        return [0]
    
    # Reshape embeddings to 2D if needed
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim > 2:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    if n_speakers is None:
        n_speakers = determine_optimal_speakers(embeddings)
    
    clustering = AgglomerativeClustering(
        n_clusters=n_speakers,
        metric='cosine',
        linkage='average'
    )
    
    labels = clustering.fit_predict(embeddings_array)
    return labels.tolist()


def merge_short_segments(labels: List[int], timestamps: List[Tuple[float, float]], min_duration: float = 0.5) -> List[int]:
    """Merge short segments with adjacent segments of the same speaker."""
    if len(labels) <= 1:
        return labels
    
    merged_labels = labels.copy()
    
    for i in range(len(labels)):
        segment_duration = timestamps[i][1] - timestamps[i][0]
        
        if segment_duration < min_duration:
            # Find the most common adjacent speaker
            adjacent_speakers = []
            
            if i > 0:
                adjacent_speakers.append(merged_labels[i-1])
            if i < len(merged_labels) - 1:
                adjacent_speakers.append(merged_labels[i+1])
            
            if adjacent_speakers:
                # Assign to most common adjacent speaker
                most_common = max(set(adjacent_speakers), key=adjacent_speakers.count)
                merged_labels[i] = most_common
    
    return merged_labels


def match_with_known_speakers(embeddings: List[np.ndarray], 
                            known_speakers: Dict[str, np.ndarray], 
                            threshold: float = 0.8) -> Tuple[List[int], Dict[str, np.ndarray]]:
    """Match embeddings with known speakers and identify new ones."""
    if not embeddings:
        return [], known_speakers
    
    embeddings_array = np.array(embeddings)
    speaker_assignments = []
    updated_speakers = known_speakers.copy()
    
    # Convert known speakers to array for comparison
    known_speaker_names = list(known_speakers.keys())
    known_embeddings = np.array([known_speakers[name] for name in known_speaker_names])
    
    next_speaker_id = len(known_speakers)
    
    for emb in embeddings:
        if len(known_embeddings) > 0:
            # Calculate similarity with known speakers
            similarities = cosine_similarity([emb], known_embeddings)[0]
            max_similarity = np.max(similarities)
            
            if max_similarity > threshold:
                # Assign to most similar known speaker
                best_match_idx = np.argmax(similarities)
                speaker_name = known_speaker_names[best_match_idx]
                speaker_id = list(known_speakers.keys()).index(speaker_name)
                speaker_assignments.append(speaker_id)
            else:
                # Create new speaker
                new_speaker_name = f"Speaker{next_speaker_id + 1}"
                updated_speakers[new_speaker_name] = emb
                speaker_assignments.append(next_speaker_id)
                next_speaker_id += 1
                
                # Update arrays for next iteration
                known_speaker_names.append(new_speaker_name)
                known_embeddings = np.vstack([known_embeddings, emb.reshape(1, -1)])
        else:
            # First speaker
            new_speaker_name = f"Speaker{next_speaker_id + 1}"
            updated_speakers[new_speaker_name] = emb
            speaker_assignments.append(next_speaker_id)
            next_speaker_id += 1
            
            known_speaker_names = [new_speaker_name]
            known_embeddings = emb.reshape(1, -1)
    
    return speaker_assignments, updated_speakers


def extract_sentence_based_embeddings(
    audio_path: str, 
    transcription_data: Dict
) -> Tuple[List[np.ndarray], List[Tuple[float, float]], List[str]]:
    """
    Extract speaker embeddings based on sentence boundaries from transcription.
    
    Args:
        audio_path: Path to audio file
        transcription_data: Dictionary containing transcription with word timestamps
        
    Returns:
        Tuple of (embeddings, timestamps, sentence_texts)
    """
    # Load audio
    sig, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        sig = resampler(sig)
    
    # Extract words with timestamps
    words = transcription_data.get("words", [])
    if not words:
        # Fallback to regular chunking if no word timestamps
        return extract_embeddings_with_overlap(audio_path, 3.0, 0.5)
    
    # Identify sentence boundaries using punctuation and word timing
    sentences = _identify_sentences(words)
    
    embeddings = []
    timestamps = []
    sentence_texts = []
    
    for sentence in sentences:
        start_time = sentence['start_time']
        end_time = sentence['end_time']
        text = sentence['text']
        
        # Convert to sample indices
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        
        # Ensure minimum segment length (1 second)
        min_samples = 16000
        if end_sample - start_sample < min_samples:
            # Extend segment to minimum length
            center = (start_sample + end_sample) // 2
            start_sample = max(0, center - min_samples // 2)
            end_sample = min(sig.shape[1], center + min_samples // 2)
        
        # Extract audio segment
        if start_sample < sig.shape[1] and end_sample > start_sample:
            segment = sig[:, start_sample:end_sample]
            
            if segment.shape[1] >= 1000:  # Minimum segment size
                try:
                    # Extract embedding
                    emb = model.encode_batch(segment).squeeze(0).numpy()
                    embeddings.append(emb)
                    timestamps.append((start_time, end_time))
                    sentence_texts.append(text)
                except Exception as e:
                    print(f"Warning: Could not extract embedding for sentence: {str(e)}")
                    continue
    
    # If we got very few sentences, fallback to regular chunking
    if len(embeddings) < 2:
        print("Few sentences detected, falling back to time-based chunking")
        return extract_embeddings_with_overlap(audio_path, 3.0, 0.5)
    
    return embeddings, timestamps, sentence_texts


def _identify_sentences(words: List[Dict]) -> List[Dict]:
    """
    Identify sentence boundaries from word list with timestamps.
    
    Args:
        words: List of word dictionaries with 'word', 'start', 'end' keys
        
    Returns:
        List of sentence dictionaries with start_time, end_time, text
    """
    if not words:
        return []
    
    sentences = []
    current_sentence_words = []
    
    # Sentence ending punctuation
    sentence_endings = {'.', '!', '?', '。', '！', '？'}  # Include some unicode variants
    
    for i, word_info in enumerate(words):
        word = word_info.get('word', '').strip()
        current_sentence_words.append(word_info)
        
        # Check if this word ends a sentence
        is_sentence_end = False
        
        # Check for punctuation at end of word
        if word and word[-1] in sentence_endings:
            is_sentence_end = True
        
        # Check for long pause to next word (indicating sentence boundary)
        if i < len(words) - 1:
            current_end = word_info.get('end', 0)
            next_start = words[i + 1].get('start', 0)
            pause_duration = next_start - current_end
            
            # Long pause (> 1 second) often indicates sentence boundary
            if pause_duration > 1.0:
                is_sentence_end = True
        
        # Force sentence end if we have too many words (prevent very long sentences)
        if len(current_sentence_words) >= 15:
            is_sentence_end = True
        
        # End of word list
        if i == len(words) - 1:
            is_sentence_end = True
        
        if is_sentence_end and current_sentence_words:
            # Create sentence
            start_time = current_sentence_words[0].get('start', 0)
            end_time = current_sentence_words[-1].get('end', start_time)
            text = ' '.join([w.get('word', '') for w in current_sentence_words])
            
            # Only keep sentences with reasonable length
            if end_time - start_time >= 0.5 and len(text.strip()) > 0:
                sentences.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text.strip(),
                    'word_count': len(current_sentence_words)
                })
            
            current_sentence_words = []
    
    return sentences


def detect_single_speaker_confidence(
    embeddings: List[np.ndarray], 
    timestamps: List[Tuple[float, float]]
) -> Tuple[bool, float]:
    """
    Determine if audio likely contains a single speaker based on embedding similarity.
    
    Args:
        embeddings: List of speaker embeddings
        timestamps: Corresponding timestamps
        
    Returns:
        Tuple of (is_single_speaker, confidence_score)
    """
    if len(embeddings) < 2:
        return True, 1.0
    
    # Calculate pairwise cosine similarities
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim > 2:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    similarities = cosine_similarity(embeddings_array)
    
    # Get upper triangular similarities (excluding diagonal)
    upper_tri_mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
    similarity_scores = similarities[upper_tri_mask]
    
    if len(similarity_scores) == 0:
        return True, 1.0
    
    # Calculate statistics
    mean_similarity = np.mean(similarity_scores)
    min_similarity = np.min(similarity_scores)
    std_similarity = np.std(similarity_scores)
    
    # Single speaker indicators:
    # 1. High mean similarity (> 0.8)
    # 2. High minimum similarity (> 0.6) 
    # 3. Low standard deviation (< 0.15)
    
    is_single_speaker = (
        mean_similarity > 0.8 and 
        min_similarity > 0.6 and 
        std_similarity < 0.15
    )
    
    # Confidence is based on how well the metrics support single speaker
    confidence = (
        min(mean_similarity / 0.8, 1.0) * 0.4 +
        min(min_similarity / 0.6, 1.0) * 0.4 +
        min((0.15 - std_similarity) / 0.15, 1.0) * 0.2
    )
    
    return is_single_speaker, confidence


def extract_embeddings(audio_path, chunk_size=2.0):
    """Legacy function maintained for backward compatibility."""
    embeddings, _ = extract_embeddings_with_overlap(audio_path, chunk_size, overlap=0.0)
    return embeddings

