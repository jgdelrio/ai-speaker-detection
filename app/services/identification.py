import torchaudio
from speechbrain.inference import SpeakerRecognition
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
from typing import Dict, List, Tuple, Optional
import os


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


def determine_optimal_speakers(embeddings: List[np.ndarray], max_speakers: int = 10) -> int:
    """Use silhouette score to determine optimal number of speakers."""
    if len(embeddings) < 2:
        return 1
    
    # Reshape embeddings to 2D if needed
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim > 2:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    best_score = -1
    best_n_speakers = 1
    
    for n_speakers in range(2, min(len(embeddings), max_speakers) + 1):
        clustering = AgglomerativeClustering(
            n_clusters=n_speakers,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings_array)
        
        if len(set(labels)) > 1:  # Ensure we have multiple clusters
            score = silhouette_score(embeddings_array, labels, metric='cosine')
            if score > best_score:
                best_score = score
                best_n_speakers = n_speakers
    
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


def extract_embeddings(audio_path, chunk_size=2.0):
    """Legacy function maintained for backward compatibility."""
    embeddings, _ = extract_embeddings_with_overlap(audio_path, chunk_size, overlap=0.0)
    return embeddings

