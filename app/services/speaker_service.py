from fastapi import UploadFile
from typing import Dict, Any, Optional, List, Tuple
import tempfile
import os
import numpy as np
from app.services.identification import (
    extract_embeddings_with_overlap,
    cluster_speakers,
    merge_short_segments,
    match_with_known_speakers
)
class SpeakerService:
    async def process_speaker_identification(
            self,
            audio_file: UploadFile,
            transcription_data: Dict[str, Any],
            known_speakers: Optional[Dict[str, List[float]]] = None
    ) -> str:
        """Process speaker identification with advanced clustering and transcription formatting."""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_audio_path = temp_file.name
        
        try:
            # Extract embeddings with overlap
            embeddings, chunk_timestamps = extract_embeddings_with_overlap(
                temp_audio_path, chunk_size=2.0, overlap=0.5
            )
            
            # Convert known speakers from list format to numpy arrays if provided
            known_speakers_np = {}
            if known_speakers:
                known_speakers_np = {
                    name: np.array(embedding) 
                    for name, embedding in known_speakers.items()
                }
            
            # Match with known speakers or create new clustering
            if known_speakers_np:
                speaker_labels, updated_speakers = match_with_known_speakers(
                    embeddings, known_speakers_np, threshold=0.8
                )
            else:
                # Cluster speakers using AgglomerativeClustering
                speaker_labels = cluster_speakers(embeddings)
                # Create speaker embeddings dictionary
                unique_labels = list(set(speaker_labels))
                updated_speakers = {}
                for label in unique_labels:
                    speaker_name = f"Speaker{label + 1}"
                    # Use the first embedding for this speaker as representative
                    first_occurrence = speaker_labels.index(label)
                    updated_speakers[speaker_name] = embeddings[first_occurrence]
            
            # Post-processing: merge short segments
            merged_labels = merge_short_segments(speaker_labels, chunk_timestamps, min_duration=0.5)
            
            # Format transcription with speaker labels
            formatted_transcription = self._format_transcription_with_speakers(
                transcription_data, merged_labels, chunk_timestamps
            )
            
            return formatted_transcription
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    def _format_transcription_with_speakers(
        self, 
        transcription_data: Dict[str, Any], 
        speaker_labels: List[int], 
        chunk_timestamps: List[Tuple[float, float]]
    ) -> str:
        """Format transcription with speaker labels based on word timestamps."""
        
        # Extract words with timestamps from transcription data
        words = []
        if "words" in transcription_data:
            words = transcription_data["words"]
        elif "segments" in transcription_data:
            # Handle segment-based transcription format
            for segment in transcription_data["segments"]:
                if "words" in segment:
                    words.extend(segment["words"])
        
        if not words:
            # Fallback: use simple text splitting
            text = transcription_data.get("text", "")
            return self._simple_speaker_formatting(text, speaker_labels, chunk_timestamps)
        
        # Map words to speaker segments
        formatted_lines = []
        current_speaker = None
        current_line = []
        
        for word_info in words:
            word = word_info.get("word", "").strip()
            word_start = word_info.get("start", 0)
            
            # Find which speaker segment this word belongs to
            assigned_speaker = self._find_speaker_for_timestamp(word_start, chunk_timestamps, speaker_labels)
            
            if assigned_speaker != current_speaker:
                # New speaker, finish current line and start new one
                if current_line:
                    formatted_lines.append(f"Speaker{current_speaker + 1}: {' '.join(current_line)}")
                current_speaker = assigned_speaker
                current_line = [word]
            else:
                current_line.append(word)
        
        # Add the last line
        if current_line and current_speaker is not None:
            formatted_lines.append(f"Speaker{current_speaker + 1}: {' '.join(current_line)}")
        
        return "\n".join(formatted_lines)
    
    def _find_speaker_for_timestamp(
        self, 
        timestamp: float, 
        chunk_timestamps: List[Tuple[float, float]], 
        speaker_labels: List[int]
    ) -> int:
        """Find which speaker segment a given timestamp belongs to."""
        for i, (start, end) in enumerate(chunk_timestamps):
            if start <= timestamp <= end:
                return speaker_labels[i]
        
        # Fallback: find closest chunk
        closest_idx = 0
        min_distance = float('inf')
        for i, (start, end) in enumerate(chunk_timestamps):
            chunk_center = (start + end) / 2
            distance = abs(timestamp - chunk_center)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return speaker_labels[closest_idx] if speaker_labels else 0
    
    def _simple_speaker_formatting(
        self, 
        text: str, 
        speaker_labels: List[int], 
        chunk_timestamps: List[Tuple[float, float]]
    ) -> str:
        """Simple fallback formatting when word-level timestamps aren't available."""
        words = text.split()
        if not words or not speaker_labels:
            return f"Speaker1: {text}"
        
        # Roughly distribute words across speaker segments
        words_per_segment = len(words) / len(speaker_labels)
        formatted_lines = []
        current_speaker = None
        current_line = []
        
        for i, word in enumerate(words):
            segment_idx = min(int(i / words_per_segment), len(speaker_labels) - 1)
            assigned_speaker = speaker_labels[segment_idx]
            
            if assigned_speaker != current_speaker:
                if current_line:
                    formatted_lines.append(f"Speaker{current_speaker + 1}: {' '.join(current_line)}")
                current_speaker = assigned_speaker
                current_line = [word]
            else:
                current_line.append(word)
        
        if current_line and current_speaker is not None:
            formatted_lines.append(f"Speaker{current_speaker + 1}: {' '.join(current_line)}")
        
        return "\n".join(formatted_lines)
