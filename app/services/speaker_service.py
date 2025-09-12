from fastapi import UploadFile
from typing import Dict, Any, Optional, List, Tuple
import tempfile
import os
import numpy as np
from app.services.identification import (
    extract_embeddings_with_overlap,
    extract_sentence_based_embeddings,
    detect_single_speaker_confidence,
    cluster_speakers,
    merge_short_segments,
    match_with_known_speakers
)
class SpeakerService:
    async def process_speaker_identification(
            self,
            audio_file: UploadFile,
            transcription_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process speaker identification with advanced clustering and transcription formatting."""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_audio_path = temp_file.name
        
        try:
            # Try sentence-aware segmentation first
            try:
                embeddings, chunk_timestamps, sentence_texts = extract_sentence_based_embeddings(
                    temp_audio_path, transcription_data
                )
                segmentation_method = "sentence-based"
            except Exception as e:
                print(f"Sentence-based segmentation failed, falling back to time-based: {str(e)}")
                # Fallback to time-based chunking
                embeddings, chunk_timestamps = extract_embeddings_with_overlap(
                    temp_audio_path, chunk_size=3.0, overlap=0.5
                )
                sentence_texts = None
                segmentation_method = "time-based"
            
            # Check if this is likely a single speaker
            is_single_speaker, single_speaker_confidence = detect_single_speaker_confidence(
                embeddings, chunk_timestamps
            )
            
            print(f"Single speaker detection: {is_single_speaker} (confidence: {single_speaker_confidence:.3f})")
            print(f"Segmentation method: {segmentation_method}")
            
            if is_single_speaker and single_speaker_confidence > 0.6:
                # Force single speaker assignment
                speaker_labels = [0] * len(embeddings)  # All segments assigned to Speaker 1
                print("Forcing single speaker assignment based on similarity analysis")
            else:
                # Proceed with clustering
                speaker_labels = cluster_speakers(embeddings, n_speakers=None)
                
                # Post-processing: merge short segments more aggressively
                speaker_labels = merge_short_segments(speaker_labels, chunk_timestamps, min_duration=1.0)
                
                # Apply additional smoothing to reduce fragmentation
                speaker_labels = self._smooth_speaker_transitions(speaker_labels, chunk_timestamps)
                
                # Additional check: if one speaker dominates >80% of time, force single speaker
                if len(set(speaker_labels)) > 1:
                    dominant_speaker_ratio = self._check_dominant_speaker(speaker_labels, chunk_timestamps)
                    if dominant_speaker_ratio > 0.80:
                        speaker_labels = [0] * len(embeddings)
                        print(f"Forcing single speaker due to dominant speaker ratio: {dominant_speaker_ratio:.3f}")
            
            # Create speaker embeddings dictionary
            unique_labels = list(set(speaker_labels))
            updated_speakers = {}
            for label in unique_labels:
                speaker_name = f"Speaker{label + 1}"
                # Use the first embedding for this speaker as representative
                first_occurrence = speaker_labels.index(label)
                updated_speakers[speaker_name] = embeddings[first_occurrence]
            
            # Format transcription with speaker labels and create structured response
            speaker_response = self._create_speaker_response(
                transcription_data, speaker_labels, chunk_timestamps, updated_speakers
            )
            
            return speaker_response
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    def _create_speaker_response(
        self, 
        transcription_data: Dict[str, Any], 
        speaker_labels: List[int], 
        chunk_timestamps: List[Tuple[float, float]],
        speaker_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Create structured speaker detection response with all required data."""
        
        # Extract words with timestamps from transcription data
        words = transcription_data.get("words", [])
        if not words and "segments" in transcription_data:
            # Handle segment-based transcription format
            for segment in transcription_data["segments"]:
                if "words" in segment:
                    words.extend(segment["words"])
        
        # Create speaker information
        unique_labels = list(set(speaker_labels))
        speakers = []
        speaker_segments = []
        speaker_words = []
        
        # Calculate speaker statistics and create speaker objects
        for label in unique_labels:
            speaker_id = f"Speaker{label + 1}"
            
            # Calculate speaking time and word count for this speaker
            speaking_time = 0.0
            word_count = 0
            
            for i, chunk_label in enumerate(speaker_labels):
                if chunk_label == label:
                    start, end = chunk_timestamps[i]
                    speaking_time += (end - start)
            
            speakers.append({
                "speaker_id": speaker_id,
                "speaker_label": speaker_id,
                "confidence": 0.95,  # Default confidence
                "total_speaking_time": speaking_time,
                "word_count": word_count  # Will be updated below
            })
        
        # Process words and create segments
        if words:
            current_speaker = None
            current_segment_words = []
            current_segment_start = None
            
            for word_info in words:
                word = word_info.get("word", "").strip()
                word_start = word_info.get("start", 0)
                word_end = word_info.get("end", word_start)
                word_index = word_info.get("word_index", 0)
                
                # Find which speaker segment this word belongs to
                assigned_speaker_idx = self._find_speaker_for_timestamp(word_start, chunk_timestamps, speaker_labels)
                assigned_speaker_id = f"Speaker{assigned_speaker_idx + 1}"
                
                # Add to speaker words list
                speaker_words.append({
                    "word_index": word_index,
                    "speaker_id": assigned_speaker_id,
                    "word": word,
                    "start_time": word_start,
                    "end_time": word_end
                })
                
                # Update word count for this speaker
                for speaker in speakers:
                    if speaker["speaker_id"] == assigned_speaker_id:
                        speaker["word_count"] += 1
                        break
                
                # Handle segment creation
                if current_speaker != assigned_speaker_id:
                    # Finish previous segment
                    if current_speaker and current_segment_words:
                        speaker_segments.append({
                            "start_time": current_segment_start,
                            "end_time": current_segment_words[-1].get("end", current_segment_start),
                            "text": " ".join([w.get("word", "") for w in current_segment_words]),
                            "speaker_id": current_speaker,
                            "speaker_label": current_speaker,
                            "confidence": 0.95
                        })
                    
                    # Start new segment
                    current_speaker = assigned_speaker_id
                    current_segment_words = [word_info]
                    current_segment_start = word_start
                else:
                    # Continue current segment
                    current_segment_words.append(word_info)
            
            # Add the final segment
            if current_speaker and current_segment_words:
                speaker_segments.append({
                    "start_time": current_segment_start,
                    "end_time": current_segment_words[-1].get("end", current_segment_start),
                    "text": " ".join([w.get("word", "") for w in current_segment_words]),
                    "speaker_id": current_speaker,
                    "speaker_label": current_speaker,
                    "confidence": 0.95
                })
        
        # Generate formatted transcription text
        formatted_lines = []
        current_speaker = None
        current_line = []
        
        for segment in speaker_segments:
            speaker_id = segment["speaker_id"]
            text = segment["text"]
            
            if speaker_id != current_speaker:
                # New speaker, finish current line and start new one
                if current_line:
                    formatted_lines.append(f"{current_speaker}: {' '.join(current_line)}")
                current_speaker = speaker_id
                current_line = [text]
            else:
                current_line.append(text)
        
        # Add the last line
        if current_line and current_speaker:
            formatted_lines.append(f"{current_speaker}: {' '.join(current_line)}")
        
        speaker_separated_text = "\n".join(formatted_lines)
        
        # Return structured response
        return {
            "speakers": speakers,
            "speaker_segments": speaker_segments,
            "speaker_words": speaker_words,
            "speaker_separated_text": speaker_separated_text,
            "detected_speakers": len(unique_labels)
        }
    
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
    
    def _smooth_speaker_transitions(
        self, 
        speaker_labels: List[int], 
        chunk_timestamps: List[Tuple[float, float]],
        window_size: int = 3
    ) -> List[int]:
        """
        Apply median filtering to smooth speaker transitions and reduce fragmentation.
        
        Args:
            speaker_labels: List of speaker labels for each chunk
            chunk_timestamps: Timestamps for each chunk
            window_size: Size of the smoothing window (should be odd)
            
        Returns:
            Smoothed speaker labels
        """
        if len(speaker_labels) <= window_size:
            return speaker_labels
        
        smoothed = speaker_labels.copy()
        half_window = window_size // 2
        
        for i in range(half_window, len(speaker_labels) - half_window):
            # Get window of speaker labels
            window = speaker_labels[i - half_window:i + half_window + 1]
            
            # Find most common speaker in window
            from collections import Counter
            most_common = Counter(window).most_common(1)[0][0]
            
            # Only change if the current label is different and isolated
            # (surrounded by segments of the same different speaker)
            current = speaker_labels[i]
            prev_speaker = speaker_labels[i-1] if i > 0 else current
            next_speaker = speaker_labels[i+1] if i < len(speaker_labels)-1 else current
            
            # If current speaker is isolated (different from neighbors) and
            # the window suggests a different speaker, smooth it
            if (current != prev_speaker and current != next_speaker and 
                prev_speaker == next_speaker and most_common == prev_speaker):
                smoothed[i] = most_common
        
        return smoothed
    
    def _check_dominant_speaker(
        self, 
        speaker_labels: List[int], 
        chunk_timestamps: List[Tuple[float, float]]
    ) -> float:
        """
        Check if one speaker dominates the conversation by time.
        
        Args:
            speaker_labels: List of speaker labels for each chunk
            chunk_timestamps: Timestamps for each chunk
            
        Returns:
            Ratio of the dominant speaker's speaking time (0.0 to 1.0)
        """
        if not speaker_labels or not chunk_timestamps:
            return 0.0
        
        # Calculate speaking time for each speaker
        speaker_times = {}
        for label, (start, end) in zip(speaker_labels, chunk_timestamps):
            duration = end - start
            speaker_times[label] = speaker_times.get(label, 0) + duration
        
        if not speaker_times:
            return 0.0
        
        # Find the dominant speaker's ratio
        total_time = sum(speaker_times.values())
        max_time = max(speaker_times.values())
        
        return max_time / total_time if total_time > 0 else 0.0
