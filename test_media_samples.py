#!/usr/bin/env python3
"""
Test script to evaluate speaker detection on all media samples.
Since all samples are known to be single-speaker, we can measure false positive rates.
"""

import os
import json
import requests
import time
from pathlib import Path

def test_speaker_detection():
    """Test speaker detection on all available media samples."""
    
    # Configuration
    base_url = "http://localhost:8000"
    media_dir = Path("media")
    
    # Find all audio and JSON pairs
    audio_files = list(media_dir.glob("*.mp3")) + list(media_dir.glob("*.audio"))
    results = []
    
    print(f"🧪 Testing speaker detection on {len(audio_files)} samples")
    print("=" * 60)
    
    for audio_file in audio_files:
        # Find corresponding JSON file
        json_file = audio_file.with_suffix('.json')
        
        if not json_file.exists():
            print(f"⚠️  No transcription found for {audio_file.name}")
            continue
            
        try:
            # Load transcription data
            with open(json_file, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
            
            print(f"\n📁 Testing: {audio_file.name}")
            print(f"   Transcription: {transcription_data.get('text', '')[:50]}...")
            
            # Prepare request
            with open(audio_file, 'rb') as f:
                files = {
                    'audio': (audio_file.name, f.read(), 'audio/mpeg')
                }
                
                data = {
                    'transcription_data': json.dumps(transcription_data)
                }
                
                start_time = time.time()
                
                # Make request
                response = requests.post(
                    f"{base_url}/speaker-identification",
                    files=files,
                    data=data,
                    timeout=120
                )
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    detected_speakers = result.get('detected_speakers', 0)
                    speakers = result.get('speakers', [])
                    
                    # Analyze results
                    is_correct = detected_speakers == 1
                    status = "✅ CORRECT" if is_correct else f"❌ FALSE POSITIVE ({detected_speakers} speakers)"
                    
                    print(f"   Result: {status}")
                    print(f"   Processing time: {processing_time:.2f}s")
                    
                    if not is_correct:
                        print(f"   Speakers detected: {[s['speaker_id'] for s in speakers]}")
                        speaking_times = [f"{s['speaker_id']}: {s['total_speaking_time']:.1f}s" for s in speakers]
                        print(f"   Speaking times: {', '.join(speaking_times)}")
                    
                    # Store result
                    results.append({
                        'file': audio_file.name,
                        'expected_speakers': 1,
                        'detected_speakers': detected_speakers,
                        'correct': is_correct,
                        'processing_time': processing_time,
                        'text_length': len(transcription_data.get('text', '')),
                        'word_count': len(transcription_data.get('words', [])),
                        'speakers': speakers
                    })
                    
                else:
                    print(f"   ❌ ERROR: HTTP {response.status_code}")
                    print(f"   Response: {response.text}")
                    
        except Exception as e:
            print(f"   ❌ EXCEPTION: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = (correct_count / total_count) * 100
        
        print(f"Total samples tested: {total_count}")
        print(f"Correctly identified as single-speaker: {correct_count}")
        print(f"False positives (multi-speaker): {total_count - correct_count}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        if total_count > correct_count:
            print(f"\n🔍 FALSE POSITIVE ANALYSIS:")
            false_positives = [r for r in results if not r['correct']]
            
            for fp in false_positives:
                print(f"   • {fp['file']}: {fp['detected_speakers']} speakers detected")
                print(f"     Text length: {fp['text_length']} chars, Words: {fp['word_count']}")
                
                # Show speaker distribution
                if fp['speakers']:
                    for speaker in fp['speakers']:
                        pct = (speaker['total_speaking_time'] / sum(s['total_speaking_time'] for s in fp['speakers'])) * 100
                        print(f"     {speaker['speaker_id']}: {speaker['total_speaking_time']:.1f}s ({pct:.1f}%)")
        
        avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
        print(f"\nAverage processing time: {avg_processing_time:.2f}s")
        
        # Save detailed results
        with open('speaker_detection_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Detailed results saved to: speaker_detection_test_results.json")
    
    else:
        print("❌ No samples could be processed")

if __name__ == "__main__":
    test_speaker_detection()