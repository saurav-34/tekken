import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
from tqdm import tqdm
import json
import csv
import re
import time
from pathlib import Path

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct"
FRAMES_DIR = "round_004"              # Directory containing frame images
OUTPUT_CSV = "punch_detection_analysis.csv" # Output file for punch detection
LIVE_RESULTS_FILE = "live_punch_results.txt" # Live results file for monitoring

# Sequence and frame processing settings
SEQUENCE_LENGTH = 8  # Frames to analyze together
FRAME_STEP = 2      # Step between sequences (for overlap)
FRAME_INTERVAL = 1   # Consecutive frames for smooth temporal analysis

# --- STRICT LEFT/RIGHT PUNCH DIFFERENTIATION PROMPT ---
PROMPT_TEMPLATE = """
You are a Tekken 3 expert specializing in precise left/right punch detection. Player 1 is on the LEFT side of screen.

**CRITICAL: PLAYER ORIENTATION**
- Player 1 (LEFT character) faces RIGHT toward opponent
- Player 1's LEFT ARM = character's arm closer to screen edge
- Player 1's RIGHT ARM = character's arm closer to opponent

**STRICT PUNCH CRITERIA:**
❌ **NOT VALID PUNCHES:**
- Standing guard positions (arms up defensively)
- Walking/movement animations
- Blocking motions
- Arms in ready stance without forward attacking motion
- ANY arm position that's not actively attacking forward
- Defensive postures or neutral stances

✅ **VALID LEFT PUNCH:**
- Player 1's LEFT arm (closer to screen edge) actively extends toward opponent
- Clear forward attacking motion from neutral to extended position
- Aggressive striking motion, not defensive
- Arm travels from body toward opponent's location

✅ **VALID RIGHT PUNCH:**
- Player 1's RIGHT arm (closer to opponent) actively extends toward opponent  
- Clear forward attacking motion from neutral to extended position
- Aggressive striking motion, not defensive
- Arm travels from body toward opponent's location

**DETECTION RULES:**
1. **AGGRESSIVE MOTION REQUIRED**: Must see clear attacking intention
2. **FORWARD EXTENSION**: Arm must travel toward opponent, not just move
3. **COMPLETE CYCLE**: Neutral → Extension → Impact → Retraction
4. **NO DEFENSIVE MOVES**: Ignore blocks, guards, or defensive positions
5. **ORIENTATION AWARENESS**: Distinguish left vs right based on Player 1's body position

**COMMON FALSE POSITIVES TO AVOID:**
- Character standing in fighting stance = NOT A PUNCH
- Arms raised for blocking = NOT A PUNCH  
- Walking with arms moving = NOT A PUNCH
- Defensive arm positions = NOT A PUNCH
- Any non-aggressive arm movement = NOT A PUNCH

**VALIDATION CHECKLIST:**
□ Is Player 1 actively attacking (not defending)?
□ Is the arm extending toward the opponent aggressively?
□ Can you clearly identify which arm (left vs right) is attacking?
□ Is this a complete attack sequence (not just a pose)?

**JSON RESPONSE:**
{
  "sequence_analysis": "Detailed description of what Player 1 is actually doing",
  "completed_punches": [
    {
      "frame_start": 45,
      "frame_impact": 47, 
      "frame_end": 49,
      "punch_type": "left",
      "description": "Player 1's left arm extends aggressively toward opponent"
    }
  ]
}

**For non-attacking movements:**
{
  "sequence_analysis": "Player 1 in defensive stance/walking/blocking - no aggressive punches",
  "completed_punches": []
}

Analyze with EXTREME PRECISION for left/right punch differentiation:
"""

def get_frame_number(frame_filename):
    """Extract frame number from filename (e.g., 'frame_120.jpg' -> 120)."""
    match = re.search(r'(\d+)', frame_filename)
    return int(match.group(1)) if match else 0

def update_live_results(seq_idx, total_sequences, total_punches, left_punches, right_punches, elapsed_time):
    """Update live results file for real-time monitoring."""
    with open(LIVE_RESULTS_FILE, 'w') as f:
        f.write("🥊 TEKKEN PUNCH DETECTION - LIVE RESULTS 🥊\n")
        f.write("="*50 + "\n\n")
        f.write(f"📊 Progress: {seq_idx + 1}/{total_sequences} sequences ({((seq_idx + 1)/total_sequences)*100:.1f}%)\n")
        f.write(f"👊 Total Punches Detected: {total_punches}\n")
        f.write(f"👈 Left Punches: {left_punches}\n")
        f.write(f"👉 Right Punches: {right_punches}\n")
        f.write(f"⏱️  Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n")
        f.write(f"🕐 Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if total_punches > 0:
            f.write(f"📈 Punch Rate: {total_punches/(seq_idx + 1):.2f} punches per sequence\n")
            f.write(f"📊 L/R Ratio: {left_punches}:{right_punches}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("Monitor this file for real-time updates!")

def write_csv_header(csv_path):
    """Initialize the CSV file with headers for complete punch sequence analysis."""
    fieldnames = [
        'sequence_id', 
        'start_frame', 
        'end_frame', 
        'completed_punches',
        'left_punches',
        'right_punches',
        'punch_impact_frames',
        'sequence_analysis',
        'punch_details',
        'raw_json_response',
        'frames_analyzed'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def create_sequences(frame_list, sequence_length, step, frame_interval=1):
    """Create sequences of frames for temporal analysis."""
    sequences = []
    sorted_frames = sorted(frame_list, key=get_frame_number)
    
    for i in range(0, len(sorted_frames), step):
        sequence = []
        for j in range(sequence_length):
            frame_idx = i + (j * frame_interval)
            if frame_idx < len(sorted_frames):
                sequence.append(sorted_frames[frame_idx])
        
        if len(sequence) == sequence_length:
            sequences.append(sequence)
            
    return sequences

def extract_json_from_response(response_text):
    """Extract and clean JSON from model response with robust error handling."""
    try:
        # Clean the response text
        cleaned = response_text.strip()
        
        # Remove code block markers
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        
        # Find the first complete JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
        if not json_match:
            # Try to find any JSON-like structure
            json_match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
            
        if not json_match:
            raise ValueError("No JSON object found in response")
        
        json_content = json_match.group(0)
        
        # Parse the JSON
        result = json.loads(json_content)
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Problematic JSON content: {json_content[:200] if 'json_content' in locals() else 'N/A'}")
        return {
            "visual_summary": f"JSON decode error: {e}",
            "punches_detected": []
        }
    except Exception as e:
        print(f"JSON extraction error: {e}")
        print(f"Raw response excerpt: {response_text[:300] if response_text else 'N/A'}")
        # Return a default error structure
        return {
            "visual_summary": f"Error parsing JSON: {e}",
            "punches_detected": []
        }

def analyze_punch_detection(analysis_result):
    """Parse the model's complete punch sequence analysis with validation."""
    try:
        if isinstance(analysis_result, str):
            analysis = extract_json_from_response(analysis_result)
        else:
            analysis = analysis_result
        
        # Handle both old format and new complete sequence format
        punches = analysis.get('completed_punches', analysis.get('punches_detected', []))
        
        # Additional validation to filter false positives
        validated_punches = []
        for punch in punches:
            description = punch.get('description', '').lower()
            sequence_analysis = analysis.get('sequence_analysis', '').lower()
            
            # Filter out obvious false positives
            false_positive_keywords = [
                'standing', 'guard', 'block', 'defensive', 'stance', 'ready position',
                'walking', 'moving', 'idle', 'neutral position', 'raised arms'
            ]
            
            # Check if description or analysis contains false positive indicators
            is_false_positive = any(keyword in description or keyword in sequence_analysis 
                                  for keyword in false_positive_keywords)
            
            # Only include if it seems like a genuine attack
            attack_keywords = ['attack', 'extend', 'toward opponent', 'aggressive', 'striking', 'forward']
            has_attack_indicators = any(keyword in description or keyword in sequence_analysis 
                                      for keyword in attack_keywords)
            
            if not is_false_positive and has_attack_indicators:
                validated_punches.append(punch)
        
        punch_count = len(validated_punches)
        
        # Extract frame information - handle both formats
        if validated_punches and 'frame_impact' in validated_punches[0]:
            # New format with complete sequence tracking
            punch_frames = [p.get('frame_impact', 'N/A') for p in validated_punches]
        else:
            # Legacy format
            punch_frames = [p.get('frame_number', 'N/A') for p in validated_punches]
        
        # Separate left and right punches for detailed analysis
        left_punches = [p for p in validated_punches if p.get('punch_type', '').lower() == 'left']
        right_punches = [p for p in validated_punches if p.get('punch_type', '').lower() == 'right']
        
        # Get analysis description with multiple fallback options
        sequence_analysis = (analysis.get('sequence_analysis') or 
                           analysis.get('motion_analysis') or 
                           'N/A')
        
        return {
            'punch_count': punch_count,
            'left_punch_count': len(left_punches),
            'right_punch_count': len(right_punches),
            'punch_frames': punch_frames,
            'sequence_analysis': sequence_analysis,
            'punch_details': validated_punches,
            'raw_json': analysis
        }
    except Exception as e:
        print(f"Error parsing punch detection analysis: {e}")
        return {
            'punch_count': 0,
            'left_punch_count': 0,
            'right_punch_count': 0,
            'punch_frames': [],
            'sequence_analysis': f'Error: {e}',
            'punch_details': [],
            'raw_json': {'error': str(e), 'raw_response': str(analysis_result)[:500]}
        }

def main():
    """Main function to run punch detection analysis."""
    print("🚀 Initializing Tekken 3 Punch Detection Analysis...")
    
    # Initialize model
    print("🤖 Loading VLM model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        ).eval()
        print("✅ Model loaded with Flash Attention 2.")
    except Exception as e:
        print(f"⚠️ Flash Attention 2 not available, falling back to default. Reason: {e}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        ).eval()
        print("✅ Model loaded successfully (default attention).")
    
    # Load frames
    try:
        frames = sorted(os.listdir(FRAMES_DIR))
        print(f"🎬 Found {len(frames)} frames in '{FRAMES_DIR}'")
    except FileNotFoundError:
        print(f"❌ Error: Directory '{FRAMES_DIR}' not found.")
        return
    
    # Create sequences
    sequences = create_sequences(frames, SEQUENCE_LENGTH, FRAME_STEP, FRAME_INTERVAL)
    print(f"📋 Created {len(sequences)} sequences of {SEQUENCE_LENGTH} frames each.")
    
    # Initialize CSV
    write_csv_header(OUTPUT_CSV)
    print(f"📄 Initialized results file: {OUTPUT_CSV}")
    
    # Initialize live results file
    with open(LIVE_RESULTS_FILE, 'w') as f:
        f.write("🥊 TEKKEN PUNCH DETECTION - STARTING... 🥊\n")
        f.write("Analysis beginning shortly...\n")
    print(f"📺 Live results available at: {LIVE_RESULTS_FILE}")
    
    # Processing variables
    total_punches_detected = 0
    total_left_punches = 0
    total_right_punches = 0
    start_time = time.time()
    
    for seq_idx, frame_sequence in enumerate(tqdm(sequences, desc="Analyzing Sequences")):
        try:
            # Load images for the sequence
            images = [Image.open(os.path.join(FRAMES_DIR, fname)).convert('RGB') for fname in frame_sequence]
            
            # Create message content with temporal emphasis
            frame_numbers = [get_frame_number(fname) for fname in frame_sequence]
            content = [{"type": "text", "text": f"TEMPORAL SEQUENCE ANALYSIS: Frames {frame_numbers[0]} to {frame_numbers[-1]} (in chronological order)"}]
            
            for i, img in enumerate(images):
                frame_num = frame_numbers[i]
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": f"Frame #{frame_num} (sequence position {i+1}/{len(images)})"})
            
            content.append({"type": "text", "text": PROMPT_TEMPLATE})
            
            messages = [{"role": "user", "content": content}]
            
            # Generate analysis
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=images, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Analyze the model's response
            punch_analysis = analyze_punch_detection(response)
            total_punches_detected += punch_analysis['punch_count']
            total_left_punches += punch_analysis['left_punch_count']
            total_right_punches += punch_analysis['right_punch_count']
            
            # 🔴 LIVE RESULTS DISPLAY 🔴
            if punch_analysis['punch_count'] > 0:
                print(f"\n🥊 COMPLETE PUNCH SEQUENCES in seq_{seq_idx:03d} (frames {get_frame_number(frame_sequence[0])}-{get_frame_number(frame_sequence[-1])}):")
                print(f"   👊 Complete Sequences: {punch_analysis['punch_count']} | Left: {punch_analysis['left_punch_count']} | Right: {punch_analysis['right_punch_count']}")
                print(f"   📍 Impact Frames: {', '.join(map(str, punch_analysis['punch_frames']))}")
                if punch_analysis['sequence_analysis'] != 'N/A':
                    print(f"   📝 Motion: {punch_analysis['sequence_analysis'][:80]}...")
                print(f"   📊 Running Total: {total_punches_detected} complete sequences ({total_left_punches}L, {total_right_punches}R)")
            else:
                print(f"✓ seq_{seq_idx:03d}: No complete punch sequences | Total so far: {total_punches_detected}")
            
            # Update live results file
            update_live_results(seq_idx, len(sequences), total_punches_detected, total_left_punches, total_right_punches, time.time() - start_time)
            
            # Prepare data for CSV with complete sequence analysis
            result = {
                'sequence_id': f"seq_{seq_idx:03d}",
                'start_frame': get_frame_number(frame_sequence[0]),
                'end_frame': get_frame_number(frame_sequence[-1]),
                'completed_punches': punch_analysis['punch_count'],
                'left_punches': punch_analysis['left_punch_count'],
                'right_punches': punch_analysis['right_punch_count'],
                'punch_impact_frames': ', '.join(map(str, punch_analysis['punch_frames'])),
                'sequence_analysis': punch_analysis['sequence_analysis'],
                'punch_details': json.dumps(punch_analysis['punch_details']),
                'raw_json_response': json.dumps(punch_analysis['raw_json']),
                'frames_analyzed': ', '.join(frame_sequence)
            }
            
            # Write to CSV
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                writer.writerow(result)
            
            # 📊 PERIODIC PROGRESS SUMMARY (every 10 sequences)
            if (seq_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time_per_seq = elapsed / (seq_idx + 1)
                remaining_sequences = len(sequences) - (seq_idx + 1)
                eta = remaining_sequences * avg_time_per_seq
                print(f"\n📈 PROGRESS UPDATE - Completed {seq_idx + 1}/{len(sequences)} sequences")
                print(f"   🎯 Total Punches Found: {total_punches_detected} ({total_left_punches}L + {total_right_punches}R)")
                print(f"   ⏱️  Elapsed: {time.strftime('%M:%S', time.gmtime(elapsed))} | ETA: {time.strftime('%M:%S', time.gmtime(eta))}")
                print("   " + "="*50)

        except Exception as e:
            print(f"❌ Error processing sequence {seq_idx}: {e}")
        finally:
            if 'images' in locals():
                for img in images: img.close()
            del images, inputs, generated_ids
            torch.cuda.empty_cache()
            
    # Final summary with temporal analysis
    elapsed_time = time.time() - start_time
    
    # Final live results update
    with open(LIVE_RESULTS_FILE, 'w') as f:
        f.write("🥊 TEKKEN PUNCH DETECTION - COMPLETED! 🥊\n")
        f.write("="*50 + "\n\n")
        f.write("✅ FINAL RESULTS:\n")
        f.write(f"📊 Total Sequences Processed: {len(sequences)}\n")
        f.write(f"👊 Total Punches Detected: {total_punches_detected}\n")
        f.write(f"👈 Left Punches: {total_left_punches}\n")
        f.write(f"👉 Right Punches: {total_right_punches}\n")
        f.write(f"⏱️  Total Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}\n")
        f.write(f"🕐 Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if total_punches_detected > 0:
            f.write(f"📈 Average Punch Rate: {total_punches_detected/len(sequences):.2f} punches per sequence\n")
            f.write(f"📊 Left/Right Ratio: {total_left_punches}:{total_right_punches}\n")
        
        f.write(f"\n💾 Detailed results saved to: {OUTPUT_CSV}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("Analysis Complete!")
    
    print("\n" + "="*40)
    print("🎯 TEMPORAL PUNCH DETECTION SUMMARY 🎯")
    print("="*40)
    print(f"✅ Processed: {len(sequences)} sequences")
    print(f"👊 Total punches detected: {total_punches_detected}")
    print(f"👈 Left punches: {total_left_punches}")
    print(f"👉 Right punches: {total_right_punches}")
    print(f"⏱️ Total processing time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
    print(f"💾 Results saved to: {OUTPUT_CSV}")
    print(f"📺 Live results file: {LIVE_RESULTS_FILE}")
    print("🎮 Temporal punch detection analysis complete!")

if __name__ == '__main__':
    main()