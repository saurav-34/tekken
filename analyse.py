import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
from tqdm import tqdm
import json
import csv
import re
import time
import datetime
from pathlib import Path

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct"
FRAMES_DIR = "dataf/step_500/round_058_generated"  # Renamed from ORIGINAL_FRAMES_DIR
OUTPUT_CSV = "action_sequence_analysis.csv" # New output file

# Sequence and frame processing settings
# --- NOTE: This analysis assumes the video is only 30 seconds long. ---
# TOTAL_EXPECTED_FRAMES is calculated for a 30-second video at the given FPS.

# OPTIMAL SETTINGS FOR FIGHTING GAME ANALYSIS:
# - Longer sequences capture complete move combinations
# - Smaller steps ensure we don't miss important transitions
# - Consecutive frames (interval=1) preserve temporal continuity
SEQUENCE_LENGTH = 8  # Increased from 3 - captures complete combos/moves (8 frames ≈ 0.27 seconds at 30fps)
FRAME_STEP = 4       # Decreased from 10 - less overlap, more coverage (4 frames ≈ 0.13 seconds gap)
FRAME_INTERVAL = 3 # Decreased from 4 - consecutive frames for smooth temporal flow

EXPECTED_FPS = 30
TOTAL_EXPECTED_FRAMES = 30 * EXPECTED_FPS

# --- ENHANCED SYSTEM PROMPT FOR ACTION DETECTION ---
PROMPT_TEMPLATE = """
 
 You are a methodical and cautious fighting game analyst. Your primary goal is to avoid misinterpretation by following a strict, step-by-step reasoning process.

**Reasoning Process:**
1.  **Motion Analysis:** Describe the character's overall motion (e.g., moving forward, extending a limb).
2.  **Intent Analysis:** Based on the motion, what is the likely intent (e.g., to strike, to remain ready)?
3.  **Impact Analysis:** Look for visual effects. Are there **red hit sparks** indicating a successful, connecting hit? This is the most important step for confirming an attack.
4.  **Final Classification:** Based on the above, provide the final JSON.

**CRITICAL RULES:**
1.  A character's idle stance must be classified as `"action_type": "Idle"`.
2.  An attack is only confirmed as a successful hit (`"hit_confirmation": "Confirmed"`) if red impact effects are visible.
3.  If a punch or kick motion is visible but there are **no red effects**, the attack is a miss or is blocked, and must be classified as `"hit_confirmation": "No"`.

Provide your analysis and classification in a single JSON object.

**Example 1: A Successful Punch**
{
  "reasoning": "The character moves forward and extends their right arm with offensive intent. Red hit sparks are clearly visible on the opponent, confirming a successful hit.",
  "action_type": "Attack",
  "attack_category": "Punch",
  "hit_confirmation": "Confirmed",
  "description": "Player 1 lands a successful right punch, confirmed by red impact sparks."
}

**Example 2: A Missed/Blocked Punch**
{
  "reasoning": "The character extends their right arm in a punch-like motion, but there are no red impact effects. The attack did not connect successfully.",
  "action_type": "Attack",
  "attack_category": "Punch",
  "hit_confirmation": "No",
  "description": "Player 1 throws a right punch that does not connect or is blocked."
}

**Example 3: Idle Stance**
{
  "reasoning": "The character is shifting weight in place with no clear offensive intent and no impact effects are visible. This is a neutral stance.",
  "action_type": "Idle",
  "attack_category": "None",
  "hit_confirmation": "N/A",
  "description": "Player 1 is standing in their neutral idle stance."
}
{{"action_type": "PUNCH/KICK/OTHER", "player2_interrupts": "YES/NO", "action_success": "COMPLETED/INTERRUPTED/BLOCKED", "description": "Brief description of what happens in the sequence"}}
"""

def write_csv_header(csv_path):
    """Initializes the CSV file with a header."""
    fieldnames = ['sequence', 'action_type', 'player2_interrupts', 'action_success', 'description', 'frames']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def create_sequences(frame_list, sequence_length, step, frame_interval=1):
    """
    Create sequences of frames for temporal analysis.
    Args:
        frame_list: List of frame filenames (should be numerically sorted)
        sequence_length: Number of frames per sequence
        step: Step size between sequence starting points
        frame_interval: Interval between frames within each sequence (1 = consecutive, 2 = every 2nd frame, etc.)
    
    Returns:
        List of sequences, where each sequence is a list of frame filenames
    """
    sequences = []
    
    # Ensure frames are properly sorted numerically (not just alphabetically)
    def extract_frame_number(filename):
        import re
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    sorted_frames = sorted(frame_list, key=extract_frame_number)
    
    for i in range(0, len(sorted_frames), step):
        sequence = []
        for j in range(sequence_length):
            frame_idx = i + (j * frame_interval)
            if frame_idx < len(sorted_frames):
                sequence.append(sorted_frames[frame_idx])
            else:
                break
        
        # Only add sequences that have the full length
        if len(sequence) == sequence_length:
            sequences.append(sequence)
        else:
            break  # Stop if we can't create a full sequence
    
    return sequences

def display_progress_summary(seq_idx, total_sequences, processed_count, error_count, start_time, last_result=None):
    """Display live progress summary with ETA and latest result."""
    current_time = time.time()
    elapsed = current_time - start_time
    
    if processed_count > 0:
        avg_time_per_sequence = elapsed / (seq_idx + 1)
        remaining_sequences = total_sequences - (seq_idx + 1)
        eta_seconds = remaining_sequences * avg_time_per_sequence
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        completion_rate = (processed_count / (seq_idx + 1)) * 100
        
        print(f"\n" + "="*80)
        print(f"📊 LIVE PROGRESS UPDATE - Sequence {seq_idx + 1}/{total_sequences}")
        print(f"✅ Successful: {processed_count} | ❌ Errors: {error_count} | Success Rate: {completion_rate:.1f}%")
        print(f"⏱️  Elapsed: {str(datetime.timedelta(seconds=int(elapsed)))} | ETA: {eta_str}")
        print(f"🔄 Average time per sequence: {avg_time_per_sequence:.1f}s")
        
        if last_result:
            print(f"📝 Latest result: {last_result[:100]}{'...' if len(last_result) > 100 else ''}")
        
        print("="*80 + "\n")

def check_gpu_memory():
    """Check available GPU memory."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")

def check_system_memory():
    """Check system RAM usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"System RAM: {mem.used / 1024**3:.2f}GB used / {mem.total / 1024**3:.2f}GB total ({mem.percent:.1f}%)")
    except ImportError:
        print("psutil not available for system memory monitoring")

def check_processing_feasibility(total_sequences):
    """Check if the processing task is feasible and get user confirmation."""
    estimated_time = total_sequences * 45  # 45 seconds per sequence estimate
    estimated_hours = estimated_time / 3600
    
    print(f"\n⚠️  Processing Feasibility Check:")
    print(f"   Total sequences to process: {total_sequences}")
    print(f"   Estimated time: {estimated_time:.0f} seconds ({estimated_hours:.1f} hours)")
    print(f"   This will analyze {total_sequences * 8} total frame combinations")
    
    if estimated_hours > 2:
        print(f"⚠️  Warning: This will take over {estimated_hours:.1f} hours to complete!")
        
    response = input("\nDo you want to proceed with this analysis? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Analysis cancelled by user.")
        return False
    
    print("✅ Proceeding with analysis...")
    return True

def main():
    """Main function to load model and run the action analysis process."""
    print("🚀 Initializing local model and processor...")
    
    print("Initial GPU memory status:")
    check_gpu_memory()
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading model '{MODEL_ID}'. This may take several minutes and significant VRAM...")
    # Note: The Qwen2-VL-72B model is very large. While you've mentioned compute is not an issue,
    # if you encounter memory problems on a different machine, you might consider a smaller model variant.
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"
        ).eval()
    except Exception as e:
        print(f"Flash attention not available, falling back to default: {e}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).eval()
    
    print("✅ Model loaded successfully onto GPU.")
    print("Memory status after model loading:")
    check_gpu_memory()

    try:
        frames = sorted(os.listdir(FRAMES_DIR))
        print(f"Found {len(frames)} frames in '{FRAMES_DIR}'")
    except FileNotFoundError:
        print(f"Error: Directory '{FRAMES_DIR}' not found.")
        return

    sequences = create_sequences(frames, SEQUENCE_LENGTH, FRAME_STEP, FRAME_INTERVAL)
    print(f"Created {len(sequences)} sequences of {SEQUENCE_LENGTH} frames each")
    print(f"🔍 Frame processing details:")
    print(f"   Total frames found: {len(frames)}")
    print(f"   Sequence length: {SEQUENCE_LENGTH} frames (≈{SEQUENCE_LENGTH/EXPECTED_FPS:.2f} seconds at {EXPECTED_FPS}fps)")
    print(f"   Frame step: {FRAME_STEP} frames (≈{FRAME_STEP/EXPECTED_FPS:.2f} seconds gap between sequence starts)")
    print(f"   Frame interval: {FRAME_INTERVAL} ({'consecutive' if FRAME_INTERVAL == 1 else f'every {FRAME_INTERVAL}th'} frame)")
    print(f"   Coverage: From frame {frames[0] if frames else 'N/A'} to {sequences[-1][-1] if sequences else 'N/A'}")
    print(f"   Expected processing time estimate: {len(sequences) * 45:.1f} seconds (assuming 45s per sequence)")
    
    # Check if processing is feasible
    if not check_processing_feasibility(len(sequences)):
        return
    
    # Initialize processing variables
    start_seq = 0
    processed_count = 0
    error_count = 0
    
    # Always start fresh - write CSV header
    write_csv_header(OUTPUT_CSV)
    print(f"📄 Initialized results file: {OUTPUT_CSV}")
    
    total_image_loading_time = 0
    total_inference_time = 0
    script_start_time = time.time()
    
    for seq_idx, frame_sequence in enumerate(sequences):
        sequence_start_time = time.time()
        last_result = None
        
        # Time image loading with proper error handling
        image_loading_start = time.time()
        images = []
        try:
            for fname in frame_sequence:
                path = os.path.join(FRAMES_DIR, fname)
                img = Image.open(path)
                # Convert to RGB to ensure consistency and reduce memory usage
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
        except Exception as e:
            # Clean up any partially loaded images
            for img in images:
                if hasattr(img, 'close'):
                    img.close()
            print(f"Error loading images for sequence {seq_idx}: {e}")
            continue
        
        image_loading_time = time.time() - image_loading_start
        total_image_loading_time += image_loading_time
        
        content = []
        content.append({"type": "text", "text": f"TEMPORAL SEQUENCE ANALYSIS - {len(images)} consecutive frames:"})
        
        # Add frame labels for better temporal understanding
        for i, img in enumerate(images, 1):
            content.append({"type": "text", "text": f"Frame {i}/{len(images)} (chronological order):"})
            content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": PROMPT_TEMPLATE})
        
        messages = [{"role": "user", "content": content}]

        # Time model inference
        inference_start = time.time()
        try:
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=images, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=300,  # Increased from 200 for more detailed temporal descriptions
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    temperature=0.1,
                    repetition_penalty=1.1,  # Prevent repetitive descriptions
                )
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Immediate cleanup of large tensors
                del inputs
                del generated_ids
                torch.cuda.empty_cache()
            
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            print(f"Raw response for sequence {seq_idx}: '{response[:200]}...'")
            
            if not response or len(response.strip()) < 10:
                raise ValueError("Empty or invalid response from model")
            
            json_str = response.strip().replace("```json", "").replace("```", "")
            
            # Use regex to find the JSON object in the response
            analysis = None
            json_match = re.search(r'\{.*"action_type":.*\}', json_str, re.DOTALL)
            
            if json_match:
                try:
                    analysis = json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error after regex match: {e}")
                    raise ValueError(f"JSON parsing failed: {e}")
            else:
                raise ValueError("Could not find valid JSON in the model's response.")
            
            sequence_name = f"seq_{seq_idx:03d}_{frame_sequence[0]}_to_{frame_sequence[-1]}"
            
            result = {
                'sequence': sequence_name,
                'action_type': analysis.get('action_type', 'N/A'),
                'player2_interrupts': analysis.get('player2_interrupts', 'N/A'),
                'action_success': analysis.get('action_success', 'N/A'),
                'description': analysis.get('description', 'N/A'),
                'frames': ', '.join(frame_sequence)
            }
            
            # Write result immediately to CSV
            fieldnames = ['sequence', 'action_type', 'player2_interrupts', 'action_success', 'description', 'frames']
            with open(OUTPUT_CSV, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)
            
            processed_count += 1
            last_result = f"{result['action_type']} - {result['description']}"
            
            sequence_total_time = time.time() - sequence_start_time
            print(f"✅ Sequence {seq_idx:03d} completed in {sequence_total_time:.2f}s")
            print(f"   📸 Image loading: {image_loading_time:.3f}s")
            print(f"   🧠 Model inference: {inference_time:.2f}s")
            print(f"   🎬 Frame sequence: {frame_sequence[0]} → {frame_sequence[-1]} ({len(frame_sequence)} frames)")
            print(f"   🥊 Action: {result['action_type']}")
            print(f"   🛡️  Player 2 Interrupts: {result['player2_interrupts']}")
            print(f"   ✨ Action Success: {result['action_success']}")
            print(f"   📝 Description: {result['description']}")
            
            # Display live progress and memory status every 3 sequences or if it's been more than 2 minutes
            if (seq_idx + 1) % 3 == 0 or (time.time() - script_start_time) > 120:
                display_progress_summary(seq_idx, len(sequences), processed_count, error_count, script_start_time, last_result)
                print("🔍 Current memory status:")
                check_gpu_memory()
                check_system_memory()
            
        except Exception as e:
            error_count += 1
            print(f"❌ Error processing sequence {seq_idx}: {e}")
            sequence_name = f"seq_{seq_idx:03d}_{frame_sequence[0]}_to_{frame_sequence[-1]}"
            result = {
                'sequence': sequence_name,
                'action_type': 'ERROR',
                'player2_interrupts': 'N/A',
                'action_success': 'N/A',
                'description': f'ERROR: {e}',
                'frames': ', '.join(frame_sequence)
            }
            
            fieldnames = ['sequence', 'action_type', 'player2_interrupts', 'action_success', 'description', 'frames']
            with open(OUTPUT_CSV, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)
        
        finally:
            # Aggressive memory cleanup to prevent leaks
            try:
                # Close and delete images immediately
                if 'images' in locals():
                    for img in images:
                        if hasattr(img, 'close'):
                            img.close()
                    del images
                
                # Clean up large variables
                if 'content' in locals():
                    del content
                if 'messages' in locals():
                    del messages
                if 'text_input' in locals():
                    del text_input
                if 'inputs' in locals():
                    del inputs
                if 'generated_ids' in locals():
                    del generated_ids
                if 'response' in locals():
                    del response
                
                # GPU memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Force garbage collection more frequently for memory-intensive operations
                if (seq_idx + 1) % 5 == 0:  # Changed from 10 to 5
                    import gc
                    gc.collect()
                    print(f"🧹 Memory cleanup performed after sequence {seq_idx}")
                    
            except Exception as cleanup_error:
                print(f"Warning: Error during cleanup: {cleanup_error}")

    # Final cleanup and summary
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Remove checkpoint file if completed successfully
    if os.path.exists('analysis_checkpoint.json') and error_count == 0:
        os.remove('analysis_checkpoint.json')
        print("🗑️  Removed checkpoint file (analysis completed successfully)")

    print(f"\n🎯 FINAL SUMMARY:")
    print(f"📊 Total sequences attempted: {len(sequences[start_seq:])}")
    print(f"✅ Successfully processed: {processed_count} sequences")
    print(f"❌ Errors encountered: {error_count} sequences")
    print(f"💾 Results saved to: {OUTPUT_CSV}")
    
    if processed_count > 0:
        avg_image_time = total_image_loading_time / processed_count
        avg_inference_time = total_inference_time / processed_count
        print(f"\n⏱️  PERFORMANCE BREAKDOWN:")
        print(f"📸 Average image loading time per sequence: {avg_image_time:.3f}s")
        print(f"🧠 Average model inference time per sequence: {avg_inference_time:.2f}s")
        print(f"🔧 Total image loading time: {total_image_loading_time:.2f}s")
        print(f"🔧 Total inference time: {total_inference_time:.2f}s")
    
    print(f"✅ Analysis complete!")

if __name__ == '__main__':
    main()