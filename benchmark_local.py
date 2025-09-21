import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
from tqdm import tqdm
import json
import csv

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct"
ORIGINAL_FRAMES_DIR = "original"
GENERATED_FRAMES_DIR = "generated"
OUTPUT_CSV = "sequence_benchmark_results.csv"

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 3  # Reduced from 5 to save memory
FRAME_STEP = 3      # Increased step to process fewer sequences
EXPECTED_FPS = 30   # Expected frames per second for 30s rounds
TOTAL_EXPECTED_FRAMES = 30 * EXPECTED_FPS  # 900 frames for 30 seconds

# --- SYSTEM PROMPT & RUBRIC ---
PROMPT_TEMPLATE = """Compare these two Tekken gameplay sequences (Reference vs Generated, {sequence_length} frames each). Rate 1-10 for each metric.

**Quick Analysis:**
1. **Action Fidelity**: Are the exact same moves performed? (10=identical, 1=completely different)
2. **Sequence Flow**: Does the timing and progression match? (10=perfect timing, 1=wrong sequence)  
3. **Position Stability**: Are character positions consistent? (10=stable, 1=teleporting/sliding)

**STRICT SCORING:**
- 10: Perfect match
- 7-9: Minor differences
- 4-6: Noticeable errors
- 1-3: Major failures

Output ONLY the JSON scores:
{{"action_fidelity": X, "sequence_adherence": X, "positional_stability": X}}
"""

def write_result_to_csv(result, csv_path, is_first_write=False):
    """
    Write a single result to CSV in real-time.
    
    Args:
        result: Dictionary containing the result data
        csv_path: Path to the CSV file
        is_first_write: Whether this is the first write (to include header)
    """
    fieldnames = ['sequence', 'action_fidelity', 'sequence_adherence', 'positional_stability', 'frames']
    
    # Write header on first write, append on subsequent writes
    mode = 'w' if is_first_write else 'a'
    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_first_write:
            writer.writeheader()
        writer.writerow(result)

def create_sequences(frame_list, sequence_length, step):
    """
    Create sequences of frames for temporal analysis.
    
    Args:
        frame_list: List of frame filenames
        sequence_length: Number of frames per sequence
        step: Step size between sequences
    
    Returns:
        List of sequences, where each sequence is a list of frame filenames
    """
    sequences = []
    for i in range(0, len(frame_list) - sequence_length + 1, step):
        sequence = frame_list[i:i + sequence_length]
        sequences.append(sequence)
    return sequences

def get_matching_generated_frames(original_sequence, generated_frames_dir):
    """
    Find matching generated frames for an original sequence.
    
    Args:
        original_sequence: List of original frame filenames
        generated_frames_dir: Directory containing generated frames
    
    Returns:
        List of generated frame paths, or None if any frame is missing
    """
    generated_sequence = []
    
    for original_filename in original_sequence:
        frame_num = os.path.splitext(original_filename)[0]
        
        # Try different generated frame naming patterns
        possible_generated_names = [
            f"{frame_num}.png",  # 0000.png
            f"frame_{int(frame_num):05d}.png",  # frame_00000.png
            f"frame_{int(frame_num)+1:05d}.png"  # frame_00001.png (if 1-indexed)
        ]
        
        generated_path = None
        for gen_name in possible_generated_names:
            test_path = os.path.join(generated_frames_dir, gen_name)
            if os.path.exists(test_path):
                generated_path = test_path
                break
        
        if generated_path is None:
            return None  # Missing frame, skip this sequence
        
        generated_sequence.append(generated_path)
    
    return generated_sequence

def check_gpu_memory():
    """Check available GPU memory"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")

def main():
    """Main function to load model and run the sequence benchmarking process."""
    print("🚀 Initializing local model and processor...")
    
    # Check initial memory
    print("Initial GPU memory status:")
    check_gpu_memory()
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading model '{MODEL_ID}'. This may take several minutes and significant VRAM...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use FP16 to save memory
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"  # Use flash attention if available
        ).eval()
    except Exception as e:
        print(f"Flash attention not available, falling back to default: {e}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use FP16 to save memory
            low_cpu_mem_usage=True
        ).eval()
    
    # Check memory after model loading
    print("✅ Model loaded successfully onto GPU.")
    print("Memory status after model loading:")
    check_gpu_memory()

    try:
        original_frames = sorted(os.listdir(ORIGINAL_FRAMES_DIR))
        print(f"Found {len(original_frames)} original frames")
    except FileNotFoundError:
        print(f"Error: Directory '{ORIGINAL_FRAMES_DIR}' or '{GENERATED_FRAMES_DIR}' not found.")
        return

    # Create sequences from original frames
    sequences = create_sequences(original_frames, SEQUENCE_LENGTH, FRAME_STEP)
    print(f"Created {len(sequences)} sequences of {SEQUENCE_LENGTH} frames each")
    
    # Initialize CSV file with header
    fieldnames = ['sequence', 'action_fidelity', 'sequence_adherence', 'positional_stability', 'frames']
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    print(f"📄 Initialized results file: {OUTPUT_CSV}")
    
    results = []
    processed_count = 0
    for seq_idx, original_sequence in enumerate(tqdm(sequences, desc="Processing sequences")):
        # Get matching generated frames for this sequence
        generated_sequence_paths = get_matching_generated_frames(original_sequence, GENERATED_FRAMES_DIR)
        
        if generated_sequence_paths is None:
            print(f"Warning: Skipping sequence {seq_idx}, missing generated frames.")
            continue
        
        # Load all images in the sequence
        original_images = []
        generated_images = []
        
        try:
            for orig_fname in original_sequence:
                orig_path = os.path.join(ORIGINAL_FRAMES_DIR, orig_fname)
                original_images.append(Image.open(orig_path))
            
            for gen_path in generated_sequence_paths:
                generated_images.append(Image.open(gen_path))
            
        except Exception as e:
            print(f"Error loading images for sequence {seq_idx}: {e}")
            continue
        
        # Prepare the conversation format for Qwen2-VL with multiple images
        content = []
        
        # Add reference sequence images
        content.append({"type": "text", "text": "Reference Sequence:"})
        for img in original_images:
            content.append({"type": "image", "image": img})
        
        # Add generated sequence images
        content.append({"type": "text", "text": "Generated Sequence:"})
        for img in generated_images:
            content.append({"type": "image", "image": img})
        
        # Add the prompt
        content.append({"type": "text", "text": PROMPT_TEMPLATE.format(sequence_length=SEQUENCE_LENGTH)})
        
        messages = [{"role": "user", "content": content}]

        try:
            # Process the input
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_images = original_images + generated_images
            inputs = processor(text=[text_input], images=all_images, return_tensors="pt").to(model.device)
            
            # Generate response with increased token limit
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=200,  # Increased from 100
                    do_sample=False, 
                    pad_token_id=processor.tokenizer.eos_token_id,
                    temperature=0.1,  # Add temperature for more consistent outputs
                )
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Clear GPU cache after each generation
                torch.cuda.empty_cache()
            
            # Debug: print the raw response
            print(f"Raw response for sequence {seq_idx}: '{response[:200]}...'")
            
            # Check if response is empty or too short
            if not response or len(response.strip()) < 10:
                print(f"❌ Empty or too short response for sequence {seq_idx}")
                raise ValueError("Empty or invalid response from model")
            
            # Parse JSON response - look for JSON object
            json_str = response.strip().replace("```json", "").replace("```", "")
            
            # Try to extract JSON from the response
            scores = None
            
            # Method 1: Look for complete JSON object
            import re
            json_pattern = r'\{[^{}]*"action_fidelity"\s*:\s*\d+[^{}]*\}'
            json_match = re.search(json_pattern, json_str)
            
            if json_match:
                try:
                    scores = json.loads(json_match.group())
                    print(f"✅ Successfully parsed JSON from pattern match")
                except json.JSONDecodeError:
                    pass
            
            # Method 2: If pattern didn't work, try extracting individual scores
            if scores is None:
                try:
                    action_match = re.search(r'"action_fidelity"\s*:\s*(\d+)', json_str)
                    sequence_match = re.search(r'"sequence_adherence"\s*:\s*(\d+)', json_str)
                    position_match = re.search(r'"positional_stability"\s*:\s*(\d+)', json_str)
                    
                    if action_match and sequence_match and position_match:
                        scores = {
                            'action_fidelity': int(action_match.group(1)),
                            'sequence_adherence': int(sequence_match.group(1)),
                            'positional_stability': int(position_match.group(1))
                        }
                        print(f"✅ Successfully extracted scores from regex")
                    else:
                        raise ValueError("Could not extract scores from response")
                        
                except (ValueError, AttributeError) as e:
                    print(f"❌ Score extraction failed: {e}")
                    raise ValueError(f"JSON parsing failed: {e}")
            
            # Method 3: Last resort - try to parse the whole response as JSON
            if scores is None:
                try:
                    scores = json.loads(json_str.strip())
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"Problematic JSON: '{json_str[:200]}...'")
                    raise ValueError(f"JSON parsing failed: {e}")
            
            sequence_name = f"seq_{seq_idx:03d}_{original_sequence[0]}_to_{original_sequence[-1]}"
            result = {
                'sequence': sequence_name,
                'action_fidelity': scores.get('action_fidelity', 'N/A'),
                'sequence_adherence': scores.get('sequence_adherence', 'N/A'),
                'positional_stability': scores.get('positional_stability', 'N/A'),
                'frames': ', '.join(original_sequence)
            }
            
            # Write result immediately to CSV
            with open(OUTPUT_CSV, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)
            
            results.append(result)
            processed_count += 1
            
            # Print real-time update
            print(f"✅ Sequence {seq_idx:03d} completed - Scores: Action={result['action_fidelity']}, Adherence={result['sequence_adherence']}, Position={result['positional_stability']}")
            
        except Exception as e:
            print(f"❌ Error processing sequence {seq_idx}: {e}")
            sequence_name = f"seq_{seq_idx:03d}_{original_sequence[0]}_to_{original_sequence[-1]}"
            result = {
                'sequence': sequence_name,
                'action_fidelity': 'ERROR',
                'sequence_adherence': 'ERROR',
                'positional_stability': 'ERROR',
                'frames': ', '.join(original_sequence)
            }
            
            # Write error result immediately to CSV
            with open(OUTPUT_CSV, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)
            
            results.append(result)
        
        finally:
            # Clean up images from memory
            try:
                for img in original_images:
                    img.close()
                for img in generated_images:
                    img.close()
                del original_images, generated_images
                torch.cuda.empty_cache()
            except:
                pass

    # Final summary
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"📊 Total sequences processed: {processed_count}/{len(sequences)}")
    
    if results:
        valid_results = [r for r in results if r['action_fidelity'] != 'ERROR']
        error_count = len(results) - len(valid_results)
        
        if valid_results:
            # Calculate average scores
            avg_action = sum(int(r['action_fidelity']) for r in valid_results if str(r['action_fidelity']).isdigit()) / len([r for r in valid_results if str(r['action_fidelity']).isdigit()])
            avg_adherence = sum(int(r['sequence_adherence']) for r in valid_results if str(r['sequence_adherence']).isdigit()) / len([r for r in valid_results if str(r['sequence_adherence']).isdigit()])
            avg_position = sum(int(r['positional_stability']) for r in valid_results if str(r['positional_stability']).isdigit()) / len([r for r in valid_results if str(r['positional_stability']).isdigit()])
            
            print(f"✅ Successfully processed: {len(valid_results)} sequences")
            print(f"❌ Errors encountered: {error_count} sequences")
            print(f"📈 Average Scores:")
            print(f"   Action Fidelity: {avg_action:.2f}/10")
            print(f"   Sequence Adherence: {avg_adherence:.2f}/10") 
            print(f"   Positional Stability: {avg_position:.2f}/10")
            
        print(f"💾 Results continuously saved to: {OUTPUT_CSV}")
        print(f"✅ Benchmarking complete! You can view results in real-time by opening {OUTPUT_CSV}")
    else:
        print("❌ No sequences were processed.")

if __name__ == '__main__':
    main()

