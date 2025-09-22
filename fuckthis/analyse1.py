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
FRAMES_DIR = "round_004"  # Directory containing frame images
ACTION_INPUTS_FILE = "frame_action_map.json"  # JSON file with frame-by-frame action inputs
OUTPUT_CSV = "control_adherence_analysis.csv"  # Output file for adherence analysis

# Sequence and frame processing settings for fighting game analysis
SEQUENCE_LENGTH = 6  # Frames to analyze together (captures move initiation to completion)
FRAME_STEP = 3       # Step between sequences (some overlap for continuity)
FRAME_INTERVAL = 1   # Consecutive frames for smooth temporal analysis

EXPECTED_FPS = 30
TOTAL_EXPECTED_FRAMES = 30 * EXPECTED_FPS

# --- TEKKEN 3 MOVE DEFINITIONS AND CONTROL ADHERENCE PROMPT ---
PROMPT_TEMPLATE = """
You are a Tekken 3 gameplay analyst specializing in control adherence verification. You will analyze frame sequences to determine if the intended actions (from controller inputs) are actually being executed in the gameplay.

**TEKKEN 3 MOVE DEFINITIONS:**

**PUNCHES:**
- **Jab/Left Punch**: Quick straight punch with left hand, minimal wind-up, fast execution
- **Right Punch**: Stronger punch with right hand, more wind-up than jab, heavier impact
- **Both Punch**: Character throws both hands forward simultaneously or in rapid succession

**KICKS:**
- **Left Kick**: Quick kick with left leg, usually low to mid-level
- **Right Kick**: Stronger kick with right leg, can be mid to high level
- **Both Kick**: Character uses both legs in succession or jumping/spinning kick motions

**MOVEMENT:**
- **Forward**: Character moves toward opponent, advancing stance
- **Backward**: Character moves away from opponent, retreating
- **Up**: Character jumps or ducks depending on context
- **Down**: Character crouches, ducks, or performs ground moves

**COMBINATIONS:**
- **Punch + Kick combinations**: Mixed striking using hands and feet
- **Movement + Attack**: Moving while attacking (advancing punch, retreating kick, etc.)

**MOVE EXECUTION CRITERIA:**
1. **Punch Recognition**: A punch is only confirmed when you see:
   - Clear arm extension/retraction motion
   - Contact with opponent (if intended to hit)
   - Character's body positioning consistent with punching motion
   - Takes 2-4 frames from initiation to completion

2. **Kick Recognition**: A kick is only confirmed when you see:
   - Clear leg lifting/extending motion
   - Hip rotation or stance change typical of kicking
   - Contact with opponent (if intended to hit)
   - Takes 3-6 frames from initiation to completion

3. **Movement Recognition**: Movement is confirmed by:
   - Clear positional change of character
   - Appropriate stance/posture for the movement
   - Consistent directional motion across frames

**CONTROL ADHERENCE ANALYSIS:**

Given the intended action inputs and the visual frame sequence, determine:

1. **Action Matching**: Does what you see in the frames match the intended input?

**Your Task:**
Analyze the provided frame sequence and compare it with the intended action inputs. For each action, determine if it was:
- **ADHERED**: The intended action was properly executed as planned
- **PARTIALLY_ADHERED**: The action was attempted but not fully executed (interrupted, blocked, or incomplete)
- **NOT_ADHERED**: The intended action was not executed at all, or a completely different action occurred

**Response Format:**
Provide your analysis in VALID JSON format only. Do not include any text before or after the JSON. The JSON must be properly formatted with correct syntax:

{
  "frame_analysis": {
    "sequence_start_frame": "frame_number",
    "sequence_end_frame": "frame_number", 
    "visual_description": "What actually happens in the frames"
  },
  "intended_actions": [
    {
      "frame": "frame_number",
      "intended_action": "action_from_input",
      "adherence_status": "ADHERED/PARTIALLY_ADHERED/NOT_ADHERED",
      "observed_action": "what_actually_happened",
      "confidence": "high/medium/low",
      "notes": "additional_observations"
    }
  ],
  "overall_adherence": {
    "percentage": "XX%",
    "summary": "Brief summary of control adherence for this sequence"
  }
}

**Example Analysis:**

Input Action: "Frame 120: Right Punch"
Visual Observation: Character extends right arm, makes contact with opponent

Response:
{
  "frame_analysis": {
    "sequence_start_frame": "118",
    "sequence_end_frame": "123",
    "visual_description": "Character winds up and executes right punch, making clean contact with opponent"
  },
  "intended_actions": [
    {
      "frame": "120",
      "intended_action": "Right Punch",
      "adherence_status": "ADHERED",
      "observed_action": "Right Punch executed successfully",
      "confidence": "high",
      "notes": "Clean execution with proper timing and contact"
    }
  ],
  "overall_adherence": {
    "percentage": "100%",
    "summary": "Perfect execution of intended right punch"
  }
}

Now analyze the provided frame sequence with the corresponding action inputs.
"""

def load_action_inputs(json_file_path):
    """Load action inputs from JSON file."""
    try:
        with open(json_file_path, 'r') as f:
            action_data = json.load(f)
        print(f"✅ Loaded action inputs from {json_file_path}")
        return action_data
    except FileNotFoundError:
        print(f"❌ Error: Action inputs file '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in '{json_file_path}': {e}")
        return None

def get_frame_number(frame_filename):
    """Extract frame number from filename (e.g., 'frame_120.jpg' -> 120)."""
    match = re.search(r'(\d+)', frame_filename)
    return int(match.group(1)) if match else 0

def get_actions_for_sequence(action_data, frame_sequence):
    """Get the intended actions for a given frame sequence."""
    sequence_actions = []
    
    for frame_file in frame_sequence:
        frame_num = get_frame_number(frame_file)
        frame_key = str(frame_num)
        
        if frame_key in action_data:
            sequence_actions.append({
                'frame': frame_num,
                'frame_file': frame_file,
                'actions': action_data[frame_key]
            })
        else:
            # No action recorded for this frame (idle/neutral)
            sequence_actions.append({
                'frame': frame_num,
                'frame_file': frame_file,
                'actions': ['idle']
            })
    
    return sequence_actions

def write_csv_header(csv_path):
    """Initialize the CSV file with headers for control adherence analysis."""
    fieldnames = [
        'sequence_id', 
        'start_frame', 
        'end_frame', 
        'intended_actions', 
        'adherence_percentage',
        'total_intended_inputs',
        'performed_inputs_count',
        'adhered_count',
        'partially_adhered_count', 
        'not_adhered_count',
        'visual_description',
        'detailed_analysis',
        'frames_analyzed'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def create_sequences(frame_list, sequence_length, step, frame_interval=1):
    """Create sequences of frames for temporal analysis."""
    sequences = []
    
    # Sort frames numerically
    sorted_frames = sorted(frame_list, key=get_frame_number)
    
    for i in range(0, len(sorted_frames), step):
        sequence = []
        for j in range(sequence_length):
            frame_idx = i + (j * frame_interval)
            if frame_idx < len(sorted_frames):
                sequence.append(sorted_frames[frame_idx])
            else:
                break
        
        if len(sequence) == sequence_length:
            sequences.append(sequence)
        else:
            break
    
    return sequences

def count_intended_inputs(sequence_actions):
    """Count the total number of intended inputs (excluding idle actions)."""
    total_inputs = 0
    for action_data in sequence_actions:
        actions = action_data['actions']
        if isinstance(actions, list):
            # Count non-idle actions
            non_idle_actions = [action for action in actions if action.lower() != 'idle']
            total_inputs += len(non_idle_actions)
        elif actions.lower() != 'idle':
            total_inputs += 1
    return total_inputs

def extract_json_from_response(response_text):
    """Extract and clean JSON from model response with robust error handling."""
    try:
        # Clean up the response
        cleaned = response_text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "")
        cleaned = cleaned.strip()
        
        # Find JSON boundaries
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        
        if not json_match:
            raise ValueError("No JSON object found in response")
            
        json_content = json_match.group(0)
        
        # Clean up common JSON issues
        json_content = re.sub(r',\s*}', '}', json_content)  # Remove trailing commas before }
        json_content = re.sub(r',\s*]', ']', json_content)  # Remove trailing commas before ]
        json_content = re.sub(r'\n\s*', ' ', json_content)  # Replace newlines with spaces
        json_content = re.sub(r'\s+', ' ', json_content)    # Normalize whitespace
        
        # Try to parse
        parsed = json.loads(json_content)
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Problematic JSON content: {json_content[:500] if 'json_content' in locals() else 'N/A'}")
        raise
    except Exception as e:
        print(f"JSON extraction error: {e}")
        raise

def analyze_adherence(analysis_result, sequence_actions=None):
    """Parse the model's analysis and extract adherence statistics."""
    try:
        # Parse the JSON response
        if isinstance(analysis_result, str):
            analysis = extract_json_from_response(analysis_result)
        else:
            analysis = analysis_result
        
        intended_actions = analysis.get('intended_actions', [])
        
        # Count adherence types
        adherence_counts = {
            'ADHERED': 0,
            'PARTIALLY_ADHERED': 0,
            'NOT_ADHERED': 0
        }
        
        for action in intended_actions:
            status = action.get('adherence_status', 'NOT_ADHERED')
            if status in adherence_counts:
                adherence_counts[status] += 1
        
        total_actions = len(intended_actions)
        total_intended_inputs = count_intended_inputs(sequence_actions) if sequence_actions else total_actions
        
        if total_intended_inputs > 0:
            # Calculate adherence percentage based on performed inputs out of total intended inputs
            performed_inputs = adherence_counts['ADHERED'] + adherence_counts['PARTIALLY_ADHERED']
            adherence_percentage = (performed_inputs / total_intended_inputs) * 100
        else:
            adherence_percentage = 0
        
        return {
            'adherence_percentage': adherence_percentage,
            'counts': adherence_counts,
            'total_actions': total_actions,
            'total_intended_inputs': total_intended_inputs,
            'performed_inputs_count': adherence_counts['ADHERED'] + adherence_counts['PARTIALLY_ADHERED'],
            'detailed_analysis': analysis,
            'visual_description': analysis.get('frame_analysis', {}).get('visual_description', 'N/A')
        }
        
    except Exception as e:
        print(f"Error parsing adherence analysis: {e}")
        print(f"Raw analysis result (first 1000 chars): {str(analysis_result)[:1000]}")
        return {
            'adherence_percentage': 0,
            'counts': {'ADHERED': 0, 'PARTIALLY_ADHERED': 0, 'NOT_ADHERED': 0},
            'total_actions': 0,
            'total_intended_inputs': 0,
            'performed_inputs_count': 0,
            'detailed_analysis': {'error': str(e), 'raw_response': str(analysis_result)[:500]},
            'visual_description': f'Error: {e}'
        }

def format_actions_for_prompt(sequence_actions):
    """Format the action sequence data for the model prompt."""
    actions_text = "\n**INTENDED ACTIONS FOR THIS SEQUENCE:**\n"
    
    for action_data in sequence_actions:
        frame_num = action_data['frame']
        actions = action_data['actions']
        actions_text += f"Frame {frame_num}: {', '.join(actions) if isinstance(actions, list) else actions}\n"
    
    return actions_text

def main():
    """Main function to run control adherence analysis."""
    print("🚀 Initializing Tekken 3 Control Adherence Analysis...")
    
    # Load action inputs
    action_data = load_action_inputs(ACTION_INPUTS_FILE)
    if action_data is None:
        return
    
    print(f"📊 Action inputs loaded: {len(action_data)} frames with recorded actions")
    
    # Initialize model
    print("🤖 Loading VLM model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
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
        print(f"Flash attention not available, using default: {e}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).eval()
    
    print("✅ Model loaded successfully")
    
    # Load frames
    try:
        frames = sorted(os.listdir(FRAMES_DIR))
        print(f"🎬 Found {len(frames)} frames in '{FRAMES_DIR}'")
    except FileNotFoundError:
        print(f"❌ Error: Directory '{FRAMES_DIR}' not found.")
        return
    
    # Create sequences
    sequences = create_sequences(frames, SEQUENCE_LENGTH, FRAME_STEP, FRAME_INTERVAL)
    print(f"📋 Created {len(sequences)} sequences of {SEQUENCE_LENGTH} frames each")
    
    # Initialize CSV
    write_csv_header(OUTPUT_CSV)
    print(f"📄 Initialized results file: {OUTPUT_CSV}")
    
    # Processing variables
    processed_count = 0
    error_count = 0
    total_adherence_sum = 0
    
    start_time = time.time()
    
    for seq_idx, frame_sequence in enumerate(sequences):
        try:
            print(f"\n🔍 Processing sequence {seq_idx + 1}/{len(sequences)}")
            print(f"   Frames: {frame_sequence[0]} → {frame_sequence[-1]}")
            
            # Get intended actions for this sequence
            sequence_actions = get_actions_for_sequence(action_data, frame_sequence)
            
            # Load images
            images = []
            for fname in frame_sequence:
                path = os.path.join(FRAMES_DIR, fname)
                img = Image.open(path).convert('RGB')
                images.append(img)
            
            # Prepare prompt with intended actions
            actions_prompt = format_actions_for_prompt(sequence_actions)
            
            # Create message content
            content = []
            content.append({"type": "text", "text": "TEKKEN 3 CONTROL ADHERENCE ANALYSIS"})
            content.append({"type": "text", "text": f"Analyzing {len(images)} consecutive frames for control adherence:"})
            
            # Add frame images with labels
            for i, img in enumerate(images, 1):
                frame_num = get_frame_number(frame_sequence[i-1])
                content.append({"type": "text", "text": f"Frame {i}/{len(images)} (Frame #{frame_num}):"})
                content.append({"type": "image", "image": img})
            
            # Add intended actions and main prompt
            content.append({"type": "text", "text": actions_prompt})
            content.append({"type": "text", "text": PROMPT_TEMPLATE})
            
            messages = [{"role": "user", "content": content}]
            
            # Generate analysis
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=images, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2000,  # Increased to allow complete JSON responses
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    temperature=0.1,
                    repetition_penalty=1.1
                )
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up GPU memory
            del inputs, generated_ids
            torch.cuda.empty_cache()
            
            # Analyze adherence
            adherence_analysis = analyze_adherence(response, sequence_actions)
            
            # Prepare CSV row
            start_frame = get_frame_number(frame_sequence[0])
            end_frame = get_frame_number(frame_sequence[-1])
            
            # Compile intended actions summary
            intended_actions_summary = []
            for action_data in sequence_actions:
                actions = action_data['actions']
                if isinstance(actions, list):
                    intended_actions_summary.extend(actions)
                else:
                    intended_actions_summary.append(actions)
            
            result = {
                'sequence_id': f"seq_{seq_idx:03d}",
                'start_frame': start_frame,
                'end_frame': end_frame,
                'intended_actions': ', '.join(intended_actions_summary),
                'adherence_percentage': f"{adherence_analysis['adherence_percentage']:.1f}%",
                'total_intended_inputs': adherence_analysis['total_intended_inputs'],
                'performed_inputs_count': adherence_analysis['performed_inputs_count'],
                'adhered_count': adherence_analysis['counts']['ADHERED'],
                'partially_adhered_count': adherence_analysis['counts']['PARTIALLY_ADHERED'],
                'not_adhered_count': adherence_analysis['counts']['NOT_ADHERED'],
                'visual_description': adherence_analysis['visual_description'],
                'detailed_analysis': json.dumps(adherence_analysis['detailed_analysis']),
                'frames_analyzed': ', '.join(frame_sequence)
            }
            
            # Write to CSV
            fieldnames = list(result.keys())
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)
            
            processed_count += 1
            total_adherence_sum += adherence_analysis['adherence_percentage']
            
            print(f"   ✅ Adherence: {adherence_analysis['adherence_percentage']:.1f}%")
            print(f"   📊 Actions: {adherence_analysis['counts']}")
            
        except Exception as e:
            error_count += 1
            print(f"   ❌ Error processing sequence {seq_idx}: {e}")
            
        finally:
            # Cleanup
            if 'images' in locals():
                for img in images:
                    img.close()
                del images
            torch.cuda.empty_cache()
    
    # Final summary
    elapsed_time = time.time() - start_time
    avg_adherence = total_adherence_sum / processed_count if processed_count > 0 else 0
    
    print(f"\n🎯 FINAL ANALYSIS SUMMARY:")
    print(f"✅ Successfully processed: {processed_count} sequences")
    print(f"❌ Errors encountered: {error_count} sequences")
    print(f"📊 Average control adherence: {avg_adherence:.1f}%")
    print(f"⏱️ Total processing time: {elapsed_time:.2f} seconds")
    print(f"💾 Results saved to: {OUTPUT_CSV}")
    print(f"🎮 Control adherence analysis complete!")

if __name__ == '__main__':
    main()