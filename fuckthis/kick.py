import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
from tqdm import tqdm
import json
import csv
import re
import time

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct"
FRAMES_DIR = "round_004"
OUTPUT_CSV = "static_adherence_analysis.csv"

# Sequence and frame processing settings
SEQUENCE_LENGTH = 8  # Increased slightly to better capture a 3-move sequence
FRAME_STEP = 4

# --- PROMPT WITH STATIC/HARDCODED INTENDED ACTIONS ---
PROMPT_TEMPLATE = """
You are a Tekken 3 gameplay analyst specializing in control adherence verification. You will analyze frame sequences to determine if a specific, predefined sequence of actions is being executed.

**TEKKEN 3 MOVE DEFINITIONS:**
- **Jab/Left Punch**: Quick straight punch with the left hand.
- **Right Punch**: Stronger punch with the right hand.
- **Left Kick**: Quick kick with the left leg.
- **Right Kick**: Stronger kick with the right leg.
- **Forward**: Character moves toward the opponent.
- **Backward**: Character moves away from the opponent.
- **Crouch**: Character ducks down.

**SEQUENCE TO VERIFY:**
You must check if the player performs the following three actions in this specific order within the provided frames:
1.  **Action 1: Left Punch (Jab)**
2.  **Action 2: Right Punch**
3.  **Action 3: Forward Movement**

**CONTROL ADHERENCE ANALYSIS:**
For each of the three actions in the "SEQUENCE TO VERIFY," determine if it was:
- **ADHERED**: The intended action was clearly and correctly executed in the correct order.
- **PARTIALLY_ADHERED**: The action was attempted but was incomplete, interrupted, or blended with another move.
- **NOT_ADHERED**: The intended action was not executed at all, or a completely different action occurred.

**Response Format:**
Return ONLY a valid JSON object. Use concise descriptions. The "actions" array must contain exactly three items, one for each action in the sequence above.

{
  "frame_analysis": {
    "description": "Briefly describe the overall action in the frames."
  },
  "actions": [
    {
      "intended": "Left Punch (Jab)",
      "status": "ADHERED/PARTIALLY_ADHERED/NOT_ADHERED",
      "observed": "Describe what was actually seen for this action."
    },
    {
      "intended": "Right Punch",
      "status": "ADHERED/PARTIALLY_ADHERED/NOT_ADHERED",
      "observed": "Describe what was actually seen for this action."
    },
    {
      "intended": "Forward Movement",
      "status": "ADHERED/PARTIALLY_ADHERED/NOT_ADHERED",
      "observed": "Describe what was actually seen for this action."
    }
  ]
}

Now analyze the provided frame sequence.
"""
# The intended sequence is now fixed
INTENDED_SEQUENCE_STRING = "Left Punch (Jab), Right Punch, Forward Movement"
TOTAL_INTENDED_INPUTS = 3

def get_frame_number(frame_filename):
    """Extract frame number from filename."""
    match = re.search(r'(\d+)', frame_filename)
    return int(match.group(1)) if match else 0

def write_csv_header(csv_path):
    """Initialize the CSV file with headers for static adherence analysis."""
    fieldnames = [
        'sequence_id', 
        'start_frame', 
        'end_frame', 
        'intended_sequence', 
        'adherence_percentage',
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

def create_sequences(frame_list, sequence_length, step):
    """Create sequences of frames for analysis."""
    sequences = []
    sorted_frames = sorted(frame_list, key=get_frame_number)
    for i in range(0, len(sorted_frames), step):
        sequence = sorted_frames[i:i + sequence_length]
        if len(sequence) == sequence_length:
            sequences.append(sequence)
    return sequences

def extract_json_from_response(response_text):
    """Extract and clean JSON from model response."""
    try:
        cleaned = response_text.strip().replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in response")
        return json.loads(json_match.group(0))
    except Exception as e:
        print(f"JSON extraction error: {e}")
        # Return a default error structure that matches what analyze_adherence expects
        return {
            "frame_analysis": {"description": f"Error parsing JSON: {e}"},
            "actions": [
                {"intended": "Action 1", "status": "NOT_ADHERED", "observed": "Parse error"},
                {"intended": "Action 2", "status": "NOT_ADHERED", "observed": "Parse error"},
                {"intended": "Action 3", "status": "NOT_ADHERED", "observed": "Parse error"}
            ]
        }

def analyze_adherence(analysis_result):
    """Parse the model's analysis and extract adherence statistics. Core logic is preserved."""
    try:
        if isinstance(analysis_result, str):
            analysis = extract_json_from_response(analysis_result)
        else:
            analysis = analysis_result
        
        actions = analysis.get('actions', [])
        adherence_counts = {'ADHERED': 0, 'PARTIALLY_ADHERED': 0, 'NOT_ADHERED': 0}
        
        for action in actions:
            status = action.get('status', 'NOT_ADHERED')
            if status in adherence_counts:
                adherence_counts[status] += 1
        
        performed_inputs = adherence_counts['ADHERED'] + adherence_counts['PARTIALLY_ADHERED']
        adherence_percentage = (performed_inputs / TOTAL_INTENDED_INPUTS) * 100 if TOTAL_INTENDED_INPUTS > 0 else 0
        
        return {
            'adherence_percentage': adherence_percentage,
            'counts': adherence_counts,
            'detailed_analysis': analysis,
            'visual_description': analysis.get('frame_analysis', {}).get('description', 'N/A')
        }
    except Exception as e:
        print(f"Error parsing adherence analysis: {e}")
        return {
            'adherence_percentage': 0,
            'counts': {'ADHERED': 0, 'PARTIALLY_ADHERED': 0, 'NOT_ADHERED': 3},
            'detailed_analysis': {'error': str(e)},
            'visual_description': f'Error: {e}'
        }

def main():
    """Main function to run static control adherence analysis."""
    print("🚀 Initializing Tekken 3 Static Adherence Analysis...")
    
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
    
    try:
        frames = sorted(os.listdir(FRAMES_DIR))
        print(f"🎬 Found {len(frames)} frames in '{FRAMES_DIR}'")
    except FileNotFoundError:
        print(f"❌ Error: Directory '{FRAMES_DIR}' not found.")
        return
    
    sequences = create_sequences(frames, SEQUENCE_LENGTH, FRAME_STEP)
    print(f"📋 Created {len(sequences)} sequences. Each will be checked for the sequence: '{INTENDED_SEQUENCE_STRING}'")
    
    write_csv_header(OUTPUT_CSV)
    print(f"📄 Initialized results file: {OUTPUT_CSV}")
    
    start_time = time.time()
    
    for seq_idx, frame_sequence in enumerate(tqdm(sequences, desc="Analyzing Sequences")):
        try:
            images = [Image.open(os.path.join(FRAMES_DIR, fname)).convert('RGB') for fname in frame_sequence]
            
            # The prompt is now static and doesn't need dynamic action inputs
            content = [{"type": "text", "text": PROMPT_TEMPLATE}]
            for img in images:
                content.append({"type": "image", "image": img})

            messages = [{"role": "user", "content": content}]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=images, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            adherence_analysis = analyze_adherence(response)
            
            # Prepare data for CSV
            result = {
                'sequence_id': f"seq_{seq_idx:03d}",
                'start_frame': get_frame_number(frame_sequence[0]),
                'end_frame': get_frame_number(frame_sequence[-1]),
                'intended_sequence': INTENDED_SEQUENCE_STRING,
                'adherence_percentage': f"{adherence_analysis['adherence_percentage']:.1f}%",
                'adhered_count': adherence_analysis['counts']['ADHERED'],
                'partially_adhered_count': adherence_analysis['counts']['PARTIALLY_ADHERED'],
                'not_adhered_count': adherence_analysis['counts']['NOT_ADHERED'],
                'visual_description': adherence_analysis['visual_description'],
                'detailed_analysis': json.dumps(adherence_analysis['detailed_analysis']),
                'frames_analyzed': ', '.join(frame_sequence)
            }
            
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=result.keys())
                writer.writerow(result)

        except Exception as e:
            print(f"❌ Error processing sequence {seq_idx}: {e}")
        finally:
            if 'images' in locals():
                for img in images: img.close()
            del images, inputs, generated_ids
            torch.cuda.empty_cache()
            
    elapsed_time = time.time() - start_time
    print("\n" + "="*40)
    print("🎯 STATIC ADHERENCE ANALYSIS SUMMARY 🎯")
    print("="*40)
    print(f"✅ Processed: {len(sequences)} sequences")
    print(f"⏱️ Total processing time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
    print(f"💾 Results saved to: {OUTPUT_CSV}")
    print("🎮 Analysis complete!")

if __name__ == '__main__':
    main()
