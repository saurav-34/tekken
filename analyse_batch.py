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
import glob
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from functools import partial

# --- CONFIGURATION ---
# MODEL OPTIONS - Choose based on your memory constraints:
# - Qwen2-VL-7B-Instruct: ~14GB memory, faster, good quality
# - Qwen2-VL-72B-Instruct: ~144GB memory, slower, best quality (HIGH RISK OF OOM)
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"  # Changed from 72B to 7B for safety
# MODEL_ID = "Qwen/Qwen2-VL-72B-Instruct"  # Uncomment for 72B (WARNING: May crash)

BASE_FRAMES_DIR = "dataf"  # Base directory containing step folders
OUTPUT_DIR = "batch_results"  # Directory for output CSV files

# Parallel processing configuration
MAX_WORKERS = 2  # Reduced from 4 to prevent memory issues with large models
GPU_MEMORY_THRESHOLD = 0.6  # Reduced from 0.8 to be more conservative with memory
PREFERRED_GPUS = [2, 3]  # Use GPUs 2 and 3 specifically

# Sequence and frame processing settings
SEQUENCE_LENGTH = 8  # Captures complete combos/moves (8 frames ≈ 0.27 seconds at 30fps)
FRAME_STEP = 4   # Overlapping sequences (0-7, 1-8, 2-9, etc.) for comprehensive coverage
FRAME_INTERVAL = 2  # Every 3rd frame for smooth temporal flow

EXPECTED_FPS = 30

# --- ENHANCED SYSTEM PROMPT FOR ACTION DETECTION ---
PROMPT_TEMPLATE = """
 
You are a methodical and cautious fighting game analyst. Your primary goal is to avoid misinterpretation by following a strict, step-by-step reasoning process.

(Always remember **PLAYER 1 IS IN WHITE OUTFIT, PLAYER 2 IS IN YELLOW**)

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

def robust_json_parse(response_text):
    """
    Robust JSON parsing that handles malformed responses from the model.
    
    Args:
        response_text: Raw response text from the model
        
    Returns:
        dict: Parsed JSON object or default structure if parsing fails
    """
    try:
        # Clean the response
        json_str = response_text.strip()
        
        # Remove common markdown formatting
        json_str = json_str.replace("```json", "").replace("```", "")
        
        # Try to find JSON object boundaries
        json_patterns = [
            # Look for complete JSON object with action_type
            r'\{[^{}]*"action_type"[^{}]*\}',
            # Look for any JSON object
            r'\{[^{}]*\}',
            # Look for nested JSON (in case of multiple braces)
            r'\{.*?"action_type".*?\}',
        ]
        
        analysis = None
        
        for pattern in json_patterns:
            matches = re.findall(pattern, json_str, re.DOTALL)
            for match in matches:
                try:
                    # Try to parse each potential JSON match
                    potential_json = json.loads(match)
                    if isinstance(potential_json, dict) and 'action_type' in potential_json:
                        analysis = potential_json
                        break
                except json.JSONDecodeError:
                    continue
            
            if analysis:
                break
        
        # If no valid JSON found, try line-by-line parsing
        if not analysis:
            lines = json_str.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('{') and 'action_type' in line:
                    try:
                        # Find the end of the JSON object
                        brace_count = 0
                        json_end = 0
                        for i, char in enumerate(line):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        
                        if json_end > 0:
                            potential_json = line[:json_end]
                            analysis = json.loads(potential_json)
                            if isinstance(analysis, dict) and 'action_type' in analysis:
                                break
                    except (json.JSONDecodeError, IndexError):
                        continue
        
        # Return parsed analysis or default structure
        if analysis and isinstance(analysis, dict) and 'action_type' in analysis:
            return analysis
        else:
            raise ValueError(f"No valid JSON with action_type found in response")
            
    except Exception as e:
        # Return a default error structure
        print(f"⚠️ JSON parsing error: {e}")
        return {
            'action_type': 'PARSE_ERROR',
            'player2_interrupts': 'N/A',
            'action_success': 'N/A',
            'description': f'JSON parsing failed: {str(e)[:100]}',
            'reasoning': 'Failed to parse model response'
        }


def process_round_worker(round_info, output_dir, gpu_id=None):
    """
    Worker function to process a single round in parallel.
    This function can be called by multiple processes.
    
    Args:
        round_info: Tuple of (step_name, round_name, round_path)
        output_dir: Output directory for CSV files
        gpu_id: GPU ID to use for this worker (None for auto)
    
    Returns:
        Dict with processing results
    """
    step_name, round_name, round_path = round_info
    worker_id = os.getpid()
    
    try:
        # Set GPU device if specified
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device_info = f"GPU {gpu_id}"
        else:
            device_info = "Auto"
        
        print(f"🔄 Worker {worker_id} ({device_info}): Starting {step_name}/{round_name}")
        
        # Initialize model and processor for this worker
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto" if gpu_id is None else f"cuda:{gpu_id}",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            ).eval()
        except Exception as e:
            print(f"Flash attention not available for worker {worker_id}, falling back: {e}")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto" if gpu_id is None else f"cuda:{gpu_id}",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).eval()
        
        # Create a mini processor for this round
        mini_processor = SingleRoundProcessor(model, processor, output_dir)
        success = mini_processor.process_round(step_name, round_name, round_path)
        
        # Cleanup
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        result = {
            'worker_id': worker_id,
            'step_name': step_name,
            'round_name': round_name,
            'success': success,
            'gpu_id': gpu_id,
            'sequences_processed': mini_processor.sequences_processed,
            'errors': mini_processor.errors
        }
        
        print(f"✅ Worker {worker_id}: Completed {step_name}/{round_name} (Success: {success})")
        return result
        
    except Exception as e:
        print(f"❌ Worker {worker_id}: Failed {step_name}/{round_name} - {e}")
        return {
            'worker_id': worker_id,
            'step_name': step_name,
            'round_name': round_name,
            'success': False,
            'error': str(e),
            'gpu_id': gpu_id,
            'sequences_processed': 0,
            'errors': 1
        }


class SingleRoundProcessor:
    """Lightweight processor for a single round (used by workers)."""
    
    def __init__(self, model, processor, output_dir):
        self.model = model
        self.processor = processor
        self.output_dir = output_dir
        self.sequences_processed = 0
        self.errors = 0
    
    def write_csv_header(self, csv_path):
        """Initialize a CSV file with headers if it doesn't exist."""
        fieldnames = ['sequence', 'action_type', 'player2_interrupts', 'action_success', 'description', 'frames']
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 
                writer.writeheader()
        return fieldnames
    
    def append_to_csv(self, csv_path, result, fieldnames):
        """Append a single result to CSV with proper error handling and flushing."""
        try:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)
                csvfile.flush()
                os.fsync(csvfile.fileno())
            return True
        except Exception as e:
            print(f"❌ Error writing to CSV: {e}")
            return False
    
    def create_sequences(self, frame_list, sequence_length, step, frame_interval=1):
        """Create sequences of frames for temporal analysis."""
        sequences = []
        
        def extract_frame_number(filename):
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
            
            if len(sequence) == sequence_length:
                sequences.append(sequence)
            else:
                break
        
        return sequences
    
    def get_output_filename(self, step_name, round_name):
        """Generate output CSV filename for a round."""
        return os.path.join(self.output_dir, f"{step_name}_{round_name}_analysis.csv")
    
    def process_round(self, step_name, round_name, round_path):
        """Process a single round and save results to CSV."""
        output_file = self.get_output_filename(step_name, round_name)
        
        try:
            # Get frame files
            frame_files = [f for f in os.listdir(round_path) if f.endswith('.jpg')]
            if not frame_files:
                print(f"❌ No frame files found in {round_path}")
                return False
            
            # Sort frames numerically
            def extract_frame_number(filename):
                match = re.search(r'(\d+)', filename)
                return int(match.group(1)) if match else 0
            
            frame_files = sorted(frame_files, key=extract_frame_number)
            
            # Use only first 50% of frames for analysis
            total_frames = len(frame_files)
            half_point = total_frames // 1
            sampled_frames = frame_files[:half_point]
            
            # Create sequences from sampled frames
            sequences = self.create_sequences(sampled_frames, SEQUENCE_LENGTH, FRAME_STEP, FRAME_INTERVAL)
            
            if not sequences:
                print(f"❌ No valid sequences created for {round_name}")
                return False
            
            # Initialize CSV with proper header handling
            fieldnames = self.write_csv_header(output_file)
            
            # Process sequences with incremental saving
            for seq_idx, frame_sequence in enumerate(sequences):
                try:
                    # Load images
                    images = []
                    for fname in frame_sequence:
                        path = os.path.join(round_path, fname)
                        img = Image.open(path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        images.append(img)
                    
                    # Prepare content for model
                    content = []
                    content.append({"type": "text", "text": f"TEMPORAL SEQUENCE ANALYSIS - {len(images)} consecutive frames:"})
                    
                    for i, img in enumerate(images, 1):
                        content.append({"type": "text", "text": f"Frame {i}/{len(images)} (chronological order):"})
                        content.append({"type": "image", "image": img})
                    
                    content.append({"type": "text", "text": PROMPT_TEMPLATE})
                    messages = [{"role": "user", "content": content}]

                    # Model inference
                    text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=[text_input], images=images, return_tensors="pt").to(self.model.device)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs, 
                            max_new_tokens=300,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            temperature=0.1,
                            repetition_penalty=1.1,
                        )
                        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # Cleanup
                        del inputs
                        del generated_ids
                        torch.cuda.empty_cache()
                    
                    # Parse response with robust JSON parsing (SingleRoundProcessor)
                    analysis = robust_json_parse(response)
                    
                    # Prepare result
                    sequence_name = f"seq_{seq_idx:03d}_{frame_sequence[0]}_to_{frame_sequence[-1]}"
                    result = {
                        'sequence': sequence_name,
                        'action_type': analysis.get('action_type', 'N/A'),
                        'player2_interrupts': analysis.get('player2_interrupts', 'N/A'),
                        'action_success': analysis.get('action_success', 'N/A'),
                        'description': analysis.get('description', 'N/A'),
                        'frames': ', '.join(frame_sequence)
                    }
                    
                    # Write to CSV immediately (live saving)
                    if self.append_to_csv(output_file, result, fieldnames):
                        self.sequences_processed += 1
                    else:
                        self.errors += 1
                    
                    # Cleanup images
                    for img in images:
                        if hasattr(img, 'close'):
                            img.close()
                    del images
                    
                except Exception as e:
                    self.errors += 1
                    error_msg = str(e)
                    
                    print(f"      ⚠️ Error in sequence {seq_idx}: {error_msg}")
                    
                    # Write error to CSV immediately with better error info
                    error_result = {
                        'sequence': f"seq_{seq_idx:03d}_ERROR",
                        'action_type': 'ERROR',
                        'player2_interrupts': 'N/A',
                        'action_success': 'N/A',
                        'description': f'Processing error: {error_msg[:200]}',
                        'frames': ', '.join(frame_sequence) if 'frame_sequence' in locals() else 'N/A'
                    }
                    
                    self.append_to_csv(output_file, error_result, fieldnames)
                    
                    # Cleanup on error
                    if 'images' in locals():
                        for img in images:
                            if hasattr(img, 'close'):
                                img.close()
                        del images
                    
                    # Clear GPU memory on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to process round {round_name}: {e}")
            return False


class BatchProcessor:
    def __init__(self, base_dir=BASE_FRAMES_DIR, output_dir=OUTPUT_DIR, max_workers=None):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.model = None
        self.processor = None
        
        # Parallel processing setup - detect GPUs first
        self.available_gpus = self.detect_available_gpus()
        self.max_workers = max_workers or self.calculate_optimal_workers()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Statistics tracking
        self.total_rounds_found = 0
        self.total_rounds_processed = 0
        self.total_sequences_processed = 0
        self.total_errors = 0
        self.start_time = None
        
        print(f"🔧 Parallel Processing Configuration:")
        print(f"   Max workers: {self.max_workers}")
        print(f"   Available GPUs: {len(self.available_gpus)}")
        print(f"   GPU IDs: {self.available_gpus}")
    
    def detect_available_gpus(self):
        """Detect available GPUs for parallel processing, using only GPUs 2 and 3."""
        if not torch.cuda.is_available():
            print("⚠️ No CUDA GPUs available, will use CPU")
            return []
        
        gpu_count = torch.cuda.device_count()
        print(f"🔍 Checking GPUs 2 and 3 specifically (Total GPUs available: {gpu_count})")
        
        selected_gpus = []
        
        for gpu_id in PREFERRED_GPUS:
            if gpu_id >= gpu_count:
                print(f"   ❌ GPU {gpu_id}: Not available (only {gpu_count} GPUs in system)")
                continue
                
            try:
                # Test GPU accessibility
                torch.cuda.set_device(gpu_id)
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                memory_used = max(memory_allocated, memory_reserved)
                memory_available = memory_total - memory_used
                usage_percent = (memory_used / memory_total) * 100
                
                print(f"   GPU {gpu_id}: {memory_available:.1f}GB available / {memory_total:.1f}GB total ({memory_used:.1f}GB used, {usage_percent:.1f}%)")
                
                # Check if GPU has sufficient memory (at least 10GB for 7B model, 80GB for 72B)
                min_memory = 80 if "72B" in MODEL_ID else 10
                if memory_available > min_memory:
                    selected_gpus.append(gpu_id)
                    print(f"   ✅ Selected GPU {gpu_id}: {memory_available:.1f}GB available")
                else:
                    print(f"   ❌ Skipped GPU {gpu_id}: {memory_available:.1f}GB available (insufficient for {MODEL_ID.split('/')[-1]})")
                    
            except Exception as e:
                print(f"   ❌ GPU {gpu_id}: Error accessing ({e})")
        
        if not selected_gpus:
            print("⚠️ No suitable GPUs found among GPUs 2 and 3. Checking all GPUs as fallback...")
            # Fallback to check all GPUs if preferred ones are not available
            return self.detect_all_available_gpus()
        else:
            print(f"🎯 Using {len(selected_gpus)} GPU(s): {selected_gpus}")
        
        return selected_gpus
    
    def detect_all_available_gpus(self):
        """Fallback method to detect any available GPUs."""
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        
        for i in range(gpu_count):
            try:
                torch.cuda.set_device(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_used = max(memory_allocated, memory_reserved)
                memory_available = memory_total - memory_used
                
                if memory_available > 20:
                    gpu_info.append({
                        'id': i,
                        'available': memory_available,
                        'usage_percent': (memory_used / memory_total) * 100
                    })
                    
            except Exception:
                continue
        
        # Sort by available memory and return top 2
        gpu_info.sort(key=lambda x: (-x['available'], x['usage_percent']))
        return [gpu['id'] for gpu in gpu_info[:2]]
    
    def calculate_optimal_workers(self):
        """Calculate optimal number of workers based on system resources."""
        # For 2 GPUs with 72B model, use conservative worker count
        if len(self.available_gpus) >= 2:
            # Use exactly 2 workers (1 per GPU) to prevent memory conflicts
            return 2
        elif len(self.available_gpus) == 1:
            # Single GPU mode
            return 1
        else:
            # CPU fallback
            return 1
        
    def initialize_model(self):
        """Initialize the model and processor with safety checks."""
        print("🚀 Initializing local model and processor...")
        
        # Safety check for model size
        if "72B" in MODEL_ID:
            print("⚠️ WARNING: Loading 72B model - this requires ~144GB VRAM!")
            print("   Consider using 7B model instead by changing MODEL_ID")
            response = input("Continue with 72B model? (y/N): ").lower()
            if response != 'y':
                print("❌ Aborted by user. Change MODEL_ID to Qwen/Qwen2-VL-7B-Instruct for safety.")
                return False
        
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        print(f"Loading model '{MODEL_ID}'. This may take several minutes and significant VRAM...")
        
        # Check available memory before loading
        if torch.cuda.is_available():
            for gpu_id in self.available_gpus:
                torch.cuda.set_device(gpu_id)
                available = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                print(f"GPU {gpu_id}: {available:.1f}GB total memory")
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            ).eval()
        except Exception as e:
            print(f"Flash attention not available, falling back to default: {e}")
            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ).eval()
            except Exception as load_error:
                print(f"❌ Failed to load model: {load_error}")
                print("💡 Try using a smaller model like Qwen2-VL-7B-Instruct")
                return False
        
        print("✅ Model loaded successfully onto GPU.")
        self.check_gpu_memory()
        return True

    def check_memory_usage(self):
        """Check system memory usage and warn if it's getting high."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            gpu_memory_used = 0
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory_used += torch.cuda.memory_allocated(i) / 1024**3
            
            if memory.percent > 85:
                print(f"⚠️ WARNING: System memory usage is high: {memory.percent:.1f}%")
            
            if gpu_memory_used > 50:  # More than 50GB across all GPUs
                print(f"⚠️ WARNING: GPU memory usage is high: {gpu_memory_used:.1f}GB")
                
        except ImportError:
            pass  # psutil not available

    def check_gpu_memory(self):
        """Check available GPU memory."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {memory_total:.2f}GB total")

    def discover_rounds(self, specific_steps=None, specific_rounds=None):
        """
        Discover all round directories in the dataf structure.
        
        Args:
            specific_steps: List of step names to process (e.g., ['step_1000', 'step_1500'])
            specific_rounds: List of round patterns to process (e.g., ['round_058_generated'])
        
        Returns:
            List of tuples (step_name, round_name, full_path)
        """
        rounds = []
        
        if not os.path.exists(self.base_dir):
            print(f"❌ Base directory '{self.base_dir}' not found!")
            return rounds
        
        # Get all step directories
        step_dirs = [d for d in os.listdir(self.base_dir) 
                    if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith('step_')]
        
        # Filter by specific steps if provided
        if specific_steps:
            step_dirs = [d for d in step_dirs if d in specific_steps]
        
        step_dirs.sort()
        
        for step_dir in step_dirs:
            step_path = os.path.join(self.base_dir, step_dir)
            
            # Get all round directories in this step
            round_dirs = [d for d in os.listdir(step_path) 
                         if os.path.isdir(os.path.join(step_path, d)) and d.startswith('round_')]
            
            # Filter by specific rounds if provided
            if specific_rounds:
                round_dirs = [d for d in round_dirs if d in specific_rounds]
            
            round_dirs.sort()
            
            for round_dir in round_dirs:
                round_path = os.path.join(step_path, round_dir)
                rounds.append((step_dir, round_dir, round_path))
        
        self.total_rounds_found = len(rounds)
        print(f"📁 Discovered {len(rounds)} rounds across {len(step_dirs)} steps")
        
        return rounds

    def create_sequences(self, frame_list, sequence_length, step, frame_interval=1):
        """Create sequences of frames for temporal analysis."""
        sequences = []
        
        def extract_frame_number(filename):
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
            
            if len(sequence) == sequence_length:
                sequences.append(sequence)
            else:
                break
        
        return sequences

    def write_csv_header(self, csv_path):
        """Initialize a CSV file with headers if it doesn't exist."""
        fieldnames = ['sequence', 'action_type', 'player2_interrupts', 'action_success', 'description', 'frames']
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 
                writer.writeheader()
        return fieldnames
    
    def append_to_csv(self, csv_path, result, fieldnames):
        """Append a single result to CSV with proper error handling and flushing."""
        try:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)
                csvfile.flush()  # Force write to disk immediately
                os.fsync(csvfile.fileno())  # Ensure data is written to storage
            return True
        except Exception as e:
            print(f"❌ Error writing to CSV: {e}")
            return False

    def get_output_filename(self, step_name, round_name):
        """Generate output CSV filename for a round."""
        return os.path.join(self.output_dir, f"{step_name}_{round_name}_analysis.csv")

    def should_skip_round(self, output_file, force_reprocess=False):
        """Check if round should be skipped (already processed)."""
        if force_reprocess:
            return False
        
        if os.path.exists(output_file):
            # Check if file has content (more than just header)
            try:
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Header + at least one data row
                        return True
            except:
                pass
        
        return False

    def process_round(self, step_name, round_name, round_path):
        """Process a single round and save results to CSV."""
        output_file = self.get_output_filename(step_name, round_name)
        
        print(f"\n🎬 Processing {step_name}/{round_name}")
        print(f"   📂 Source: {round_path}")
        print(f"   💾 Output: {output_file}")
        
        try:
            # Get frame files
            frame_files = [f for f in os.listdir(round_path) if f.endswith('.jpg')]
            if not frame_files:
                print(f"❌ No frame files found in {round_path}")
                return False
            
            # Sort frames numerically
            def extract_frame_number(filename):
                match = re.search(r'(\d+)', filename)
                return int(match.group(1)) if match else 0
            
            frame_files = sorted(frame_files, key=extract_frame_number)
            
            # Use only first 50% of frames for analysis
            total_frames = len(frame_files)
            half_point = total_frames // 2
            sampled_frames = frame_files[:half_point]  # Take first half of frames
            
            print(f"   📸 Found {total_frames} frames, using first {len(sampled_frames)} frames (first 50%)")
            
            # Create sequences from sampled frames
            sequences = self.create_sequences(sampled_frames, SEQUENCE_LENGTH, FRAME_STEP, FRAME_INTERVAL)
            
            # Limit sequences to prevent memory issues (conservative approach)
            max_sequences = 25 if "72B" in MODEL_ID else 50  # Fewer sequences for larger models
            if len(sequences) > max_sequences:
                original_count = len(sequences)
                sequences = sequences[:max_sequences]
                print(f"   🔗 Limited to {len(sequences)} sequences (from {original_count} total) to prevent memory issues")
            else:
                print(f"   🔗 Created {len(sequences)} sequences")
            
            if not sequences:
                print(f"❌ No valid sequences created for {round_name}")
                return False
            
            # Initialize CSV with proper header handling
            fieldnames = self.write_csv_header(output_file)
            
            # Process sequences with incremental saving
            round_processed_count = 0
            round_error_count = 0
            
            for seq_idx, frame_sequence in enumerate(sequences):
                print(f"      🔄 Processing sequence {seq_idx+1}/{len(sequences)} ({(seq_idx+1)/len(sequences)*100:.1f}%)")
                
                try:
                    # Load images
                    images = []
                    for fname in frame_sequence:
                        path = os.path.join(round_path, fname)
                        img = Image.open(path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        images.append(img)
                    
                    # Prepare content for model
                    content = []
                    content.append({"type": "text", "text": f"TEMPORAL SEQUENCE ANALYSIS - {len(images)} consecutive frames:"})
                    
                    for i, img in enumerate(images, 1):
                        content.append({"type": "text", "text": f"Frame {i}/{len(images)} (chronological order):"})
                        content.append({"type": "image", "image": img})
                    
                    content.append({"type": "text", "text": PROMPT_TEMPLATE})
                    messages = [{"role": "user", "content": content}]

                    # Model inference
                    text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=[text_input], images=images, return_tensors="pt").to(self.model.device)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs, 
                            max_new_tokens=300,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            temperature=0.1,
                            repetition_penalty=1.1,
                        )
                        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
                        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # Cleanup
                        del inputs
                        del generated_ids
                        torch.cuda.empty_cache()
                    
                    # Parse response with robust JSON parsing (BatchProcessor)
                    analysis = robust_json_parse(response)
                    
                    # Prepare result
                    sequence_name = f"seq_{seq_idx:03d}_{frame_sequence[0]}_to_{frame_sequence[-1]}"
                    result = {
                        'sequence': sequence_name,
                        'action_type': analysis.get('action_type', 'N/A'),
                        'player2_interrupts': analysis.get('player2_interrupts', 'N/A'),
                        'action_success': analysis.get('action_success', 'N/A'),
                        'description': analysis.get('description', 'N/A'),
                        'frames': ', '.join(frame_sequence)
                    }
                    
                    # Write to CSV immediately (live saving)
                    if self.append_to_csv(output_file, result, fieldnames):
                        round_processed_count += 1
                        self.total_sequences_processed += 1
                        print(f"      ✅ Sequence {seq_idx+1}/{len(sequences)}: {result['action_type']} (SAVED)")
                    else:
                        print(f"      ❌ Failed to save sequence {seq_idx+1}")
                        round_error_count += 1
                        self.total_errors += 1
                    
                    # Cleanup images
                    for img in images:
                        if hasattr(img, 'close'):
                            img.close()
                    del images
                    
                except Exception as e:
                    round_error_count += 1
                    self.total_errors += 1
                    error_msg = str(e)
                    
                    print(f"      ⚠️ Error in sequence {seq_idx}: {error_msg}")
                    
                    # Write error to CSV immediately with better error reporting
                    error_result = {
                        'sequence': f"seq_{seq_idx:03d}_ERROR",
                        'action_type': 'ERROR',
                        'player2_interrupts': 'N/A',
                        'action_success': 'N/A',
                        'description': f'Processing error: {error_msg[:200]}',
                        'frames': ', '.join(frame_sequence) if 'frame_sequence' in locals() else 'N/A'
                    }
                    
                    self.append_to_csv(output_file, error_result, fieldnames)
                    
                    # Cleanup on error
                    if 'images' in locals():
                        for img in images:
                            if hasattr(img, 'close'):
                                img.close()
                        del images
                    
                    # Clear GPU memory on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            print(f"   📊 Round completed: {round_processed_count} successful, {round_error_count} errors")
            self.total_rounds_processed += 1
            return True
            
        except Exception as e:
            print(f"❌ Failed to process round {round_name}: {e}")
            return False

    def run_batch_analysis(self, specific_steps=None, specific_rounds=None, force_reprocess=False, use_parallel=True):
        """Run batch analysis on all discovered rounds with optional parallel processing."""
        self.start_time = time.time()
        
        # Discover rounds
        rounds = self.discover_rounds(specific_steps, specific_rounds)
        
        if not rounds:
            print("❌ No rounds found to process!")
            return
        
        # Filter out already processed rounds if not forcing reprocess
        if not force_reprocess:
            rounds_to_process = []
            skipped_count = 0
            
            for step_name, round_name, round_path in rounds:
                output_file = self.get_output_filename(step_name, round_name)
                if self.should_skip_round(output_file, force_reprocess):
                    skipped_count += 1
                    print(f"⏭️ Skipping {step_name}/{round_name} (already processed)")
                else:
                    rounds_to_process.append((step_name, round_name, round_path))
            
            rounds = rounds_to_process
            print(f"📊 Rounds to process: {len(rounds)} (skipped: {skipped_count})")
        
        if not rounds:
            print("✅ All rounds already processed!")
            return
        
        print(f"\n🎯 Starting batch processing of {len(rounds)} rounds...")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔄 Parallel processing: {'Enabled' if use_parallel and len(self.available_gpus) > 0 else 'Disabled'}")
        
        if use_parallel and len(self.available_gpus) > 0:
            self.run_parallel_processing(rounds)
        else:
            self.run_sequential_processing(rounds)
        
        # Final summary
        elapsed_time = time.time() - self.start_time
        print(f"\n🎉 BATCH PROCESSING COMPLETE!")
        print(f"📊 Total rounds found: {self.total_rounds_found}")
        print(f"✅ Rounds processed: {self.total_rounds_processed}")
        print(f"🔗 Total sequences processed: {self.total_sequences_processed}")
        print(f"❌ Total errors: {self.total_errors}")
        print(f"⏱️ Total time: {str(datetime.timedelta(seconds=int(elapsed_time)))}")
        print(f"📁 Results saved in: {self.output_dir}")
    
    def run_parallel_processing(self, rounds):
        """Run processing using parallel workers with smart GPU allocation."""
        print(f"🚀 Starting parallel processing with {self.max_workers} workers on {len(self.available_gpus)} GPUs")
        
        if not self.available_gpus:
            print("⚠️ No suitable GPUs available, falling back to sequential processing")
            self.run_sequential_processing(rounds)
            return
        
        # Process rounds in parallel
        completed_count = 0
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs with round-robin GPU assignment
            future_to_round = {}
            for i, round_info in enumerate(rounds):
                # Assign GPU in round-robin fashion among available GPUs
                gpu_id = self.available_gpus[i % len(self.available_gpus)]
                future = executor.submit(process_round_worker, round_info, self.output_dir, gpu_id)
                future_to_round[future] = (round_info, gpu_id)
                print(f"📤 Submitted {round_info[0]}/{round_info[1]} to GPU {gpu_id}")
            
            # Process completed jobs as they finish
            for future in as_completed(future_to_round):
                round_info, assigned_gpu = future_to_round[future]
                step_name, round_name, _ = round_info
                
                try:
                    result = future.result()
                    completed_count += 1
                    
                    if result['success']:
                        self.total_rounds_processed += 1
                        self.total_sequences_processed += result.get('sequences_processed', 0)
                        self.total_errors += result.get('errors', 0)
                        
                        print(f"✅ [{completed_count}/{len(rounds)}] {step_name}/{round_name} completed by worker {result['worker_id']} on GPU {assigned_gpu}")
                    else:
                        self.total_errors += 1
                        error_msg = result.get('error', 'Unknown error')
                        print(f"❌ [{completed_count}/{len(rounds)}] {step_name}/{round_name} failed on GPU {assigned_gpu}: {error_msg}")
                
                except Exception as e:
                    completed_count += 1
                    self.total_errors += 1
                    print(f"❌ [{completed_count}/{len(rounds)}] {step_name}/{round_name} crashed on GPU {assigned_gpu}: {e}")
                
                # Show progress
                progress = completed_count / len(rounds) * 100
                elapsed = time.time() - self.start_time
                eta = (elapsed / completed_count * len(rounds)) - elapsed if completed_count > 0 else 0
                print(f"📊 Progress: {progress:.1f}% | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
    
    def run_sequential_processing(self, rounds):
        """Run processing sequentially (fallback method)."""
        print("🔄 Running sequential processing (single-threaded)")
        
        # Initialize model for sequential processing
        if not self.model:
            self.initialize_model()
        
        for i, (step_name, round_name, round_path) in enumerate(rounds, 1):
            print(f"\n{'='*80}")
            print(f"📊 PROGRESS: Round {i}/{len(rounds)} ({i/len(rounds)*100:.1f}%)")
            print(f"⏱️ Elapsed: {str(datetime.timedelta(seconds=int(time.time() - self.start_time)))}")
            
            # Process round
            success = self.process_round(step_name, round_name, round_path)
            
            if success:
                self.total_rounds_processed += 1
            
            # Memory cleanup every few rounds
            if i % 3 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                print(f"🧹 Memory cleanup performed")
                self.check_gpu_memory()
                self.check_memory_usage()  # Check system memory too

def main():
    parser = argparse.ArgumentParser(description='Batch process fighting game rounds for action analysis')
    parser.add_argument('--steps', nargs='+', help='Specific step folders to process (e.g., step_1000 step_1500)')
    parser.add_argument('--rounds', nargs='+', help='Specific round patterns to process (e.g., round_058_generated)')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of already processed rounds')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory for CSV files')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: 2 for 2 GPUs)')
    parser.add_argument('--sequential', action='store_true', help='Force sequential processing (disable parallel)')
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 3], help='Specific GPU IDs to use (default: 2 3)')
    parser.add_argument('--dry-run', action='store_true', help='Test configuration without processing (safety check)')
    
    args = parser.parse_args()
    
    print("🥊 TEKKEN BATCH ANALYSIS SYSTEM")
    print("=" * 50)
    print(f"📋 Model: {MODEL_ID}")
    print(f"💾 Memory estimate: {'~14GB' if '7B' in MODEL_ID else '~144GB'}")
    print()
    
    processor = BatchProcessor(output_dir=args.output_dir, max_workers=args.workers)
    
    # Dry run mode - just test configuration
    if args.dry_run:
        print("🧪 DRY RUN MODE - Testing configuration only")
        rounds = processor.discover_rounds(args.steps, args.rounds)
        print(f"📊 Would process {len(rounds)} rounds")
        print("✅ Configuration test complete - script should be safe to run")
        return
    
    # Override GPU selection if specified or use default [2, 3]
    if args.gpus != [2, 3]:  # Only override if user specified different GPUs
        print(f"🎯 Using user-specified GPUs: {args.gpus}")
    else:
        print(f"🎯 Using default GPUs: {args.gpus}")
        
    # Validate GPU IDs
    if torch.cuda.is_available():
        available_gpu_count = torch.cuda.device_count()
        valid_gpus = [gpu_id for gpu_id in args.gpus if 0 <= gpu_id < available_gpu_count]
        if len(valid_gpus) != len(args.gpus):
            invalid_gpus = [gpu_id for gpu_id in args.gpus if gpu_id not in valid_gpus]
            print(f"⚠️ Invalid GPU IDs: {invalid_gpus} (available: 0-{available_gpu_count-1})")
        valid_gpus = valid_gpus[:2]  # Limit to 2 GPUs
        processor.available_gpus = valid_gpus
        processor.max_workers = min(len(processor.available_gpus), args.workers or 2)
        print(f"🔧 Final configuration: GPUs {processor.available_gpus}, {processor.max_workers} workers")
    else:
        print("⚠️ No CUDA available, will use CPU mode")
    
    processor.run_batch_analysis(
        specific_steps=args.steps,
        specific_rounds=args.rounds,
        force_reprocess=args.force,
        use_parallel=not args.sequential
    )

if __name__ == '__main__':
    # Set multiprocessing start method for better compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Start method already set
    
    main()