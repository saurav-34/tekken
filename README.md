# Tekken 3 Gameplay Analysis System

A sophisticated computer vision system for analyzing Tekken 3 gameplay using large vision-language models to verify control adherence and move execution accuracy.

## Overview

This project uses the **Qwen2-VL-72B-Instruct** model to analyze frame sequences from Tekken 3 gameplay and determine if intended controller inputs are accurately reflected in the actual gameplay execution. The system processes fighting game frames to assess control adherence and move execution patterns.

## Features

- **Frame Sequence Analysis**: Processes 6-frame sequences to capture complete move animations
- **Control Adherence Verification**: Compares intended inputs vs actual gameplay execution
- **Move Recognition**: Identifies Tekken 3 moves and combos from visual analysis
- **GPU Acceleration**: Optimized for multi-GPU setups (tested on 8x NVIDIA H200)
- **Batch Processing**: Efficiently processes large numbers of frame sequences

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 24GB+ VRAM (72B model requires significant memory)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: ~200GB free space (137GB for model weights + frame data)

### Software
- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.10
- **CUDA**: 12.1+
- **Conda**: For environment management

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd tekken
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate benchmark-env
```

### 3. Alternative: Install with pip
```bash
pip install -r requirements.txt
```

### 4. Verify GPU Setup
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Model Setup

The system uses **Qwen2-VL-72B-Instruct** (~137GB). On first run, the model will be automatically downloaded and cached:

```bash
python analyse1.py
```

**Note**: Model download requires stable internet and ~30-60 minutes depending on connection speed.

## Project Structure

```
tekken/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment specification
├── analyse1.py                         # Main analysis script
├── analyse.py                          # Alternative analysis implementation
├── benchmark_local.py                  # Local benchmarking script
├── frame_action_map.json              # Frame-to-action mappings
├── p1_inputs.csv                      # Player 1 input data
├── round_004/                         # Frame images (199 files)
│   ├── 0000.jpg
│   ├── 0001.jpg
│   └── ...
└── results/
    ├── action_sequence_analysis.csv    # Analysis results
    ├── benchmark_results_local.csv     # Benchmark data
    └── control_adherence_analysis.csv  # Control adherence metrics
```

## Usage

### Basic Analysis
```bash
conda activate benchmark-env
cd tekken
python analyse1.py
```

### Configuration Options

Edit `analyse1.py` to modify:
- **Model**: Change `MODEL_ID` (default: "Qwen/Qwen2-VL-72B-Instruct")
- **Sequence Length**: Modify `SEQUENCE_LENGTH` (default: 6 frames)
- **Frame Step**: Adjust `FRAME_STEP` (default: 3 frames)
- **Output Format**: Change output CSV files

### Input Data Format

**Frame Images**: Sequential JPG files (0000.jpg, 0001.jpg, etc.)
**Action Mapping**: JSON file mapping frame numbers to controller inputs:
```json
{
  "0": ["Right", "Right Kick"],
  "1": ["Right"],
  "2": ["Right"]
}
```

## Analysis Pipeline

1. **Frame Loading**: Loads sequential frame images from `round_004/`
2. **Sequence Creation**: Groups frames into 6-frame sequences with 3-frame steps
3. **Vision-Language Analysis**: Processes sequences through Qwen2-VL-72B
4. **Move Recognition**: Identifies executed moves and techniques
5. **Control Verification**: Compares intended vs actual actions
6. **Results Export**: Generates CSV reports with adherence metrics

## Output Files

- **`control_adherence_analysis.csv`**: Frame-by-frame adherence analysis
- **`action_sequence_analysis.csv`**: Detailed sequence breakdowns
- **`benchmark_results_local.csv`**: Performance benchmarks

## Performance

**Tested Configuration**:
- 8x NVIDIA H200 GPUs
- Model: Qwen2-VL-72B-Instruct (bfloat16)
- Processing: ~199 frames in sequences of 6

**Expected Performance**:
- Model loading: ~2-3 minutes (first time longer due to download)
- Frame processing: ~30-60 seconds per sequence (depending on GPU)

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size or sequence length
- Use smaller model variant if available
- Ensure no other GPU processes running

**Model Download Fails**:
- Check internet connection
- Verify Hugging Face Hub access
- Clear cache: `rm -rf ~/.cache/huggingface/`

**Flash Attention Installation**:
- Flash-attn requires CUDA compiler tools
- Optional dependency - model works without it
- Install with: `pip install flash-attn --no-build-isolation`

### GPU Requirements

| Model Size | Min VRAM | Recommended |
|------------|----------|-------------|
| 72B        | 48GB     | 64GB+       |
| Multi-GPU  | 24GB/GPU | 32GB+/GPU   |

## Development

### Adding New Analysis Features
1. Modify prompt templates in `analyse1.py`
2. Update frame processing logic
3. Extend output CSV columns
4. Test with sample data

### Custom Models
- Replace `MODEL_ID` with your model
- Ensure model supports vision-language tasks
- Update prompt format as needed

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push branch: `git push origin feature-name`
5. Submit pull request

## License

[Add your license information here]

## Citation

If you use this work in research, please cite:
```
[Add citation format]
```

## Acknowledgments

- **Qwen2-VL Team**: For the excellent vision-language model
- **Tekken 3**: Classic fighting game for analysis
- **Hugging Face**: Model hosting and transformers library