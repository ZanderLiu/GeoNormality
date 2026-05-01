# GeoNormality: 3D Geometric Consistency Distillation for Robust Image Forgery Detection
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**GeoNormality** is a robust image forgery detection framework that leverages **depth information** from monocular depth estimation to enhance detection performance. The framework combines **Swin Transformer** backbone with multi-modal fusion strategies and knowledge distillation from Depth Anything V3.

**When this paper have already been published, we will upload the other code files immediately.**
---

## 🪐 Key Features

- ✅ **Multi-level Depth Fusion**: Three-tier fusion architecture (input-level enhancement, feature-level fusion, output-level supervision)
- ✅ **Two Fusion Schemes**: Early fusion (Scheme 1) and late fusion with skip connections (Scheme 2, recommended)
- ✅ **Depth Knowledge Distillation**: Leverages Depth Anything V3 as teacher for depth map, edge, and contrastive learning
- ✅ **Robust Architecture**: Swin Transformer + Noise Branch + PAA + R-DRB + PCR Loss
- ✅ **Region-Aware Distillation**: Smart masking strategy focusing on authentic regions and tampered edges
- ✅ **Flexible Configuration**: YAML-based config system with ablation experiment support

---

## 📊 Performance

| Configuration | F1 Score | IoU | AUC | Training Time |
|--------------|----------|-----|-----|---------------|
| Baseline (No Depth) | ~0.65 | ~0.48 | ~0.82 | - |

*Results may vary depending on dataset and training settings.*

---

## 🏗️ Architecture

### Multi-Level Depth Fusion

```
Coming Soon!
```

### Two Fusion Schemes

**Scheme 1: Early Fusion (Fusion Layer)**
- Concatenates RGB, Noise, and Depth features at fusion layer
- PAA learns adaptive weights for all three modalities
- Simple and effective

**Scheme 2: Late Fusion (Skip Connections) ⭐ Recommended**
- Injects multi-scale depth features into decoder via residual connections
- Preserves spatial details at multiple scales
- Better performance with similar parameter count

---

## 🏃🏻‍♀️ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd deep-ds

# Install dependencies
pip install -r requirements.txt

# Install Depth Anything V3 (optional, for depth distillation)
cd models/DepthAnything3
pip install -e .
cd ../..
```

### 2. Prepare Data

Organize your dataset in the following structure:

```
train_add/
├── images/
│   ├── img001.png
│   ├── img002.png
│   └── ...
└── masks/
    ├── img001_gt.png
    ├── img002_gt.png
    └── ...
```

### 3. Download Pretrained Models

```bash
# Download Swin-S pretrained weights
mkdir -p pretrained
# Place swin_s.safetensors in pretrained/ directory
```

### 4. Training

#### Basic Training (Scheme 2, Recommended)

```bash
python train_with_depth.py --config config/ablation_scheme2.yaml --gpu 0
```

#### With Custom Settings

```bash
python train_with_depth.py \
    --config config/ablation_scheme2.yaml \
    --gpu 0 \
    --opts train.lr 1.5e-4 data.batch_size 16
```

#### Ablation Experiments

```bash
# Run all ablation experiments
chmod +x run_ablation_experiments.sh
./run_ablation_experiments.sh

# Or run individual experiments
python train_with_depth.py --config config/ablation_no_depth.yaml --gpu 0
python train_with_depth.py --config config/ablation_implicit_only.yaml --gpu 0
python train_with_depth.py --config config/ablation_scheme1.yaml --gpu 0
python train_with_depth.py --config config/ablation_scheme2.yaml --gpu 0
```

### 5. Visualization

```bash
# Visualize depth features
python visualize_depth_forgery.py --image_path path/to/image.png

# Visualize modality features
python visualize_modality_features.py --image_path path/to/image.png
```

---

## 📁 Project Structure

```
deep-ds/
├── config/                          # Configuration files
│   ├── default.yaml                 # Default configuration
│   ├── ablation_no_depth.yaml       # Baseline (no depth)
│   ├── ablation_implicit_only.yaml  # Implicit fusion only
│   ├── ablation_scheme1.yaml        # Scheme 1 (early fusion)
│   ├── ablation_scheme2.yaml        # Scheme 2 (skip connect, recommended)
│   ├── ablation_scheme2_bs8.yaml    # Optimized for small batch size
│   ├── ablation_scheme2_rtx5090.yaml # Optimized for RTX 5090
│   ├── ablation_scheme2_real_edge.yaml # With edge distillation
│   └── ablation_scheme2_contrastive.yaml # With contrastive distillation
│
├── models/                          # Model definitions

├── data/                            # Data loading
│   ├── dataset.py                   # Dataset class
│   └── test_dataset.py              # Dataset testing script
│
├── utils/                           # Utilities


├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚙️ Configuration

### Key Parameters

#### Model Configuration

```yaml
model:
  use_depth_branch: True              # Enable depth branch
  depth_fusion_scheme: 'skip_connection'  # 'fusion_layer' or 'skip_connection'
  use_paa: false                      # Use PAA (recommended: false for Scheme 2)
  use_rdrb: true                      # Use R-DRB
  swin_reduce_dim: 256                # Swin feature reduction dimension
```

#### Training Configuration

```yaml
train:
  epochs: 20
  lr: 1.5e-4
  weight_decay: 5.0e-5
  
  # Distillation weights
  lambda_depth_distill: 0.15          # Depth distillation weight
  lambda_edge_distill: 0.05           # Edge distillation weight
  lambda_contrastive_distill: 0.0     # Contrastive distillation weight
  
  # Distillation strategy
  distill_strategy: 'region'          # 'full' or 'region'
  use_real_plus_edge_distill: true    # Use real region + edge strategy
  use_contrastive_distill: false      # Enable contrastive distillation
  
  # Optimization
  accumulation_steps: 1               # Gradient accumulation steps
  use_amp: false                      # Automatic mixed precision
```

#### Data Configuration

```yaml
data:
  img_dir: ./train_add/images
  mask_dir: ./train_add/masks
  mask_suffix: "_gt.png"
  img_size: [384, 384]
  batch_size: 16
  num_workers: 8
  noise_method: "multiscale_laplacian"
```

---

## 🔬 Ablation Studies

The project includes comprehensive ablation experiment configurations:

### Experiment Suite

1. **No Depth Baseline** (`ablation_no_depth.yaml`)
   - Establishes baseline performance without depth information
   
2. **Implicit Only** (`ablation_implicit_only.yaml`)
   - Tests input-level depth enhancement only
   
3. **Scheme 1** (`ablation_scheme1.yaml`)
   - Early fusion at fusion layer
   
4. **Scheme 2** (`ablation_scheme2.yaml`) ⭐
   - Late fusion with skip connections (recommended)
   
5. **Scheme 2 + Edge** (`ablation_scheme2_real_edge.yaml`)
   - Adds edge distillation to Scheme 2
   
6. **Scheme 2 + Contrastive** (`ablation_scheme2_contrastive.yaml`)
   - Adds contrastive distillation to Scheme 2

### Running Ablations

```bash
# Run all experiments sequentially
./run_ablation_experiments.sh

# Run specific experiment
python train_with_depth.py --config config/ablation_scheme2.yaml --gpu 0
```

See `ABLATION_EXPERIMENTS_GUIDE.md` for detailed instructions.

---

## 💡 Advanced Usage

### Memory Optimization

For systems with limited memory (32GB RAM):

```yaml
# config/ablation_scheme2_bs8.yaml
data:
  batch_size: 8
  num_workers: 4

train:
  lr: 7.5e-5
  accumulation_steps: 2
  use_amp: true
  epochs: 30
```

```bash
python train_with_depth.py --config config/ablation_scheme2_bs8.yaml --gpu 0
```

### GPU-Specific Optimization

For RTX 5090 (32GB VRAM):

```yaml
# config/ablation_scheme2_rtx5090.yaml
data:
  batch_size: 12
  num_workers: 6

train:
  lr: 1.0e-4
  accumulation_steps: 1
  use_amp: true
  epochs: 25
```

```bash
python train_with_depth.py --config config/ablation_scheme2_rtx5090.yaml --gpu 0
```

### Dynamic Configuration Override

```bash
# Override any config parameter via command line
python train_with_depth.py \
    --config config/ablation_scheme2.yaml \
    --opts \
        train.lr 1.0e-4 \
        data.batch_size 12 \
        train.lambda_edge_distill 0.05 \
        train.use_real_plus_edge_distill true
```

---

## 📈 Monitoring & Visualization

### TensorBoard

```bash
tensorboard --logdir logs/
```

### Training Logs

Logs are saved in `logs/<experiment_name>/`:
- `training_log.csv`: Epoch-by-epoch metrics
- `<experiment>.log`: Detailed training log
- `events.out.tfevents.*`: TensorBoard events

### Checkpoints

Model checkpoints are saved in `checkpoints_<timestamp>/<experiment_name>/`:
- Automatically saves best model based on F1 score
- Saves every N epochs (configured by `save_freq`)

---

## 🎯 Key Design Decisions

### Why Scheme 2 is Recommended?

1. **Multi-scale Features**: Captures depth patterns at different resolutions
2. **Spatial Preservation**: Skip connections preserve fine-grained details
3. **Better Performance**: Consistently outperforms Scheme 1 by 2-4% F1
4. **Similar Parameters**: Comparable model size to Scheme 1

### Why Disable PAA for Scheme 2?

- Depth features are injected via skip connections (not controlled by PAA)
- PAA only controls RGB+Noise fusion, adding unnecessary complexity
- Simplified architecture trains more stably

### Distillation Head Design

**Edge Head**: 3×3 Conv + ReLU + 1×1 Conv + Sigmoid
- Needs Sigmoid for BCE loss (output range [0,1])
- 3×3 conv captures local edge patterns

**Depth Head**: 3×3 Conv + ReLU + 1×1 Conv (no activation)
- No Sigmoid needed (L1/MSE handles any range)
- Allows flexible depth value learning
- Avoids gradient saturation

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{deepds2024,
  title={Deep-DS: Depth-Enhanced Image Forgery Detection with Swin Transformer},
  author={Zhengdao Liu},
  year={2024},
  journal={arXiv preprint}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
---

## 👨‍💻 Author

**Zhengdao Liu and Chenghao Shen**

---

## 🙏 Acknowledgments

- [Swin Transformers](https://github.com/microsoft/Swin-Transformer)
- [Depth Anything V3](https://github.com/ByteDance-Seed/depth-anything-3)
- [timm](https://github.com/huggingface/pytorch-image-models)

---

## 📧 Contact

For questions or collaborations, please open an issue or contact me.

Best wishes!

Zhengdao Liu

📮: karatelife_liu@outlook.com 

🍠: 4350277740 
---

**Happy Detecting! 🎉**
