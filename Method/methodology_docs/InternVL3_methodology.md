# InternVL3-1B Cross-Modal Retrieval Methodology

## Overview

This methodology replaces the baseline's separate BERT text encoder and Swin vision encoder with **InternVL3-1B**, a unified Vision-Language Model that provides better cross-modal alignment and understanding capabilities.

## Key Improvements

1. **Unified VLM Architecture**: Uses InternVL3-1B's integrated vision and text encoders
2. **Better Cross-Modal Alignment**: Pre-trained on large-scale vision-language data
3. **Enhanced Spatial Understanding**: Improved spatial relation modeling
4. **Efficient Training**: Adapter-based fine-tuning approach
5. **Comprehensive Logging**: WandB integration for experiment tracking

## Architecture

### Comprehensive Model Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     INPUT PROCESSING                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  Image: [B, 3, 384, 384]                    Text: [B, max_tokens]                      │
│  ↓ Preprocessing                            ↓ Tokenization                              │
│  • Resize to 384x384                        • BERT Tokenizer                            │
│  • Normalize RGB                            • Padding to max_tokens                     │
│  • Tensor conversion                        • Attention masks                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              ↓
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                INTERNVL3-1B BASE MODEL                                 │
│                                (1B Parameters - FROZEN)                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────┐            ┌─────────────────────────────┐           │
│  │        VISION PATH          │            │          TEXT PATH          │           │
│  │   ┌─────────────────────┐   │            │   ┌─────────────────────┐   │           │
│  │   │    ViT-Large-336    │   │            │   │     LLaMA-2-7B      │   │           │
│  │   │                     │   │            │   │                     │   │           │
│  │   │ • Patch: 14x14      │   │            │   │ • Vocab: 32000      │   │           │
│  │   │ • Layers: 24        │   │            │   │ • Layers: 32        │   │           │
│  │   │ • Heads: 16         │   │            │   │ • Heads: 32         │   │           │
│  │   │ • Hidden: 1024      │   │            │   │ • Hidden: 4096      │   │           │
│  │   └─────────────────────┘   │            │   └─────────────────────┘   │           │
│  └─────────────────────────────┘            └─────────────────────────────┘           │
│              ↓                                           ↓                            │
│  Vision Features: [B, 576, 1024]            Text Features: [B, max_tokens, 896]      │
│  • 576 = (384/14)² patches                  • Dynamic dimension detected              │
│  • 1024d per patch                          • 896d per token (detected)              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              ↓
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DYNAMIC ADAPTER LAYERS                                    │
│                             (Trainable Parameters: ~528K)                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────┐            ┌─────────────────────────────┐           │
│  │      VISION ADAPTER         │            │       TEXT ADAPTER          │           │
│  │                             │            │                             │           │
│  │  Input: [B, 576, 1024]      │            │  Input: [B, max_tokens, 896]│           │
│  │        ↓                    │            │        ↓                    │           │
│  │  Linear(1024 → 256)         │            │  Linear(896 → 256)          │           │
│  │        ↓                    │            │        ↓                    │           │
│  │  ReLU + Dropout(0.1)        │            │  ReLU + Dropout(0.1)        │           │
│  │        ↓                    │            │        ↓                    │           │
│  │  Output: [B, 576, 256]      │            │  Output: [B, max_tokens, 256]│           │
│  │                             │            │                             │           │
│  │  Trainable Params: 262K     │            │  Trainable Params: 230K     │           │
│  └─────────────────────────────┘            └─────────────────────────────┘           │
│                                                                                         │
│  # Feature Extraction (CLS tokens)                                                     │
│  Vision Embedding: [B, 256]  ←── vision_features[:, 0, :]                             │
│  Text Embedding: [B, 256]    ←── text_features[:, 0, :]                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              ↓
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                CROSS-MODAL FUSION                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────┐            ┌─────────────────────────────┐           │
│  │     VISION PROJECTOR        │            │      TEXT PROJECTOR         │           │
│  │                             │            │                             │           │
│  │  Input: [B, 256]            │            │  Input: [B, 256]            │           │
│  │        ↓                    │            │        ↓                    │           │
│  │  Linear(256 → 256)          │            │  Linear(256 → 256)          │           │
│  │        ↓                    │            │        ↓                    │           │
│  │  L2 Normalization           │            │  L2 Normalization           │           │
│  │        ↓                    │            │        ↓                    │           │
│  │  Vision Proj: [B, 256]      │            │  Text Proj: [B, 256]        │           │
│  │                             │            │                             │           │
│  │  Trainable Params: 65K      │            │  Trainable Params: 65K      │           │
│  └─────────────────────────────┘            └─────────────────────────────┘           │
│                                                                                         │
│  # Concatenated Features for Multi-task Heads                                          │
│  Fused Features: [B, 512] ←── torch.cat([vision_proj, text_proj], dim=1)              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              ↓
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MULTI-TASK PREDICTION HEADS                               │
│                             (Trainable Parameters: ~36K)                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐           │
│  │   ITM HEAD          │  │   BBOX HEAD         │  │  SPATIAL HEAD       │           │
│  │                     │  │                     │  │                     │           │
│  │ Input: [B, 512]     │  │ Input: [B, 512]     │  │ Input: [B, 512]     │           │
│  │       ↓             │  │       ↓             │  │       ↓             │           │
│  │ Linear(512 → 2)     │  │ Linear(512 → 4)     │  │ Linear(512 → 6)     │           │
│  │       ↓             │  │       ↓             │  │       ↓             │           │
│  │ Match Logits: [B,2] │  │ BBox Coords: [B,4]  │  │ Spatial Rels: [B,6] │           │
│  │ • [No Match, Match] │  │ • [x1, y1, x2, y2]  │  │ • [left, right,     │           │
│  │                     │  │                     │  │   above, below,     │           │
│  │ Params: 1K          │  │ Params: 2K          │  │   inside, outside]  │           │
│  └─────────────────────┘  └─────────────────────┘  │ Params: 3K          │           │
│                                                   └─────────────────────┘           │
│                                                                                         │
│  # Contrastive Learning (ITC)                                                          │
│  Vision Proj: [B, 256] ──┐                                                             │
│  Text Proj: [B, 256]   ──┼──→ Similarity Matrix: [B, B]                               │
│  Learnable Temp: 0.07 ──┘    ↓                                                        │
│                              Contrastive Loss (InfoNCE)                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                              ↓
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  LOSS COMPUTATION                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                            LOSS COMPONENTS                                          │ │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                                     │ │
│  │  1. ITC Loss (Image-Text Contrastive):                                             │ │
│  │     • InfoNCE loss on similarity matrix                                            │ │
│  │     • Weight: 1.0                                                                  │ │
│  │     • Formula: -log(exp(sim_i2t) / Σ exp(sim_all))                                │ │
│  │                                                                                     │ │
│  │  2. ITM Loss (Image-Text Matching):                                                │ │
│  │     • Binary cross-entropy on match predictions                                    │ │
│  │     • Weight: 1.0                                                                  │ │
│  │     • Formula: BCE(match_logits, match_targets)                                    │ │
│  │                                                                                     │ │
│  │  3. BBox Loss (Bounding Box Regression):                                           │ │
│  │     • L1 loss on bounding box coordinates                                          │ │
│  │     • Weight: 0.1                                                                  │ │
│  │     • Formula: L1(bbox_pred, bbox_target)                                          │ │
│  │                                                                                     │ │
│  │  4. Spatial Loss (Spatial Relation Classification):                                │ │
│  │     • Cross-entropy on spatial relation predictions                                │ │
│  │     • Weight: 0.1                                                                  │ │
│  │     • Formula: CE(spatial_logits, spatial_targets)                                 │ │
│  │                                                                                     │ │
│  │  Total Loss = ITC_loss + ITM_loss + 0.1*BBox_loss + 0.1*Spatial_loss             │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                          TRAINING PARAMETERS                                        │ │
│  ├─────────────────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                                     │ │
│  │  • Optimizer: AdamW (lr=1e-4, weight_decay=0.01)                                   │ │
│  │  • Scheduler: Linear with warmup (0.1 ratio)                                       │ │
│  │  • Batch Size: 4 (optimized for A40 GPU)                                           │ │
│  │  • Epochs: 5                                                                       │ │
│  │  • Precision: Float32 (for stable adapter training)                                │ │
│  │  • Gradient Accumulation: Dynamic based on memory                                  │ │
│  │  • Only Adapters + Task Heads are trainable (~528K params)                        │ │
│  │  • Base InternVL3-1B model is frozen (1B params)                                   │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Technical Pipeline Details

#### 1. **Dynamic Dimension Detection**
- Vision features: Automatically detected as 1024-dimensional
- Text features: Automatically detected as 896-dimensional (not 1024 as initially assumed)
- Adapters dynamically created based on actual feature dimensions

#### 2. **Parameter Efficiency**
```
Total Parameters: ~1.528B
├── Frozen InternVL3-1B: 1B parameters (99.97%)
└── Trainable Components: ~528K parameters (0.03%)
    ├── Vision Adapter: 262K params (1024×256 + 256)
    ├── Text Adapter: 230K params (896×256 + 256)
    ├── Vision Projector: 65K params (256×256 + 256)
    ├── Text Projector: 65K params (256×256 + 256)
    ├── ITM Head: 1K params (512×2 + 2)
    ├── BBox Head: 2K params (512×4 + 4)
    └── Spatial Head: 3K params (512×6 + 6)
```

#### 3. **Memory Optimization Strategy**
- **Float32 Precision**: Ensures stable gradient flow through adapters
- **Gradient Checkpointing**: Reduces memory usage during backpropagation
- **Batch Size 4**: Optimized for A40 GPU memory constraints
- **Frozen Base Model**: Significantly reduces memory requirements

#### 4. **Multi-Task Learning Approach**
- **Primary Tasks**: Image-Text Contrastive (ITC) + Image-Text Matching (ITM)
- **Auxiliary Tasks**: Bounding Box Regression + Spatial Relation Classification
- **Loss Weighting**: Balanced approach with reduced weights for auxiliary tasks

#### 5. **Training Stability Features**
- **Dynamic Adapter Creation**: Handles varying feature dimensions
- **Comprehensive Error Handling**: Robust to spatial loss computation failures
- **Gradient Flow Management**: Proper module registration and in-place operation handling
- **WandB Integration**: Real-time monitoring and logging

## 🧪 Evaluation Commands

### Original InternVL3 Single-Stage Training

#### Training
```bash
# Standard training
python internvl3_bbox.py \
    --config configs/internvl3_bbox.yaml \
    --output_dir ./output/internvl3_single_stage \
    --use_wandb

# Training with custom batch size
python internvl3_bbox.py \
    --config configs/internvl3_bbox.yaml \
    --output_dir ./output/internvl3_single_stage \
    --bs 8 \
    --use_wandb
```

#### Evaluation
```bash
# Evaluate trained model
python internvl3_bbox.py \
    --config configs/internvl3_bbox.yaml \
    --checkpoint ./output/internvl3_single_stage/checkpoint_4.pth \
    --evaluate \
    --output_dir ./output/internvl3_eval

# Evaluation with WandB logging
python internvl3_bbox.py \
    --config configs/internvl3_bbox.yaml \
    --checkpoint ./output/internvl3_single_stage/checkpoint_4.pth \
    --evaluate \
    --output_dir ./output/internvl3_eval \
    --use_wandb
```

### Two-Stage Training Approach

#### Training
```bash
# Automated two-stage training
python train_two_stage.py \
    --output_dir ./output/two_stage_training \
    --use_wandb

# Manual Stage 1 training
python internvl3_bbox.py \
    --config configs/internvl3_stage1.yaml \
    --output_dir ./output/stage1 \
    --use_wandb

# Manual Stage 2 training (requires Stage 1 checkpoint)
python internvl3_bbox.py \
    --config configs/internvl3_stage2.yaml \
    --output_dir ./output/stage2 \
    --checkpoint ./output/stage1/checkpoint_2.pth \
    --use_wandb
```

#### Evaluation
```bash
# Evaluate Stage 1 model
python internvl3_bbox.py \
    --config configs/internvl3_stage1.yaml \
    --checkpoint ./output/stage1/checkpoint_2.pth \
    --evaluate \
    --output_dir ./output/stage1_eval

# Evaluate Stage 2 model (final model)
python internvl3_bbox.py \
    --config configs/internvl3_stage2.yaml \
    --checkpoint ./output/stage2/checkpoint_4.pth \
    --evaluate \
    --output_dir ./output/stage2_eval \
    --use_wandb
```

### Comparative Analysis

#### Performance Comparison
```bash
# Compare single-stage vs two-stage
python eval_compare_baselines.py \
    --single_stage_checkpoint ./output/internvl3_single_stage/checkpoint_4.pth \
    --two_stage_checkpoint ./output/stage2/checkpoint_4.pth \
    --output_dir ./output/comparison

# Benchmark evaluation
python benchmark_evaluation.py \
    --checkpoints ./output/internvl3_single_stage/checkpoint_4.pth ./output/stage2/checkpoint_4.pth \
    --labels "Single-Stage" "Two-Stage" \
    --output_dir ./output/benchmark
```

#### Training Efficiency Analysis
```bash
# Analyze training logs
python analyze_training_efficiency.py \
    --single_stage_log ./output/internvl3_single_stage/log.txt \
    --two_stage_log ./output/two_stage_training/experiment_summary.json \
    --output_dir ./output/efficiency_analysis
```

### Evaluation Metrics

The evaluation framework measures:

1. **Cross-modal Retrieval**: R@1, R@5, R@10 for both image-to-text and text-to-image
2. **Bounding Box Accuracy**: IoU, GIoU, L1 loss
3. **Spatial Relations**: Classification accuracy, F1 score
4. **Training Efficiency**: Epochs to convergence, training time
5. **Resource Usage**: GPU memory, parameter efficiency

### Expected Performance Comparison

| Metric | Single-Stage | Two-Stage | Improvement |
|--------|-------------|-----------|-------------|
| R@1 (I2T) | 65-68% | 70-73% | +5% |
| R@5 (I2T) | 85-88% | 88-91% | +3% |
| R@10 (I2T) | 92-94% | 94-96% | +2% |
| Training Epochs | 12-15 | 8 | 40-47% faster |
| Convergence Time | 8-10 hours | 6 hours | 25-40% faster |
| Memory Usage | ~18GB | ~16GB | 11% reduction |

### Methodology References

For detailed implementation of the two-stage training approach, see:
- `methodology_docs/internvl3_two_stage_methodology.md` - Complete two-stage training methodology
- `configs/internvl3_stage1.yaml` - Stage 1 configuration
- `configs/internvl3_stage2.yaml` - Stage 2 configuration  
- `train_two_stage.py` - Automated two-stage training script
- `models/model_internvl3_two_stage.py` - Enhanced two-stage model implementation