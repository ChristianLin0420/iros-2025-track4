# Evaluation Results Repository

This document records evaluation metrics for various methodologies tested on the re_bbox task.

## Metrics Explanation

### Text Retrieval Metrics
- **txt_r1**: Text-to-Image Retrieval Recall@1 (%)
- **txt_r5**: Text-to-Image Retrieval Recall@5 (%)  
- **txt_r10**: Text-to-Image Retrieval Recall@10 (%)
- **txt_r_mean**: Average of txt_r1, txt_r5, txt_r10

### Image Retrieval Metrics
- **img_r1**: Image-to-Text Retrieval Recall@1 (%)
- **img_r5**: Image-to-Text Retrieval Recall@5 (%)
- **img_r10**: Image-to-Text Retrieval Recall@10 (%)
- **img_r_mean**: Average of img_r1, img_r5, img_r10

### Overall Performance
- **r_mean**: Overall mean of txt_r_mean and img_r_mean

## Evaluation Results

### Baseline
**Date**: 2025-01-09  
**Model**: GeoText1652 Official Checkpoint  
**Evaluation Time**: 0:26:49  

| Metric | Value |
|--------|-------|
| txt_r1 | 51.11% |
| txt_r5 | 81.37% |
| txt_r10 | 90.45% |
| txt_r_mean | 74.31% |
| img_r1 | 29.96% |
| img_r5 | 46.14% |
| img_r10 | 54.01% |
| img_r_mean | 43.37% |
| **r_mean** | **58.84%** |

**Configuration**:
- Image resolution: 384x384
- Batch size: 1
- Max tokens: 50
- Embed dimension: 256
- Temperature: 0.07
- K_test: 256

**Command Used**:
```bash
python3 run.py --task "re_bbox" --dist "gpu0" --evaluate --output_dir "output/eva" --checkpoint "../checkpoints/GeoText1652_model/geotext_official_checkpoint.pth"
```

## Performance Comparison

Each row represents a different method, with columns showing various evaluation metrics:

| Method | txt_r1 | txt_r5 | txt_r10 | txt_r_mean | img_r1 | img_r5 | img_r10 | img_r_mean | r_mean | Eval Time |
|--------|--------|--------|---------|------------|--------|--------|---------|------------|---------|-----------|
| Baseline | 51.11% | 81.37% | 90.45% | 74.31% | 29.96% | 46.14% | 54.01% | 43.37% | **58.84%** | 0:26:49 |
| [Method 2] | - | - | - | - | - | - | - | - | - | - |
| [Method 3] | - | - | - | - | - | - | - | - | - | - |

**To add a new method**: Replace `[Method X]` with your method name and fill in the corresponding metric values.

## Best Results So Far

- **Best r_mean**: 58.84% (Baseline)
- **Best txt_r1**: 51.11% (Baseline)
- **Best img_r1**: 29.96% (Baseline)

## Experimental Setup

### Dataset
- **Training**: `/localhome/local-chrislin/vla/iros-2025-track4/datasets/track4-cross-modal-drone-navigation/train.json`
- **Testing**: `/localhome/local-chrislin/vla/iros-2025-track4/datasets/track4-cross-modal-drone-navigation/test_24G_version.json`
- **Images**: `/localhome/local-chrislin/vla/iros-2025-track4/datasets/track4-cross-modal-drone-navigation/images`

### Hardware
- GPU: [Add GPU information]
- Memory: [Add memory information]

### Software Environment
- PyTorch version: [Add version]
- CUDA version: [Add version]
- Python version: [Add version]

## Notes

1. All results are evaluated on the test set unless otherwise specified
2. Evaluation time may vary based on hardware and system load
3. Higher values are better for all metrics
4. r_mean is the primary metric for overall performance comparison 