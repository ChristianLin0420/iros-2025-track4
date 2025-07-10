#!/usr/bin/env python3
"""
Two-Stage Training Script for InternVL3 Cross-modal Drone Navigation

This script orchestrates the complete two-stage training process:
- Stage 1: Foundation training with ITC + ITM losses only
- Stage 2: Multi-task refinement with all losses and progressive weighting

Usage:
    python train_two_stage.py --output_dir ./output/two_stage_training --use_wandb
"""

import argparse
import os
import sys
import shutil
import subprocess
import time
import datetime
import json
from pathlib import Path
from ruamel.yaml import YAML

def print_stage_header(stage_name, description):
    """Print a formatted header for each stage"""
    print("\n" + "="*80)
    print(f"  {stage_name}")
    print(f"  {description}")
    print("="*80)

def run_stage_training(config_path, output_dir, stage_name, use_wandb=False, checkpoint_path=None):
    """Run training for a specific stage"""
    print(f"\n🚀 Starting {stage_name} training...")
    
    # Build command
    cmd = [
        sys.executable, "internvl3_bbox.py",
        "--config", config_path,
        "--output_dir", output_dir,
        "--device", "cuda"
    ]
    
    if use_wandb:
        cmd.append("--use_wandb")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        cmd.extend(["--checkpoint", checkpoint_path])
        print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    print(f"📜 Command: {' '.join(cmd)}")
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        end_time = time.time()
        duration = str(datetime.timedelta(seconds=int(end_time - start_time)))
        print(f"✅ {stage_name} completed successfully in {duration}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {stage_name} failed with error: {e}")
        return False

def find_best_checkpoint(output_dir, pattern="checkpoint_*.pth"):
    """Find the best checkpoint from a directory"""
    import glob
    checkpoints = glob.glob(os.path.join(output_dir, pattern))
    if not checkpoints:
        return None
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]

def update_stage2_config(stage2_config_path, stage1_checkpoint_path):
    """Update Stage 2 configuration with Stage 1 checkpoint path"""
    yaml = YAML()
    yaml.preserve_quotes = True
    
    with open(stage2_config_path, 'r') as f:
        config = yaml.load(f)
    
    # Update checkpoint path
    config['stage2_config']['stage1_checkpoint_path'] = stage1_checkpoint_path
    
    with open(stage2_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"📝 Updated Stage 2 config with checkpoint: {stage1_checkpoint_path}")

def validate_environment():
    """Validate that the environment is set up correctly"""
    print("🔍 Validating environment...")
    
    # Check if required files exist
    required_files = [
        "configs/internvl3_stage1.yaml",
        "configs/internvl3_stage2.yaml",
        "models/model_internvl3_two_stage.py",
        "internvl3_bbox.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, training will be slow")
        else:
            gpu_count = torch.cuda.device_count()
            print(f"✅ Found {gpu_count} GPU(s)")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    print("✅ Environment validation passed")
    return True

def create_experiment_summary(output_dir, stage1_time, stage2_time, stage1_success, stage2_success):
    """Create a summary of the training experiment"""
    summary = {
        "experiment_name": "InternVL3_Two_Stage_Training",
        "timestamp": datetime.datetime.now().isoformat(),
        "stages": {
            "stage1": {
                "description": "Foundation training (ITC + ITM)",
                "duration": stage1_time,
                "success": stage1_success,
                "config": "configs/internvl3_stage1.yaml"
            },
            "stage2": {
                "description": "Multi-task refinement (All losses)",
                "duration": stage2_time,
                "success": stage2_success,
                "config": "configs/internvl3_stage2.yaml"
            }
        },
        "total_duration": str(datetime.timedelta(seconds=int(stage1_time + stage2_time))),
        "output_directory": output_dir
    }
    
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Experiment summary saved to: {summary_path}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Training for InternVL3 Cross-modal Drone Navigation")
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for training results')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--skip_stage1', action='store_true', help='Skip Stage 1 and start from Stage 2')
    parser.add_argument('--stage1_checkpoint', type=str, help='Path to Stage 1 checkpoint (if skipping Stage 1)')
    parser.add_argument('--dry_run', action='store_true', help='Validate setup without running training')
    
    args = parser.parse_args()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    if args.dry_run:
        print("✅ Dry run completed successfully")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Print experiment header
    print_stage_header(
        "InternVL3 Two-Stage Training",
        "Cross-modal Drone Navigation with Progressive Multi-task Learning"
    )
    
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 WandB logging: {'Enabled' if args.use_wandb else 'Disabled'}")
    
    # Stage directories
    stage1_dir = os.path.join(args.output_dir, "stage1")
    stage2_dir = os.path.join(args.output_dir, "stage2")
    
    # Initialize timing variables
    stage1_time = 0
    stage2_time = 0
    stage1_success = False
    stage2_success = False
    
    # ==================== STAGE 1: Foundation Training ====================
    if not args.skip_stage1:
        print_stage_header(
            "STAGE 1: Foundation Training",
            "Core cross-modal alignment with ITC + ITM losses (3 epochs)"
        )
        
        Path(stage1_dir).mkdir(parents=True, exist_ok=True)
        
        stage1_start = time.time()
        stage1_success = run_stage_training(
            config_path="configs/internvl3_stage1.yaml",
            output_dir=stage1_dir,
            stage_name="Stage 1",
            use_wandb=args.use_wandb
        )
        stage1_time = time.time() - stage1_start
        
        if not stage1_success:
            print("❌ Stage 1 failed. Aborting two-stage training.")
            sys.exit(1)
        
        # Find best Stage 1 checkpoint
        stage1_checkpoint = find_best_checkpoint(stage1_dir)
        if not stage1_checkpoint:
            print("❌ No Stage 1 checkpoint found. Aborting.")
            sys.exit(1)
        
        print(f"✅ Stage 1 completed. Best checkpoint: {stage1_checkpoint}")
    
    else:
        print("⏩ Skipping Stage 1")
        stage1_checkpoint = args.stage1_checkpoint
        if not stage1_checkpoint or not os.path.exists(stage1_checkpoint):
            print("❌ Stage 1 checkpoint not found. Please provide valid checkpoint path.")
            sys.exit(1)
        stage1_success = True
    
    # ==================== STAGE 2: Multi-task Refinement ====================
    print_stage_header(
        "STAGE 2: Multi-task Refinement",
        "Progressive multi-task learning with all losses (5 epochs)"
    )
    
    Path(stage2_dir).mkdir(parents=True, exist_ok=True)
    
    # Update Stage 2 config with Stage 1 checkpoint
    stage2_config_path = "configs/internvl3_stage2.yaml"
    stage2_config_updated = os.path.join(stage2_dir, "internvl3_stage2.yaml")
    shutil.copy(stage2_config_path, stage2_config_updated)
    update_stage2_config(stage2_config_updated, stage1_checkpoint)
    
    stage2_start = time.time()
    stage2_success = run_stage_training(
        config_path=stage2_config_updated,
        output_dir=stage2_dir,
        stage_name="Stage 2",
        use_wandb=args.use_wandb,
        checkpoint_path=stage1_checkpoint
    )
    stage2_time = time.time() - stage2_start
    
    # ==================== Training Summary ====================
    print_stage_header(
        "Training Summary",
        "Two-stage training completed"
    )
    
    # Create experiment summary
    summary = create_experiment_summary(
        args.output_dir, stage1_time, stage2_time, stage1_success, stage2_success
    )
    
    # Print results
    if stage1_success and stage2_success:
        print("🎉 Two-stage training completed successfully!")
        print(f"📈 Stage 1 duration: {datetime.timedelta(seconds=int(stage1_time))}")
        print(f"📈 Stage 2 duration: {datetime.timedelta(seconds=int(stage2_time))}")
        print(f"📈 Total duration: {datetime.timedelta(seconds=int(stage1_time + stage2_time))}")
        
        # Find final checkpoint
        final_checkpoint = find_best_checkpoint(stage2_dir)
        if final_checkpoint:
            print(f"🏆 Final model checkpoint: {final_checkpoint}")
        
        print(f"📂 All results saved to: {args.output_dir}")
    
    elif stage1_success:
        print("⚠️  Stage 1 completed but Stage 2 failed")
        print("🔍 Check Stage 2 logs for debugging")
    
    else:
        print("❌ Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 