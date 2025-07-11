import argparse
import os
import sys
import math
import warnings
warnings.filterwarnings("ignore")

from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_internvl3_two_stage import InternVL3TwoStageModel
import logging
import gc

from transformers import AutoTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

# WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: WandB not available. Install with: pip install wandb")

def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bb', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_spatial', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    
    # WandB logging setup
    use_wandb = config.get('wandb', {}).get('project', None) is not None and WANDB_AVAILABLE
    log_interval = config.get('wandb', {}).get('log_interval', 10)
    
    for i, (image, text, idx, sens, target_bboxes) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        sens = [list(i) for i in zip(*sens)]
        batch_length = idx.size(0)
        
        # Move to device
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        
        # Tokenize text
        caption = tokenizer(text, padding='longest', max_length=config['max_tokens'], 
                           return_tensors="pt", truncation=True).to(device)
        
        # Prepare bbox pairs
        pair_text_bbox = []
        for batch_idx in range(batch_length):  # Changed from 'i' to 'batch_idx'
            for j in range(3):
                if target_bboxes[batch_idx][j][0] > 0:
                    target_bbox = target_bboxes[batch_idx][j].to(device)  # Move to device
                    sen = sens[batch_idx][j]
                    
                    sen_token = tokenizer(sen, padding='longest', max_length=config['max_tokens'], 
                                        return_tensors="pt", truncation=True).to(device)
                    pair_text_bbox.append([batch_idx, sen_token, target_bbox])
        
        # Forward pass
        outputs = model(image, caption.input_ids, caption.attention_mask, idx=idx, pair=pair_text_bbox)
        
        # Parse outputs
        if len(outputs) == 2:
            loss_itc, loss_itm = outputs
            loss = loss_itc + loss_itm
            loss_bb = torch.tensor(0.0, device=device)
            loss_spatial = torch.tensor(0.0, device=device)
        elif len(outputs) == 3:
            loss_itc, loss_itm, loss_bb = outputs
            loss = loss_itc + loss_itm + loss_bb
            loss_spatial = torch.tensor(0.0, device=device)
        elif len(outputs) == 4:
            loss_itc, loss_itm, loss_bb, loss_spatial = outputs
            loss = loss_itc + loss_itm + loss_bb + loss_spatial
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update metrics - ensure all losses are tensors
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_bb=loss_bb.item() if hasattr(loss_bb, 'item') else loss_bb)
        metric_logger.update(loss_spatial=loss_spatial.item() if hasattr(loss_spatial, 'item') else loss_spatial)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # WandB logging
        if use_wandb and i % log_interval == 0 and utils.is_main_process():
            wandb.log({
                'train/loss_total': loss.item(),
                'train/loss_itc': loss_itc.item(),
                'train/loss_itm': loss_itm.item(),
                'train/loss_bb': loss_bb.item() if hasattr(loss_bb, 'item') else loss_bb,
                'train/loss_spatial': loss_spatial.item() if hasattr(loss_spatial, 'item') else loss_spatial,
                'train/lr': optimizer.param_groups[0]["lr"],
                'train/epoch': epoch,
                'train/step': epoch * len(data_loader) + i
            })
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    
    print('Computing features for evaluation...')
    start_time = time.time()
    
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']
    
    # Extract text features
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, 
                              max_length=config['max_tokens'], return_tensors="pt").to(device)
        
        # Get text features from model
        text_features = model.get_text_features(text_input.input_ids, text_input.attention_mask)
        text_embed = F.normalize(model.text_proj(text_features[:, 0, :]), dim=-1)
        text_embeds.append(text_embed)
    
    text_embeds = torch.cat(text_embeds, dim=0)
    
    # Extract image features
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        vision_features = model.get_vision_features(image)
        image_embed = F.normalize(model.vision_proj(vision_features[:, 0, :]), dim=-1)
        image_embeds.append(image_embed)
    
    image_embeds = torch.cat(image_embeds, dim=0)
    
    print("Feature extraction completed")
    
    # Compute similarity matrix
    sims_matrix = image_embeds @ text_embeds.t()
    
    # Image-to-text retrieval
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        score_matrix_i2t[start + i, topk_idx] = topk_sim
    
    # Synchronize across processes
    if utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
    
    score_matrix_i2t_c = score_matrix_i2t.cpu().numpy()
    
    # Text-to-image retrieval
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
    
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        score_matrix_t2i[start + i, topk_idx] = topk_sim
    
    if utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    
    return score_matrix_i2t_c, score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, img2building):
    """Evaluate retrieval performance"""
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        for i in range(len(inds)):
            inds[i] = img2building[txt2img[inds[i]]]
        target = np.where(inds == img2building[index])[0]
        ranks[index] = target[0]
    
    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    
    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        for i in range(len(inds)):
            inds[i] = img2building[inds[i]]
        building = img2building[txt2img[index]]
        ranks[index] = np.where(inds == building)[0][0]
    
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    
    eval_result = {
        'txt_r1': tr1,
        'txt_r5': tr5,
        'txt_r10': tr10,
        'txt_r_mean': tr_mean,
        'img_r1': ir1,
        'img_r5': ir5,
        'img_r10': ir10,
        'img_r_mean': ir_mean,
        'r_mean': r_mean
    }
    
    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    
    world_size = utils.get_world_size()
    
    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size
    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # Initialize WandB
    if args.use_wandb and config.get('wandb', {}).get('project', None) is not None and WANDB_AVAILABLE and utils.is_main_process():
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            tags=config['wandb']['tags'],
            config=config
        )
    
    print("Creating InternVL3 two-stage model", flush=True)
    model = InternVL3TwoStageModel(config=config)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
    
    # Use InternVL3 tokenizer
    tokenizer = model_without_ddp.tokenizer
    
    print("Creating dataset", flush=True)
    train_dataset, test_dataset = create_dataset('re_bbox', config, args.evaluate)
    
    start_time = time.time()
    print("### output_dir: ", args.output_dir, flush=True)
    
    if args.evaluate:
        print("Start evaluating", flush=True)
        test_loader = create_loader([test_dataset], [None],
                                   batch_size=[config['batch_size_test']],
                                   num_workers=[0],
                                   is_trains=[False],
                                   collate_fns=[None])[0]
        
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
        
        if utils.is_main_process():
            test_result = itm_eval(score_test_i2t, score_test_t2i, 
                                 test_loader.dataset.txt2img, test_loader.dataset.img2txt, 
                                 test_loader.dataset.img2building)
            print(test_result)
            
            # Log to WandB
            if args.use_wandb and WANDB_AVAILABLE and config.get('wandb', {}).get('project', None) is not None:
                wandb.log({
                    'eval/txt_r1': test_result['txt_r1'],
                    'eval/txt_r5': test_result['txt_r5'],
                    'eval/txt_r10': test_result['txt_r10'],
                    'eval/txt_r_mean': test_result['txt_r_mean'],
                    'eval/img_r1': test_result['img_r1'],
                    'eval/img_r5': test_result['img_r5'],
                    'eval/img_r10': test_result['img_r10'],
                    'eval/img_r_mean': test_result['img_r_mean'],
                    'eval/r_mean': test_result['r_mean']
                })
        
        if utils.is_dist_avail_and_initialized():
            dist.barrier()
    
    else:
        print("Start training", flush=True)
        
        train_dataset_size = len(train_dataset)
        
        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")
        
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
        else:
            samplers = [None, None]
        
        train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers,
                                                 batch_size=[config['batch_size_train']] + [config['batch_size_test']],
                                                 num_workers=[0, 0],
                                                 is_trains=[True, False],
                                                 collate_fns=[None, None])
        
        # Initialize adapters before training to avoid in-place operation errors
        print("Initializing adapters...", flush=True)
        sample_batch = next(iter(train_loader))
        sample_image, sample_text, sample_idx, sample_sens, sample_target_bboxes = sample_batch
        sample_image = sample_image[:1].to(device)  # Take first sample
        sample_caption = tokenizer(sample_text[:1], padding='longest', max_length=config['max_tokens'], 
                                  return_tensors="pt", truncation=True).to(device)
        
        # Initialize adapters by running a forward pass
        model_without_ddp.initialize_adapters_if_needed(
            sample_image, 
            sample_caption.input_ids, 
            sample_caption.attention_mask
        )
        print("Adapters initialized successfully!", flush=True)
        
        # Create optimizer and scheduler
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)
        
        max_epoch = config['schedular']['epochs']
        best_r_mean = 0
        best_epoch = 0
        
        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            
            # Update model epoch for progressive training
            if hasattr(model_without_ddp, 'update_epoch'):
                model_without_ddp.update_epoch(epoch + 1)  # 1-indexed epochs
            
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)
            
            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
                
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                
                # Save checkpoint
                if epoch <= config['schedular']['epochs'] - 1:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))
            
            if utils.is_dist_avail_and_initialized():
                dist.barrier()
            torch.cuda.empty_cache()
        
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)
            
            os.system(f"cat {args.output_dir}/log.txt")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB for logging')
    
    args = parser.parse_args()
    
    yaml = YAML(typ='rt')
    with open(args.config, 'r') as file:
        config = yaml.load(file)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    
    main(args, config) 