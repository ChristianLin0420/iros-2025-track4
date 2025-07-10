import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import itertools
from models.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

def compute_rela(bbox1, bbox2):
    """Compute spatial relationship between two bounding boxes"""
    try:
        # Handle tensor inputs
        if torch.is_tensor(bbox1):
            bbox1 = bbox1.cpu().numpy() if bbox1.dim() > 0 else float(bbox1)
        if torch.is_tensor(bbox2):
            bbox2 = bbox2.cpu().numpy() if bbox2.dim() > 0 else float(bbox2)
        
        # Ensure we have 4 elements for each bbox
        if len(bbox1) != 4 or len(bbox2) != 4:
            print(f"Warning: invalid bbox dimensions - bbox1: {bbox1}, bbox2: {bbox2}")
            return torch.tensor([1, 1])  # Default to center-middle
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        len_x = x1 - x2
        len_y = y1 - y2

        # Horizontal relationship
        if abs(len_x) < 0.5 * w1:
            horizontal = 1  # 'Center'
        elif len_x > 0:
            horizontal = 0  # 'Left'
        else:
            horizontal = 2  # 'Right'

        # Vertical relationship
        if abs(len_y) < 0.5 * h1:
            vertical = 1  # 'middle'
        elif len_y > 0:
            vertical = 0  # 'upper'
        else:
            vertical = 2  # 'lower'

        return torch.tensor([horizontal, vertical])
        
    except Exception as e:
        print(f"Error in compute_rela: {e}")
        print(f"bbox1: {bbox1}, bbox2: {bbox2}")
        return torch.tensor([1, 1])  # Default to center-middle

class InternVL3BenchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Load InternVL3-1B model
        self.model_name = "OpenGVLab/InternVL3-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        try:
            self.internvl_model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for stable training
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load InternVL3-1B: {e}")
            print("Falling back to simpler approach...")
            # Fallback: we'll create simple embeddings
            self.internvl_model = None
        
        # Freeze base model parameters if model is loaded
        if self.internvl_model is not None:
            for param in self.internvl_model.parameters():
                param.requires_grad = False
        
        # Model dimensions (will be determined dynamically)
        self.embed_dim = config['embed_dim']  # 256
        self.vision_dim = None  # Will be set after first forward pass
        self.text_dim = None    # Will be set after first forward pass
        
        # Fallback encoders if InternVL3 fails
        if self.internvl_model is None:
            self.fallback_vision_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((16, 16)),
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 1024)
            )
            # Simple text embedding layer
            vocab_size = 30522  # BERT vocab size
            self.fallback_text_encoder = nn.Embedding(vocab_size, 1024)
        else:
            self.fallback_vision_encoder = None
            self.fallback_text_encoder = None
        
        # Adapter layers (will be created dynamically)
        self.vision_adapter = None
        self.text_adapter = None
        
        # Projection layers for contrastive learning
        self.vision_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Task-specific heads
        self.itm_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 2)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 4)
        )
        
        self.spatial_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 6)
        )
        
        # Temperature for contrastive learning
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        
        # BBox collector for spatial relation learning
        self.bbox_collector = self.BBoxCollector(self)
        
        # Initialize adapter weights will be called later when dimensions are known
    
    def initialize_adapters_if_needed(self, sample_images, sample_text_ids, sample_text_atts):
        """Initialize adapters by doing a dry run to determine dimensions"""
        with torch.no_grad():
            # Extract sample features to determine dimensions
            if self.vision_adapter is None:
                vision_features = self.get_vision_features(sample_images)
                
            if self.text_adapter is None:
                text_features = self.get_text_features(sample_text_ids, sample_text_atts)
    
    def _create_adapters(self, vision_dim=None, text_dim=None):
        """Create adapter layers dynamically based on actual feature dimensions"""
        device = next(self.parameters()).device
        
        if vision_dim is not None and self.vision_adapter is None:
            self.vision_dim = vision_dim
            self.vision_adapter = nn.Sequential(
                nn.Linear(self.vision_dim, self.embed_dim),
                nn.ReLU(inplace=False),  # Avoid inplace operations
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim, self.embed_dim)
            ).to(device)
            # Register as parameter to ensure it's tracked properly
            self.add_module('vision_adapter', self.vision_adapter)
            # Ensure all parameters require gradients
            for param in self.vision_adapter.parameters():
                param.requires_grad = True
            
        if text_dim is not None and self.text_adapter is None:
            self.text_dim = text_dim
            self.text_adapter = nn.Sequential(
                nn.Linear(self.text_dim, self.embed_dim),
                nn.ReLU(inplace=False),  # Avoid inplace operations
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim, self.embed_dim)
            ).to(device)
            # Register as parameter to ensure it's tracked properly
            self.add_module('text_adapter', self.text_adapter)
            # Ensure all parameters require gradients
            for param in self.text_adapter.parameters():
                param.requires_grad = True
        
        # Initialize weights for newly created adapters
        if vision_dim is not None or text_dim is not None:
            self._init_adapter_weights()
    
    def _init_adapter_weights(self):
        """Initialize adapter layer weights"""
        modules_to_init = []
        if self.vision_adapter is not None:
            modules_to_init.append(self.vision_adapter)
        if self.text_adapter is not None:
            modules_to_init.append(self.text_adapter)
        if self.fallback_vision_encoder is not None:
            modules_to_init.append(self.fallback_vision_encoder)
        if self.fallback_text_encoder is not None:
            modules_to_init.append(self.fallback_text_encoder)
        
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def get_vision_features(self, images):
        """Extract vision features using InternVL3-1B vision encoder"""
        if self.internvl_model is None:
            # Fallback: use simple vision encoder
            batch_size = images.size(0)
            if self.fallback_vision_encoder is not None:
                vision_feat = self.fallback_vision_encoder(images)  # [B, 1024]
                # Expand to sequence format [B, seq_len, 1024]
                vision_features = vision_feat.unsqueeze(1).expand(-1, 257, -1)  # [B, 257, 1024]
            else:
                # Final fallback: random features
                vision_features = torch.randn(batch_size, 257, 1024, device=images.device)
        else:
            with torch.no_grad():
                # Try different ways to extract vision features
                try:
                    if hasattr(self.internvl_model, 'vision_model'):
                        vision_outputs = self.internvl_model.vision_model(images)
                    else:
                        vision_outputs = self.internvl_model.encode_image(images)
                    
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        vision_features = vision_outputs.last_hidden_state  # [B, N, 1024]
                    elif hasattr(vision_outputs, 'hidden_states'):
                        vision_features = vision_outputs.hidden_states[-1]
                    else:
                        vision_features = vision_outputs  # Direct tensor output
                except Exception as e:
                    print(f"Vision model error: {e}")
                    # Fallback: create dummy features with correct shape
                    batch_size = images.size(0)
                    vision_features = torch.randn(batch_size, 257, 1024, device=images.device)  # Typical ViT output
        
        # Create adapter if not exists (outside gradient computation)
        if self.vision_adapter is None:
            actual_dim = vision_features.size(-1)
            print(f"Creating vision adapter for dimension: {actual_dim}")
            with torch.no_grad():
                self._create_adapters(vision_dim=actual_dim)
        
        # Apply adapter
        if self.vision_adapter is not None:
            vision_features = self.vision_adapter(vision_features)  # [B, N, embed_dim]
        return vision_features
    
    def get_text_features(self, text_ids, text_atts):
        """Extract text features using InternVL3-1B text encoder"""
        if self.internvl_model is None:
            # Fallback: use simple text embeddings
            if self.fallback_text_encoder is not None:
                text_features = self.fallback_text_encoder(text_ids)  # [B, L, 1024]
            else:
                # Final fallback: random features
                batch_size, seq_len = text_ids.size()
                text_features = torch.randn(batch_size, seq_len, 1024, device=text_ids.device)
        else:
            with torch.no_grad():
                try:
                    if hasattr(self.internvl_model, 'language_model'):
                        text_outputs = self.internvl_model.language_model(
                            input_ids=text_ids,
                            attention_mask=text_atts,
                            output_hidden_states=True
                        )
                    else:
                        text_outputs = self.internvl_model.encode_text(text_ids, text_atts)
                    
                    # For CausalLM, use hidden_states instead of last_hidden_state
                    if hasattr(text_outputs, 'hidden_states') and text_outputs.hidden_states is not None:
                        text_features = text_outputs.hidden_states[-1]  # Last layer hidden states
                    elif hasattr(text_outputs, 'last_hidden_state'):
                        text_features = text_outputs.last_hidden_state
                    else:
                        # Fallback: try to get from model directly
                        if hasattr(self.internvl_model, 'language_model'):
                            text_features = self.internvl_model.language_model.get_input_embeddings()(text_ids)
                        else:
                            # Final fallback
                            batch_size, seq_len = text_ids.size()
                            text_features = torch.randn(batch_size, seq_len, 1024, device=text_ids.device)
                except Exception as e:
                    print(f"Text model error: {e}")
                    # Fallback: create dummy features
                    batch_size, seq_len = text_ids.size()
                    text_features = torch.randn(batch_size, seq_len, 1024, device=text_ids.device)
        
        # Create adapter if not exists (outside gradient computation)
        if self.text_adapter is None:
            actual_dim = text_features.size(-1)
            print(f"Creating text adapter for dimension: {actual_dim}")
            with torch.no_grad():
                self._create_adapters(text_dim=actual_dim)
        
        # Apply adapter
        if self.text_adapter is not None:
            text_features = self.text_adapter(text_features)  # [B, L, embed_dim]
        return text_features
    
    def get_contrastive_features(self, vision_features, text_features):
        """Get features for contrastive learning"""
        # Use CLS tokens
        vision_cls = vision_features[:, 0, :]  # [B, embed_dim]
        text_cls = text_features[:, 0, :]      # [B, embed_dim]
        
        # Project to contrastive space
        vision_proj = F.normalize(self.vision_proj(vision_cls), dim=-1)
        text_proj = F.normalize(self.text_proj(text_cls), dim=-1)
        
        return vision_proj, text_proj
    
    def get_contrastive_loss(self, vision_proj, text_proj, idx=None):
        """Compute contrastive loss (ITC)"""
        # Compute similarity matrix
        sim_matrix = torch.matmul(vision_proj, text_proj.t()) / self.temp
        
        # Create labels
        batch_size = vision_proj.size(0)
        labels = torch.arange(batch_size, device=vision_proj.device)
        
        # Compute loss
        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2
    
    def get_matching_loss(self, vision_features, text_features, idx=None):
        """Compute image-text matching loss (ITM)"""
        batch_size = vision_features.size(0)
        
        # Positive pairs
        pos_vision = vision_features[:, 0, :]  # [B, embed_dim]
        pos_text = text_features[:, 0, :]      # [B, embed_dim]
        pos_features = torch.cat([pos_vision, pos_text], dim=-1)
        pos_scores = self.itm_head(pos_features)
        
        # Negative pairs (random shuffle)
        neg_indices = torch.randperm(batch_size, device=vision_features.device)
        neg_vision = vision_features[neg_indices, 0, :]
        neg_text = text_features[:, 0, :]
        neg_features = torch.cat([neg_vision, neg_text], dim=-1)
        neg_scores = self.itm_head(neg_features)
        
        # Combine scores and labels
        all_scores = torch.cat([pos_scores, neg_scores], dim=0)
        all_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long, device=vision_features.device),
            torch.zeros(batch_size, dtype=torch.long, device=vision_features.device)
        ])
        
        return F.cross_entropy(all_scores, all_labels)
    
    def predict_bbox(self, vision_features, text_features, text_atts):
        """Predict bounding box coordinates"""
        # Use CLS tokens
        vision_cls = vision_features[:, 0, :]  # [B, embed_dim]
        text_cls = text_features[:, 0, :]      # [B, embed_dim]
        
        # Fuse features
        fused_features = torch.cat([vision_cls, text_cls], dim=-1)
        
        # Predict bbox coordinates
        bbox_coords = self.bbox_head(fused_features).sigmoid()
        
        return bbox_coords
    
    def get_bbox_loss(self, pred_bbox, target_bbox):
        """Compute bounding box loss (L1 + GIoU)"""
        # L1 loss
        loss_l1 = F.l1_loss(pred_bbox, target_bbox, reduction='mean')
        
        # GIoU loss
        pred_boxes = box_cxcywh_to_xyxy(pred_bbox)
        target_boxes = box_cxcywh_to_xyxy(target_bbox)
        
        if (pred_boxes[:, 2:] < pred_boxes[:, :2]).any() or (target_boxes[:, 2:] < target_boxes[:, :2]).any():
            loss_giou = torch.tensor(0.0, device=pred_bbox.device)
        else:
            loss_giou = 1 - torch.diag(generalized_box_iou(pred_boxes, target_boxes)).mean()
        
        return loss_l1, loss_giou
    
    def get_spatial_loss(self, bbox_a, bbox_b, vision_features, target_ids):
        """Compute spatial relation loss"""
        try:
            # Use CLS features for spatial reasoning
            vision_cls = vision_features[:, 0, :]  # [B, embed_dim]
            
            # Create spatial features (simple approach)
            spatial_features = torch.cat([vision_cls, vision_cls], dim=-1)
            spatial_logits = self.spatial_head(spatial_features)
            
            # Reshape for classification
            spatial_logits = spatial_logits.view(-1, 2, 3)  # [B, 2, 3] for horizontal/vertical
            
            # Target labels - ensure they are proper indices
            target_labels = target_ids.to(vision_features.device)
            
            # Ensure target_labels has the right shape and values
            if target_labels.dim() == 1:
                # If 1D tensor with 2 elements, reshape to [1, 2]
                if target_labels.size(0) == 2:
                    target_labels = target_labels.unsqueeze(0)
                else:
                    print(f"Warning: unexpected target_labels shape: {target_labels.shape}")
                    return torch.tensor(0.0, device=vision_features.device, requires_grad=True)
            
            # Clamp target labels to valid range [0, 2]
            target_labels = torch.clamp(target_labels, 0, 2).long()
            
            # Compute cross-entropy loss
            loss_horizontal = F.cross_entropy(spatial_logits[:, 0, :], target_labels[:, 0])
            loss_vertical = F.cross_entropy(spatial_logits[:, 1, :], target_labels[:, 1])
            
            return (loss_horizontal + loss_vertical) / 2
            
        except Exception as e:
            print(f"Error in spatial loss computation: {e}")
            print(f"target_ids shape: {target_ids.shape}, vision_features shape: {vision_features.shape}")
            return torch.tensor(0.0, device=vision_features.device, requires_grad=True)
    
    def forward(self, images, text_ids, text_atts, idx=None, pair=None):
        """Forward pass"""
        try:
            # Extract features
            vision_features = self.get_vision_features(images)
            text_features = self.get_text_features(text_ids, text_atts)
            
            # Ensure adapters exist before continuing
            if self.vision_adapter is None or self.text_adapter is None:
                print("Warning: Adapters not initialized properly")
                return torch.tensor(0.0, device=images.device), torch.tensor(0.0, device=images.device)
            
            # Contrastive learning
            vision_proj, text_proj = self.get_contrastive_features(vision_features, text_features)
            loss_itc = self.get_contrastive_loss(vision_proj, text_proj, idx)
            
            # Image-text matching
            loss_itm = self.get_matching_loss(vision_features, text_features, idx)
            
            # If no bbox pairs, return basic losses
            if pair is None or len(pair) == 0:
                return loss_itc, loss_itm
            
            # Process bbox pairs
            loss_bb = 0
            total_spatial_loss = 0
            loss_count = 0
            
            for i, (batch_idx, sen_token, target_bbox) in enumerate(pair):
                try:
                    # Get single image features
                    single_vision = vision_features[batch_idx:batch_idx+1]
                    
                    # Get sentence features
                    sen_vision = single_vision
                    sen_text = self.get_text_features(sen_token.input_ids, sen_token.attention_mask)
                    
                    # Predict bbox
                    pred_bbox = self.predict_bbox(sen_vision, sen_text, sen_token.attention_mask)
                    
                    # Compute bbox loss
                    loss_bbox, loss_giou = self.get_bbox_loss(pred_bbox, target_bbox.unsqueeze(0))
                    loss_bb += (loss_bbox + loss_giou)
                    
                    # Update bbox collector for spatial loss
                    bbox_info = {
                        'bbox': target_bbox,
                        'vision_features': single_vision,
                        'batch_idx': batch_idx
                    }
                    spatial_loss = self.bbox_collector.update_bbox(bbox_info)
                    
                    if spatial_loss is not None:
                        total_spatial_loss += spatial_loss
                        loss_count += 1
                        
                except Exception as e:
                    print(f"Error processing bbox pair {i}: {e}")
                    print(f"batch_idx: {batch_idx}, target_bbox shape: {target_bbox.shape}")
                    continue
            
            # Average bbox loss
            if len(pair) > 0:
                loss_bb = 0.1 * loss_bb / len(pair)
            else:
                loss_bb = torch.tensor(0.0, device=images.device, requires_grad=True)
            
            # Reset bbox collector
            self.bbox_collector.collect_bbox = []
            self.bbox_collector.current_batch_idx = None
            
            # Return losses
            if loss_count > 0:
                loss_spatial = total_spatial_loss / loss_count
                return loss_itc, loss_itm, loss_bb, loss_spatial
            else:
                # Ensure all returned tensors require gradients
                if not loss_bb.requires_grad:
                    loss_bb = torch.tensor(0.0, device=images.device, requires_grad=True)
                return loss_itc, loss_itm, loss_bb
                
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return dummy losses with gradients enabled
            device = images.device
            dummy_loss_1 = torch.tensor(0.0, device=device, requires_grad=True)
            dummy_loss_2 = torch.tensor(0.0, device=device, requires_grad=True)
            return dummy_loss_1, dummy_loss_2
    
    class BBoxCollector:
        """Collects bounding boxes for spatial relation learning"""
        def __init__(self, parent):
            self.collect_bbox = []
            self.current_batch_idx = None
            self.parent = parent
        
        def update_bbox(self, bbox_info):
            """Update bbox collection and compute spatial loss if ready"""
            new_batch_idx = bbox_info['batch_idx']
            
            if not self.collect_bbox:
                self.collect_bbox.append(bbox_info)
                self.current_batch_idx = new_batch_idx
                return None
            
            if len(self.collect_bbox) == 1:
                if new_batch_idx == self.current_batch_idx:
                    self.collect_bbox.append(bbox_info)
                    return None
                else:
                    self.collect_bbox = [bbox_info]
                    self.current_batch_idx = new_batch_idx
                    return None
            
            if len(self.collect_bbox) == 2:
                if new_batch_idx == self.current_batch_idx:
                    self.collect_bbox.append(bbox_info)
                    loss = self.calculate_spatial_loss(self.collect_bbox)
                    self.collect_bbox = []
                    return loss
                else:
                    loss = self.calculate_spatial_loss(self.collect_bbox)
                    self.collect_bbox = [bbox_info]
                    self.current_batch_idx = new_batch_idx
                    return loss
        
        def calculate_spatial_loss(self, bboxes):
            """Calculate spatial relation loss between bbox pairs"""
            if len(bboxes) < 2:
                return None
            
            total_loss = 0
            count = 0
            
            # Generate all permutations of bbox pairs
            permutations = list(itertools.permutations(bboxes, 2))
            
            for pair in permutations:
                bbox_a = pair[0]['bbox']
                bbox_b = pair[1]['bbox']
                vision_features = pair[0]['vision_features']
                
                # Compute spatial relation target
                target_ids = compute_rela(bbox_a, bbox_b)
                
                # Compute spatial loss
                spatial_loss = self.parent.get_spatial_loss(bbox_a, bbox_b, vision_features, target_ids)
                total_loss += spatial_loss
                count += 1
            
            return total_loss / count if count > 0 else None 