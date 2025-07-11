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
        # Determine device from input tensors
        device = bbox1.device if torch.is_tensor(bbox1) else (bbox2.device if torch.is_tensor(bbox2) else torch.device('cpu'))
        
        # Handle tensor inputs
        if torch.is_tensor(bbox1):
            bbox1 = bbox1.cpu().numpy() if bbox1.dim() > 0 else float(bbox1)
        if torch.is_tensor(bbox2):
            bbox2 = bbox2.cpu().numpy() if bbox2.dim() > 0 else float(bbox2)
        
        # Ensure we have 4 elements for each bbox
        if len(bbox1) != 4 or len(bbox2) != 4:
            print(f"Warning: invalid bbox dimensions - bbox1: {bbox1}, bbox2: {bbox2}")
            return torch.tensor([1, 1], device=device)  # Default to center-middle
        
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

        return torch.tensor([horizontal, vertical], device=device)
        
    except Exception as e:
        print(f"Error in compute_rela: {e}")
        print(f"bbox1: {bbox1}, bbox2: {bbox2}")
        # Try to infer device from error context
        device = bbox1.device if torch.is_tensor(bbox1) else (bbox2.device if torch.is_tensor(bbox2) else torch.device('cpu'))
        return torch.tensor([1, 1], device=device)  # Default to center-middle

class EnhancedAdapter(nn.Module):
    """Enhanced adapter with skip connections and layer normalization"""
    def __init__(self, input_dim, output_dim, dropout=0.1, use_layer_norm=True, use_skip_connection=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm
        self.use_skip_connection = use_skip_connection
        
        # Main adapter layers
        self.adapter_layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        
        # Skip connection projection (if dimensions differ)
        if use_skip_connection and input_dim != output_dim:
            self.skip_projection = nn.Linear(input_dim, output_dim)
        else:
            self.skip_projection = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights"""
        for m in self.adapter_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with optional skip connection and layer norm"""
        # Main adapter forward
        adapted = self.adapter_layers(x)
        
        # Skip connection
        if self.use_skip_connection:
            if self.skip_projection is not None:
                skip = self.skip_projection(x)
            else:
                skip = x
            adapted = adapted + skip
        
        # Layer normalization
        if self.use_layer_norm:
            adapted = self.layer_norm(adapted)
        
        return adapted

class InternVL3TwoStageModel(nn.Module):
    """InternVL3 model with two-stage training support"""
    
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
            self.internvl_model = None
        
        # Configuration
        self.config = config
        self.embed_dim = config['embed_dim']  # 256
        self.current_stage = config.get('loss_config', {}).get('stage', 1)
        
        # Model dimensions (will be determined dynamically)
        self.vision_dim = None
        self.text_dim = None
        
        # Initialize freeze status
        self.freeze_base_model = config.get('model_config', {}).get('freeze_base_model', True)
        self._freeze_base_model()
        
        # Enhanced adapter layers (will be created dynamically)
        self.vision_adapter = None
        self.text_adapter = None
        
        # Projection layers for contrastive learning
        self.vision_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Task-specific heads
        self.itm_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, 2)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, 4)
        )
        
        self.spatial_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, 6)
        )
        
        # Temperature for contrastive learning
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        
        # BBox collector for spatial relation learning
        self.bbox_collector = self.BBoxCollector(self)
        
        # Progressive loss weighting
        self.current_epoch = 0
        self.loss_weights = config.get('loss_weights', {})
        self.progressive_loss_config = config.get('progressive_loss', {})
        
        print(f"Initialized InternVL3TwoStageModel for Stage {self.current_stage}")
    
    def _freeze_base_model(self):
        """Freeze base model parameters"""
        if self.internvl_model is not None and self.freeze_base_model:
            for param in self.internvl_model.parameters():
                param.requires_grad = False
            print("Base InternVL3 model frozen")
    
    def _unfreeze_components(self, components):
        """Unfreeze specific components of the base model"""
        if self.internvl_model is None:
            return
            
        for component in components:
            if component == "last_layer":
                # Unfreeze last layer of vision and text models
                if hasattr(self.internvl_model, 'vision_model'):
                    for param in list(self.internvl_model.vision_model.parameters())[-10:]:
                        param.requires_grad = True
                if hasattr(self.internvl_model, 'language_model'):
                    for param in list(self.internvl_model.language_model.parameters())[-10:]:
                        param.requires_grad = True
                        
            elif component == "last_two_layers":
                # Unfreeze last two layers
                if hasattr(self.internvl_model, 'vision_model'):
                    for param in list(self.internvl_model.vision_model.parameters())[-20:]:
                        param.requires_grad = True
                if hasattr(self.internvl_model, 'language_model'):
                    for param in list(self.internvl_model.language_model.parameters())[-20:]:
                        param.requires_grad = True
        
        print(f"Unfroze components: {components}")
    
    def update_epoch(self, epoch):
        """Update current epoch for progressive training"""
        self.current_epoch = epoch
        
        # Update unfreezing schedule for Stage 2
        if self.current_stage == 2:
            unfreezing_schedule = self.config.get('model_config', {}).get('unfreezing_schedule', {})
            components_to_unfreeze = unfreezing_schedule.get(f'epoch_{epoch}', [])
            if components_to_unfreeze and "last_layer" in components_to_unfreeze:
                self._unfreeze_components(components_to_unfreeze)
    
    def get_progressive_loss_weights(self, epoch):
        """Get progressive loss weights for current epoch"""
        if not self.progressive_loss_config.get('enabled', False):
            return self.loss_weights
        
        # Get epoch-specific scaling
        epoch_scaling = self.progressive_loss_config.get('epoch_scaling', {})
        current_scaling = epoch_scaling.get(epoch, epoch_scaling.get(str(epoch), {}))
        
        # Apply scaling to base weights
        progressive_weights = self.loss_weights.copy()
        
        if 'bbox_scale' in current_scaling:
            progressive_weights['bbox_weight'] = self.loss_weights['bbox_weight'] * current_scaling['bbox_scale']
        if 'spatial_scale' in current_scaling:
            progressive_weights['spatial_weight'] = self.loss_weights['spatial_weight'] * current_scaling['spatial_scale']
        
        return progressive_weights
    
    def initialize_adapters_if_needed(self, sample_images, sample_text_ids, sample_text_atts):
        """Initialize adapters by doing a dry run to determine dimensions"""
        with torch.no_grad():
            # Extract sample features to determine dimensions
            if self.vision_adapter is None:
                vision_features = self.get_vision_features(sample_images)
                
            if self.text_adapter is None:
                text_features = self.get_text_features(sample_text_ids, sample_text_atts)
    
    def _create_adapters(self, vision_dim=None, text_dim=None):
        """Create enhanced adapter layers dynamically"""
        device = next(self.parameters()).device
        
        # Get adapter configuration
        adapter_config = self.config.get('model_config', {}).get('adapter_config', {})
        dropout = adapter_config.get('dropout', 0.1)
        use_layer_norm = adapter_config.get('use_layer_norm', True)
        use_skip_connection = adapter_config.get('use_skip_connection', True)
        
        if vision_dim is not None and self.vision_adapter is None:
            self.vision_dim = vision_dim
            self.vision_adapter = EnhancedAdapter(
                input_dim=self.vision_dim,
                output_dim=self.embed_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                use_skip_connection=use_skip_connection
            ).to(device)
            
            # Register as parameter to ensure it's tracked properly
            self.add_module('vision_adapter', self.vision_adapter)
            
        if text_dim is not None and self.text_adapter is None:
            self.text_dim = text_dim
            self.text_adapter = EnhancedAdapter(
                input_dim=self.text_dim,
                output_dim=self.embed_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                use_skip_connection=use_skip_connection
            ).to(device)
            
            # Register as parameter to ensure it's tracked properly
            self.add_module('text_adapter', self.text_adapter)
        
        if vision_dim is not None or text_dim is not None:
            print(f"Created enhanced adapters - Vision: {vision_dim}→{self.embed_dim}, Text: {text_dim}→{self.embed_dim}")
    
    def get_vision_features(self, images):
        """Extract vision features using InternVL3-1B vision encoder"""
        if self.internvl_model is None:
            # Fallback: create dummy features
            batch_size = images.size(0)
            vision_features = torch.randn(batch_size, 257, 1024, device=images.device)
        else:
            with torch.no_grad():
                # Extract vision features
                try:
                    if hasattr(self.internvl_model, 'vision_model'):
                        vision_outputs = self.internvl_model.vision_model(images)
                    else:
                        vision_outputs = self.internvl_model.encode_image(images)
                    
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        vision_features = vision_outputs.last_hidden_state
                    elif hasattr(vision_outputs, 'hidden_states'):
                        vision_features = vision_outputs.hidden_states[-1]
                    else:
                        vision_features = vision_outputs
                except Exception as e:
                    print(f"Vision model error: {e}")
                    batch_size = images.size(0)
                    vision_features = torch.randn(batch_size, 257, 1024, device=images.device)
        
        # Create adapter if not exists
        if self.vision_adapter is None:
            actual_dim = vision_features.size(-1)
            self._create_adapters(vision_dim=actual_dim)
        
        # Apply enhanced adapter
        if self.vision_adapter is not None:
            vision_features = self.vision_adapter(vision_features)
        
        return vision_features
    
    def get_text_features(self, text_ids, text_atts):
        """Extract text features using InternVL3-1B text encoder"""
        if self.internvl_model is None:
            # Fallback: create dummy features
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
                    
                    if hasattr(text_outputs, 'hidden_states') and text_outputs.hidden_states is not None:
                        text_features = text_outputs.hidden_states[-1]
                    elif hasattr(text_outputs, 'last_hidden_state'):
                        text_features = text_outputs.last_hidden_state
                    else:
                        batch_size, seq_len = text_ids.size()
                        text_features = torch.randn(batch_size, seq_len, 1024, device=text_ids.device)
                except Exception as e:
                    print(f"Text model error: {e}")
                    batch_size, seq_len = text_ids.size()
                    text_features = torch.randn(batch_size, seq_len, 1024, device=text_ids.device)
        
        # Create adapter if not exists
        if self.text_adapter is None:
            actual_dim = text_features.size(-1)
            self._create_adapters(text_dim=actual_dim)
        
        # Apply enhanced adapter
        if self.text_adapter is not None:
            text_features = self.text_adapter(text_features)
        
        return text_features
    
    def get_contrastive_features(self, vision_features, text_features):
        """Get features for contrastive learning"""
        # Use CLS tokens
        vision_cls = vision_features[:, 0, :]
        text_cls = text_features[:, 0, :]
        
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
        pos_vision = vision_features[:, 0, :]
        pos_text = text_features[:, 0, :]
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
        vision_cls = vision_features[:, 0, :]
        text_cls = text_features[:, 0, :]
        
        # Fuse features
        fused_features = torch.cat([vision_cls, text_cls], dim=-1)
        
        # Predict bbox coordinates
        bbox_coords = self.bbox_head(fused_features).sigmoid()
        
        return bbox_coords
    
    def get_bbox_loss(self, pred_bbox, target_bbox):
        """Compute bounding box loss (L1 + GIoU)"""
        # Ensure both tensors are on the same device
        target_bbox = target_bbox.to(pred_bbox.device)
        
        # L1 loss
        loss_l1 = F.l1_loss(pred_bbox, target_bbox, reduction='mean')
        
        # GIoU loss
        pred_boxes = box_cxcywh_to_xyxy(pred_bbox)
        target_boxes = box_cxcywh_to_xyxy(target_bbox)
        
        if (pred_boxes[:, 2:] < pred_boxes[:, :2]).any() or (target_boxes[:, 2:] < target_boxes[:, :2]).any():
            loss_giou = torch.tensor(0.0, device=pred_bbox.device, requires_grad=True)
        else:
            loss_giou = 1 - torch.diag(generalized_box_iou(pred_boxes, target_boxes)).mean()
        
        return loss_l1, loss_giou
    
    def get_spatial_loss(self, bbox_a, bbox_b, vision_features, target_ids):
        """Compute spatial relation loss"""
        try:
            # Use CLS features for spatial reasoning
            vision_cls = vision_features[:, 0, :]
            
            # Create spatial features
            spatial_features = torch.cat([vision_cls, vision_cls], dim=-1)
            spatial_logits = self.spatial_head(spatial_features)
            
            # Reshape for classification
            spatial_logits = spatial_logits.view(-1, 2, 3)
            
            # Target labels
            target_labels = target_ids.to(vision_features.device)
            
            if target_labels.dim() == 1:
                if target_labels.size(0) == 2:
                    target_labels = target_labels.unsqueeze(0)
                else:
                    return torch.tensor(0.0, device=vision_features.device, requires_grad=True)
            
            # Clamp target labels to valid range
            target_labels = torch.clamp(target_labels, 0, 2).long()
            
            # Compute cross-entropy loss
            loss_horizontal = F.cross_entropy(spatial_logits[:, 0, :], target_labels[:, 0])
            loss_vertical = F.cross_entropy(spatial_logits[:, 1, :], target_labels[:, 1])
            
            return (loss_horizontal + loss_vertical) / 2
            
        except Exception as e:
            print(f"Error in spatial loss computation: {e}")
            return torch.tensor(0.0, device=vision_features.device, requires_grad=True)
    
    def forward(self, images, text_ids, text_atts, idx=None, pair=None):
        """Forward pass with stage-aware loss computation"""
        try:
            # Extract features
            vision_features = self.get_vision_features(images)
            text_features = self.get_text_features(text_ids, text_atts)
            
            # Ensure adapters exist
            if self.vision_adapter is None or self.text_adapter is None:
                print("Warning: Adapters not initialized properly")
                return torch.tensor(0.0, device=images.device), torch.tensor(0.0, device=images.device)
            
            # Get progressive loss weights
            current_weights = self.get_progressive_loss_weights(self.current_epoch)
            
            # Loss configuration
            loss_config = self.config.get('loss_config', {})
            losses = []
            
            # Always compute ITC and ITM losses
            if loss_config.get('enable_itc', True):
                vision_proj, text_proj = self.get_contrastive_features(vision_features, text_features)
                loss_itc = self.get_contrastive_loss(vision_proj, text_proj, idx)
                losses.append(loss_itc * current_weights.get('itc_weight', 1.0))
            
            if loss_config.get('enable_itm', True):
                loss_itm = self.get_matching_loss(vision_features, text_features, idx)
                losses.append(loss_itm * current_weights.get('itm_weight', 1.0))
            
            # Stage 1: Only return ITC + ITM losses
            if self.current_stage == 1 or not loss_config.get('enable_bbox', False):
                return losses[0] if len(losses) > 0 else torch.tensor(0.0), \
                       losses[1] if len(losses) > 1 else torch.tensor(0.0)
            
            # Stage 2: Add bbox and spatial losses
            if pair is not None and len(pair) > 0:
                loss_bb = torch.tensor(0.0, device=images.device, requires_grad=True)
                total_spatial_loss = torch.tensor(0.0, device=images.device, requires_grad=True)
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
                        target_bbox_tensor = target_bbox.unsqueeze(0).to(pred_bbox.device)
                        loss_bbox, loss_giou = self.get_bbox_loss(pred_bbox, target_bbox_tensor)
                        loss_bb = loss_bb + loss_bbox + loss_giou
                        
                        # Update bbox collector for spatial loss
                        bbox_info = {
                            'bbox': target_bbox,
                            'vision_features': single_vision,
                            'batch_idx': batch_idx
                        }
                        spatial_loss = self.bbox_collector.update_bbox(bbox_info)
                        
                        if spatial_loss is not None:
                            total_spatial_loss = total_spatial_loss + spatial_loss
                            loss_count += 1
                            
                    except Exception as e:
                        print(f"Error processing bbox pair {i}: {e}")
                        continue
                
                # Average bbox loss
                if len(pair) > 0:
                    loss_bb = current_weights.get('bbox_weight', 0.1) * loss_bb / len(pair)
                    losses.append(loss_bb)
                else:
                    # No bbox pairs - append zero loss
                    losses.append(torch.tensor(0.0, device=images.device, requires_grad=True))
                
                # Add spatial loss
                if loss_count > 0:
                    loss_spatial = current_weights.get('spatial_weight', 0.1) * total_spatial_loss / loss_count
                    losses.append(loss_spatial)
                else:
                    # No spatial pairs - append zero loss
                    losses.append(torch.tensor(0.0, device=images.device, requires_grad=True))
                
                # Reset bbox collector
                self.bbox_collector.collect_bbox = []
                self.bbox_collector.current_batch_idx = None
            
            # Return appropriate number of losses
            if len(losses) == 2:
                return losses[0], losses[1]
            elif len(losses) == 3:
                return losses[0], losses[1], losses[2]
            elif len(losses) == 4:
                return losses[0], losses[1], losses[2], losses[3]
            else:
                return losses[0], losses[1]
                
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.tensor(0.0, device=images.device), torch.tensor(0.0, device=images.device)
    
    class BBoxCollector:
        """Helper class for collecting bounding boxes and computing spatial relations"""
        
        def __init__(self, parent):
            self.parent = parent
            self.collect_bbox = []
            self.current_batch_idx = None
        
        def update_bbox(self, bbox_info):
            """Update bbox collector and compute spatial loss when appropriate"""
            batch_idx = bbox_info['batch_idx']
            
            # If new batch, reset collector
            if self.current_batch_idx != batch_idx:
                self.collect_bbox = []
                self.current_batch_idx = batch_idx
            
            # Add bbox to collector
            self.collect_bbox.append(bbox_info)
            
            # If we have 2 or more bboxes, compute spatial loss
            if len(self.collect_bbox) >= 2:
                return self.calculate_spatial_loss(self.collect_bbox)
            
            return None
        
        def calculate_spatial_loss(self, bboxes):
            """Calculate spatial relationship loss between bounding boxes"""
            if len(bboxes) < 2:
                return None
            
            try:
                # Take first two bboxes
                bbox1_info = bboxes[0]
                bbox2_info = bboxes[1]
                
                bbox1 = bbox1_info['bbox']
                bbox2 = bbox2_info['bbox']
                vision_features = bbox1_info['vision_features']
                
                # Compute spatial relationship
                spatial_relation = compute_rela(bbox1, bbox2)
                
                # Compute spatial loss
                spatial_loss = self.parent.get_spatial_loss(
                    bbox1, bbox2, vision_features, spatial_relation
                )
                
                return spatial_loss
                
            except Exception as e:
                print(f"Error calculating spatial loss: {e}")
                return None

# Create alias for backward compatibility
InternVL3BenchModel = InternVL3TwoStageModel 