import os
import random
from functools import partial
from copy import deepcopy
from timm.models.vision_transformer import Block
import torch
import time
from torch import nn
import torch.nn.functional as F
import clip
import numpy as np
from models.vit_mae import MaskedAutoencoderViT
from models.perceiver_resampler import PerceiverResampler
from models.gpt2 import GPT2Model
from transformers import GPT2Config
from pdb import set_trace
import random 
import matplotlib.pyplot as plt


def generate_attention_mask(K, num_A, num_B, atten_goal, atten_goal_state,
                            atten_only_obs,
                            attn_robot_proprio_state,
                            mask_l_obs_ratio,
                            num_obs_token, action_pred_steps):
    # num_A: 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
    # num_A: text, state, image_embedding, image_cls_token_embedding
    # num_B: self.NUM_OBS_TOKEN+self.action_pred_steps
    # num_B: obs_tokens(if exists), action_pred_token, state_pred_token (if exists)
    sequence_length = (num_A + num_B) * K
    attention_mask = torch.zeros((sequence_length, sequence_length))
    for i in range(K):
        start_index = i * (num_A + num_B)
        end_index = start_index + num_A + num_B
        
        # the i-th sub-sequence can not attend to the sub-sequences that after the i-th
        attention_mask[start_index:end_index, end_index:] = -float('inf')
        
        # the sub-sub-sequence B can not be attended to
        attention_mask[:, start_index+num_A:end_index] = -float('inf')
        
        # if obs_token exists, action_pred_token should attend to it
        if num_obs_token > 0 and action_pred_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
        if num_obs_token > 0 and atten_only_obs and action_pred_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps] = -float('inf')
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+2:start_index+num_A] = 0.0
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
            if attn_robot_proprio_state:
                attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+1:start_index+2] = 0.0
            if mask_l_obs_ratio > 0:
                count = int(mask_l_obs_ratio * (num_obs_token))
                selected_numbers = np.random.choice(range(num_obs_token), size=count, replace=False)
                for num in selected_numbers:
                    attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A+num] = -float('inf')
        if num_obs_token > 0 and atten_goal:
            if i < K - atten_goal:
                pred_end_index = (i + atten_goal) * (num_A + num_B)
                if atten_goal_state:
                    attention_mask[start_index+num_A:start_index+num_A+num_obs_token,pred_end_index+1:pred_end_index+2] = 0.0

    return attention_mask

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

class SeerAgent(nn.Module):
    def __init__(
        self,
        finetune_type,
        clip_device,
        vit_checkpoint_path,
        sequence_length=10,
        num_resampler_query=9,
        num_obs_token_per_image=10,
        obs_pred=False,
        atten_only_obs=False,
        attn_robot_proprio_state=False,
        atten_goal=False,
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        calvin_input_image_size=224,
        patch_size=16,
        mask_ratio=0.0,
        num_token_per_timestep=41,
        input_self=False,
        action_pred_steps=1,
        transformer_layers=12,
        hidden_dim=384,
        transformer_heads=12,
        phase="",
        gripper_width=False,
    ):
        super().__init__()
        self.finetune_type = finetune_type
        self.device = clip_device
        self.sequence_length = sequence_length
        self.action_pred_steps = action_pred_steps
        self.obs_pred = obs_pred
        self.atten_goal = atten_goal
        self.atten_goal_state = atten_goal_state
        self.atten_only_obs = atten_only_obs
        self.attn_robot_proprio_state = attn_robot_proprio_state
        self.mask_l_obs_ratio = mask_l_obs_ratio
        self.hidden_dim = hidden_dim
        self.phase = phase
        assert self.phase in ["pretrain", "finetune", "evaluate"]
        self.vit_checkpoint_path = vit_checkpoint_path

        # text projector
        self.text_projector = nn.Linear(512, self.hidden_dim)        

        # ==== STATE ENCODER (MODIFIED) ====
        # New state encoders for hand, pose, robot
        HAND_STATE_FEATURE_DIM = self.hidden_dim
        POSE_STATE_FEATURE_DIM = self.hidden_dim
        ROBOT_STATE_FEATURE_DIM = self.hidden_dim
        
        self.hand_state_encoder = nn.Linear(12, HAND_STATE_FEATURE_DIM)
        self.pose_state_encoder = nn.Linear(27, POSE_STATE_FEATURE_DIM)
        self.robot_state_encoder = nn.Linear(29, ROBOT_STATE_FEATURE_DIM)
        
        # Combined state projector
        self.state_projector = nn.Linear(
            HAND_STATE_FEATURE_DIM + POSE_STATE_FEATURE_DIM + ROBOT_STATE_FEATURE_DIM, 
            self.hidden_dim
        )

        # ==== ACTION ENCODER (MODIFIED) ====
        # New action encoders for hand, pose, robot
        HAND_ACTION_FEATURE_DIM = self.hidden_dim
        POSE_ACTION_FEATURE_DIM = self.hidden_dim
        ROBOT_ACTION_FEATURE_DIM = self.hidden_dim
        
        self.action_hand_encoder = nn.Linear(12, HAND_ACTION_FEATURE_DIM)
        self.action_pose_encoder = nn.Linear(24, POSE_ACTION_FEATURE_DIM)
        self.action_robot_encoder = nn.Linear(29, ROBOT_ACTION_FEATURE_DIM)
        
        # Combined action projector
        self.action_projector = nn.Linear(
            HAND_ACTION_FEATURE_DIM + POSE_ACTION_FEATURE_DIM + ROBOT_ACTION_FEATURE_DIM, 
            self.hidden_dim
        )

        # vision encoder (frozen)
        self.vision_encoder = MaskedAutoencoderViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # resampler
        self.RESAMPLER_hidden_dim = 768  
        self.NUM_RESAMPLER_QUERY = num_resampler_query
        self.perceiver_resampler = PerceiverResampler(dim=self.RESAMPLER_hidden_dim, num_latents=self.NUM_RESAMPLER_QUERY, depth=3)
        
        self.image_left_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
        self.cls_token_left_projector = nn.Linear(768, self.hidden_dim)
        self.image_right_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
        self.cls_token_right_projector = nn.Linear(768, self.hidden_dim)

        # action_pred_token
        if self.action_pred_steps > 0:
            self.action_pred_token = nn.Parameter(torch.zeros(1, 1, self.action_pred_steps, self.hidden_dim))

        # obs_token
        self.NUM_OBS_TOKEN_PER_IMAGE = num_obs_token_per_image
        self.NUM_OBS_TOKEN = self.NUM_OBS_TOKEN_PER_IMAGE * 2
        if self.obs_pred:
            self.obs_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_OBS_TOKEN, self.hidden_dim))
        
        # causal transformer
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_dim)
        
        if self.obs_pred:
            this_num_obs_token = self.NUM_OBS_TOKEN
        else:
            this_num_obs_token = 0
        self.attention_mask = nn.Parameter(generate_attention_mask(
                                    K=self.sequence_length, 
                                    num_A=1+1+self.NUM_RESAMPLER_QUERY*2+1*2, 
                                    num_B=this_num_obs_token+self.action_pred_steps,
                                    atten_goal=self.atten_goal,
                                    atten_goal_state=self.atten_goal_state,
                                    atten_only_obs=self.atten_only_obs,
                                    attn_robot_proprio_state = self.attn_robot_proprio_state,
                                    mask_l_obs_ratio=self.mask_l_obs_ratio,
                                    num_obs_token=this_num_obs_token,
                                    action_pred_steps=self.action_pred_steps), 
                                    requires_grad=False)
        num_non_learnable_token_per_timestep = 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
        self.transformer_backbone_position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, 1, self.hidden_dim), requires_grad=True)
        config = GPT2Config()
        config.hidden_size = self.hidden_dim
        config.n_layer = transformer_layers
        config.vocab_size = 1
        config.n_head = transformer_heads
        self.transformer_backbone = GPT2Model(config)

        # ==== ACTION DECODER (MODIFIED) ====
        MLP_hidden_dim = self.hidden_dim // 2
        self.action_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
            nn.Linear(MLP_hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
        )
        
        # New decoders for hand, pose, robot
        self.hand_action_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 12),
            # torch.nn.Tanh(),
        )
        
        self.pose_action_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 24),
            # torch.nn.Tanh(),
        )
        
        self.robot_action_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 29),
            # torch.nn.Tanh(),
        )

        # ==== STATE RECONSTRUCTION DECODER (MODIFIED) ====
        self.recon_state_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
            nn.Linear(MLP_hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
        ) # not used
        
        # New state reconstruction decoders
        self.recon_hand_state_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 12),
            # torch.nn.Tanh(),
        ) # not used
        
        self.recon_pose_state_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 27),
            # torch.nn.Tanh(),
        ) # not used
        
        self.recon_robot_state_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 29),
            # torch.nn.Tanh(),
        ) # not used

        # Image decoder components
        self.IMAGE_DECODER_hidden_dim = self.hidden_dim
        self.NUM_MASK_TOKEN = int(calvin_input_image_size**2 / patch_size / patch_size)  # i.e. num_patch
        self.PATCH_SIZE = patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.IMAGE_DECODER_hidden_dim))
        self.image_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.IMAGE_DECODER_hidden_dim)
        self.image_decoder_position_embedding = nn.Parameter(torch.zeros(1, self.NUM_OBS_TOKEN_PER_IMAGE + self.NUM_MASK_TOKEN, self.IMAGE_DECODER_hidden_dim), requires_grad=False)  # fixed sin-cos embedding #   cls_token is alse passed to the decoder in mae
        self.image_decoder = nn.Sequential(
            Block(self.IMAGE_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            Block(self.IMAGE_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
            )
        self.image_decoder_norm = nn.LayerNorm(self.IMAGE_DECODER_hidden_dim)
        self.image_decoder_pred = nn.Linear(self.IMAGE_DECODER_hidden_dim, self.PATCH_SIZE**2 * 3)

        # initialize network
        self.initialize_weights()

        # freeze vision encoder
        vit_checkpoint = torch.load(self.vit_checkpoint_path, map_location='cpu')
        msg = self.vision_encoder.load_state_dict(vit_checkpoint['model'], strict=False)

        # freeze text encoder
        if os.path.exists("checkpoints/clip/ViT-B-32.pt"):
            self.clip_model, self.image_processor = clip.load("checkpoints/clip/ViT-B-32.pt", device=clip_device)
        else:
            self.clip_model, self.image_processor = clip.load("ViT-B/32", device=clip_device)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        image_decoder_position_embedding_obs = get_2d_sincos_pos_embed(self.IMAGE_DECODER_hidden_dim, int(self.NUM_OBS_TOKEN_PER_IMAGE**.5), cls_token=False)
        image_decoder_position_embedding_mask = get_2d_sincos_pos_embed(self.IMAGE_DECODER_hidden_dim, int(self.NUM_MASK_TOKEN**.5), cls_token=False)
        image_decoder_position_embedding = np.concatenate((image_decoder_position_embedding_obs, image_decoder_position_embedding_mask), axis=0)
        self.image_decoder_position_embedding.data.copy_(torch.from_numpy(image_decoder_position_embedding).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.transformer_backbone_position_embedding, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_model_type(self):
        self.vision_encoder_type = next(self.vision_encoder.parameters()).type()
        self.perceiver_resampler_type = next(self.perceiver_resampler.parameters()).type()
        self.transformer_backbone_type = next(self.transformer_backbone.parameters()).type()
        self.action_decoder_type = next(self.action_decoder.parameters()).type()

    def forward(self, image_left, image_right, state, text_token, action=None):  
        """
        Modified forward pass to handle new input structure
        
        Args:
            image_left: Left camera image, shape [B, S, C, H, W]
            image_right: Right camera image, shape [B, S, C, H, W]
            state: State tensor containing hand, pose, robot parts 
                   [B, S, hand_dim(12) + pose_dim(27) + robot_dim(29)]
            text_token: Text tokens from CLIP, shape [B, S, token_length]
            action: Optional action tensor containing hand, pose, robot parts
                    [B, S, hand_dim(12) + pose_dim(24) + robot_dim(29)]
                    
        Returns:
            Predicted hand_action, pose_action, robot_action, image reconstruction,
            and optionally predicted states and loss
        """
        if self.training and self.phase == "pretrain":
            if self.obs_pred:
                this_num_obs_token = self.NUM_OBS_TOKEN
            else:
                this_num_obs_token = 0
            
            self.attention_mask = nn.Parameter(generate_attention_mask(
                            K=self.sequence_length, 
                            num_A=1+1+self.NUM_RESAMPLER_QUERY*2+1*2, 
                            num_B=this_num_obs_token+self.action_pred_steps,
                            atten_goal=self.atten_goal,
                            atten_goal_state=self.atten_goal_state,
                            atten_only_obs=self.atten_only_obs,
                            attn_robot_proprio_state = self.attn_robot_proprio_state,
                            mask_l_obs_ratio=self.mask_l_obs_ratio,
                            num_obs_token=this_num_obs_token,
                            action_pred_steps=self.action_pred_steps).to(self.device), 
                            requires_grad=False)
                            
        B, S, _ = state.shape
        device = image_left.device
        S_AND_FUTURE = image_left.shape[1]
        image_pred = None
        hand_pred_action, pose_pred_action, robot_pred_action = None, None, None
        hand_pred_state, pose_pred_state, robot_pred_state = None, None, None
        loss_action = None
        
        # text embedding
        with torch.no_grad():
            text_feature = self.clip_model.encode_text(text_token.flatten(0, 1))
            text_feature = text_feature.type(state.type())
        text_embedding = self.text_projector(text_feature)
        text_embedding = text_embedding.view(B, S, -1, self.hidden_dim) 

        # ==== STATE EMBEDDING (MODIFIED) ====
        state = state.flatten(0, 1)  # [B*S, state_dim]
        
        # Extract components from state
        # Assuming state is concatenated in order: [hand(12), pose(27), robot(29)]
        hand_state = state[:, :12]
        pose_state = state[:, 12:12+27]
        robot_state = state[:, 12+27:12+27+29]
        
        # Encode each component
        hand_state_feature = self.hand_state_encoder(hand_state)
        pose_state_feature = self.pose_state_encoder(pose_state)
        robot_state_feature = self.robot_state_encoder(robot_state)
        
        # Combine and project state features
        state_embedding = self.state_projector(
            torch.cat((hand_state_feature, pose_state_feature, robot_state_feature), dim=1)
        )
        state_embedding = state_embedding.view(B, S, -1, self.hidden_dim)

        # ==== IMAGE FEATURE EXTRACTION (RENAMED variables from primary/wrist to left/right) ====
        if image_left.type() != self.vision_encoder_type:
            image_left = image_left.type(self.vision_encoder_type)
            image_right = image_right.type(self.vision_encoder_type)
            
        with torch.no_grad():
            image_left_feature, _, _ = self.vision_encoder.forward_encoder(image_left.flatten(0, 1), mask_ratio=0.0)
            image_right_feature, _, _ = self.vision_encoder.forward_encoder(image_right.flatten(0, 1), mask_ratio=0.0)
            
        if image_left_feature.type() != self.perceiver_resampler_type:
            image_left_feature = image_left_feature.type(self.perceiver_resampler_type)
            image_right_feature = image_right_feature.type(self.perceiver_resampler_type)
            
        image_left_feature = image_left_feature.view(B, S_AND_FUTURE, image_left_feature.shape[-2], image_left_feature.shape[-1])
        image_right_feature = image_right_feature.view(B, S_AND_FUTURE, image_right_feature.shape[-2], image_right_feature.shape[-1])
        
        image_left_cls_token = image_left_feature[:, :, :1, :]
        image_right_cls_token = image_right_feature[:, :, :1, :]
        image_left_feature = image_left_feature[:, :, 1:, :]
        image_right_feature = image_right_feature[:, :, 1:, :]
        
        label_image_left_feature = image_left_feature.clone()
        label_image_right_feature = image_right_feature.clone()

        # perceiver resampler
        image_left_feature = self.perceiver_resampler(image_left_feature.reshape(B*S, 196, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))
        image_right_feature = self.perceiver_resampler(image_right_feature.reshape(B*S, 196, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))
        
        # Project features using renamed projectors
        image_left_embedding = self.image_left_projector(image_left_feature.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
        image_right_embedding = self.image_right_projector(image_right_feature.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
        image_embedding = torch.cat((image_left_embedding, image_right_embedding), dim=2)
        
        image_cls_token_left_embedding = self.cls_token_left_projector(image_left_cls_token.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
        image_cls_token_right_embedding = self.cls_token_right_projector(image_right_cls_token.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
        image_cls_token_embedding = torch.cat((image_cls_token_left_embedding, image_cls_token_right_embedding), dim=2)
        
        # aggregate embeddings and add timestep position encoding
        embeddings = torch.cat((text_embedding, state_embedding, image_embedding, image_cls_token_embedding), dim=2)
        pred_token_start_idx = embeddings.shape[2]
        transformer_input_list = [embeddings]
        if self.obs_pred:
            transformer_input_list.append(self.obs_tokens.repeat(B, S, 1, 1))
        if self.action_pred_steps > 0:
            transformer_input_list.append(self.action_pred_token.repeat(B, S, 1, 1))
        transformer_input = torch.cat(transformer_input_list, dim=2)  
        transformer_input = transformer_input + self.transformer_backbone_position_embedding.repeat(B, 1, transformer_input.shape[-2], 1)
        transformer_input = transformer_input.flatten(1, 2)

        # causal transformer forward
        if transformer_input.type() != self.transformer_backbone_type:
            transformer_input = transformer_input.type(self.transformer_backbone_type)
        transformer_input = self.embedding_layer_norm(transformer_input)
        transformer_output = self.transformer_backbone(inputs_embeds=transformer_input, attention_mask=self.attention_mask)
        transformer_output = transformer_output.view(B, S, -1, self.hidden_dim)

        # Image prediction if enabled
        if self.obs_pred:
            obs_pred_feature = transformer_output[:, :, pred_token_start_idx : pred_token_start_idx+self.NUM_OBS_TOKEN, :]
            obs_pred_embedding = self.image_decoder_obs_pred_projector(obs_pred_feature.reshape(-1, self.hidden_dim))
            obs_pred_embedding = obs_pred_embedding.view(B * S * (self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE), self.NUM_OBS_TOKEN_PER_IMAGE, self.IMAGE_DECODER_hidden_dim)
            mask_tokens = self.mask_token.repeat(B * S * (self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE), self.NUM_MASK_TOKEN, 1)
            image_decoder_input = torch.cat((obs_pred_embedding, mask_tokens), dim=1) 
            image_decoder_input = image_decoder_input + self.image_decoder_position_embedding
            image_decoder_output = self.image_decoder(image_decoder_input)
            image_pred_feature = image_decoder_output[:, -self.NUM_MASK_TOKEN:, :]
            image_pred_feature = self.image_decoder_norm(image_pred_feature.reshape(-1, self.IMAGE_DECODER_hidden_dim))
            image_pred = self.image_decoder_pred(image_pred_feature)  
            image_pred = image_pred.view(B * S, self.NUM_OBS_TOKEN // self.NUM_OBS_TOKEN_PER_IMAGE, self.NUM_MASK_TOKEN, -1)
        
        # ==== ACTION PREDICTION (MODIFIED) ====
        if self.action_pred_steps > 0:
            if self.obs_pred:
                this_num_obs_token = self.NUM_OBS_TOKEN
            else:
                this_num_obs_token = 0
                
            action_pred_feature = transformer_output[:, :, pred_token_start_idx+this_num_obs_token:pred_token_start_idx+this_num_obs_token+self.action_pred_steps, :]
            action_pred_feature = self.action_decoder(action_pred_feature)
            
            # Decode separate action components
            hand_pred_action = self.hand_action_decoder(action_pred_feature)
            pose_pred_action = self.pose_action_decoder(action_pred_feature)
            robot_pred_action = self.robot_action_decoder(action_pred_feature)
        
        return hand_pred_action, pose_pred_action, robot_pred_action, image_pred, hand_pred_state, pose_pred_state, robot_pred_state, loss_action


def test_my_seer_agent():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 2
    window_size = 10
    sequence_length = 7
    
    # 创建模拟图像数据 - 主视角和手腕视角
    # [batch_size, window_size, channels, height, width]
    image_left = torch.rand(batch_size, window_size, 3, 224, 224, device=device)
    image_right = torch.rand(batch_size, window_size, 3, 224, 224, device=device)
    
    # 创建模拟状态数据 - 包含hand, pose, robot状态
    # [batch_size, window_size, 12+27+29]
    state = torch.rand(batch_size, window_size, 68, device=device)
    
    # 创建模拟文本token数据 - CLIP模型使用的文本token
    # [batch_size, token_length]
    text_token = torch.randint(0, 49408, (batch_size, 77), device=device).unsqueeze(1).repeat(1, window_size, 1)  # CLIP词汇表大小为49408
    
    # 创建模拟动作数据（用于训练时）
    # [batch_size, window_size, 12+24+29] - 12个值是hand动作，24个值是pose动作，29个值是robot动作
    action = torch.rand(batch_size, window_size, 65, device=device)
    
    # 打印输入数据的形状
    print(f"Image Left Shape: {image_left.shape}")  # [2, 10, 3, 224, 224]
    print(f"Image Right Shape: {image_right.shape}")  # [2, 10, 3, 224, 224]
    print(f"State Shape: {state.shape}")  # [2, 10,  68]
    print(f"Text Token Shape: {text_token.shape}")  # [2, 10, 77]
    print(f"Action Shape: {action.shape}")  # [2, 10, 65]

    # 截断到sequence_length
    image_left = image_left[:, :sequence_length, :, :, :]
    image_right = image_right[:, :sequence_length, :, :, :]
    state = state[:, :sequence_length, :]
    text_token = text_token[:, :sequence_length, :]
    action = action[:, :sequence_length, :]
    
    # 初始化SeerAgent模型
    # 注意：这里使用了一些默认参数，实际使用时可能需要调整
    model = SeerAgent(
        finetune_type="real",
        clip_device=device,
        vit_checkpoint_path="ckpt/mae_pretrain_vit_base.pth",
        sequence_length=sequence_length,
        num_resampler_query=6,
        num_obs_token_per_image=9,
        calvin_input_image_size=224,
        patch_size=16,
        action_pred_steps=3,
        obs_pred=True,
        atten_only_obs=False,
        attn_robot_proprio_state=False,
        atten_goal=0,
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        transformer_layers=1, # Original: 24
        hidden_dim=384,
        transformer_heads=12,
        phase="finetune",
    ).to(device)

    model._init_model_type()
    
    model.eval()
    
    try:
        with torch.no_grad():
            hand_pred_action, pose_pred_action, robot_pred_action, image_pred, hand_pred_state, pose_pred_state, robot_pred_state, loss_action = model(
                image_left, image_right, state, text_token, action
            )
        
        # 打印输出结果的形状
        print("\nOutput shapes:")
        if hand_pred_action is not None:
            print(f"Hand Predicted Action Shape: {hand_pred_action.shape}")  # [2, 10, 12]
        if pose_pred_action is not None:
            print(f"Pose Predicted Action Shape: {pose_pred_action.shape}")  # [2, 10, 24]
        if robot_pred_action is not None:
            print(f"Robot Predicted Action Shape: {robot_pred_action.shape}")  # [2, 10, 29]
        if image_pred is not None:
            print(f"Image Prediction Shape: {image_pred.shape}")  # [20, 2, 196, 768]
        if hand_pred_state is not None:
            print(f"Hand Predicted State Shape: {hand_pred_state.shape}")  # Not Predicted
        if pose_pred_state is not None:
            print(f"Pose Predicted State Shape: {pose_pred_state.shape}")  # Not Predicted
        if robot_pred_state is not None:
            print(f"Robot Predicted State Shape: {robot_pred_state.shape}")  # Not Predicted
        
        print("\nForward pass successful!")
        
        # 可视化一些预测结果
        # if image_pred is not None:
        #     # 重建一张图像进行可视化
        #     img_idx = 0  # 第一个批次的第一个序列
        #     patch_size = 16
        #     num_patches = int(224 / patch_size)
            
        #     # 将预测的patch重组为图像
        #     pred_img = image_pred[img_idx, 0].reshape(num_patches, num_patches, patch_size, patch_size, 3)
        #     pred_img = pred_img.permute(0, 2, 1, 3, 4).reshape(224, 224, 3)
        #     pred_img = pred_img.cpu().numpy()
            
        #     # 归一化到[0, 1]范围
        #     pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())
            
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(pred_img)
        #     plt.title("Predicted Image")
        #     plt.axis('off')
        #     plt.savefig("predicted_image.png")
        #     print("Saved predicted image visualization to 'predicted_image.png'")
        
        if hand_pred_action is not None:
            # 打印一些预测的动作值
            print("\nSample Hand Action Predictions:")
            print(hand_pred_action[0, 0])  # 第一个批次的第一个序列的预测
        
        if pose_pred_action is not None:
            # 打印一些预测的夹爪值
            print("\nSample Pose Action Predictions:")
            print(pose_pred_action[0, 0])  # 第一个批次的第一个序列的预测
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_my_seer_agent()
