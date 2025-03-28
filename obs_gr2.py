"""
This scripts demonstrates how to evaluate a pretrained ACT policy on a Fourier GR2 robot.
"""

from pathlib import Path

import imageio
import numpy as np
import torch
import sys
import functools
from PIL import Image
import clip
from huggingface_hub import snapshot_download

from torchvision import transforms as v2

from omegaconf import OmegaConf

from controller.gr2_player import GR2Player

from models.my_seer_model import SeerAgent
from models.normalize_utils import StateActionNormalizer


def preprocess_text_calvin(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text


if __name__ == "__main__":
    # Load the pretrained policy
    arm_mode = "bimanual"

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        device = torch.device("cpu")
        print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")

    # Initialize the GR2 player
    player = GR2Player(
                OmegaConf.load("controller/configs/gr2t5.yml")
            )
    # Move robot to the initial position
    player.reset_robot()

    seq_len = 7
    action_pred_steps = 3
    control_type = "position"
    model = SeerAgent(
        finetune_type="bottle",
        clip_device=device,
        vit_checkpoint_path="/home/gjt/lxy/Seer/ckpt/mae_pretrain_vit_base.pth",
        sequence_length=seq_len,
        num_resampler_query=6,
        num_obs_token_per_image=9,
        calvin_input_image_size=224,
        patch_size=16,
        action_pred_steps=action_pred_steps,
        obs_pred=True,
        atten_only_obs=False,
        attn_robot_proprio_state=False,
        atten_goal=0,
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        transformer_layers=24,
        hidden_dim=384,
        transformer_heads=12,
        phase="evaluate",
        control_type=control_type,
    )
    
    checkpoint = torch.load("/home/gjt/lxy/Seer/checkpoints/pick_bottle_scratch_position/best_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], False)
    model.to(device)
    model._init_model_type()
    model.eval()

    _, image_processor = clip.load("ViT-B/32", device=device)
    preprocess_text_fn = functools.partial(
        preprocess_text_calvin, tokenizer=clip
    )
    lang = ["Pick up the bottle"]
    text_tensors = preprocess_text_fn(lang)
    normalizer = StateActionNormalizer("/home/gjt/lxy/Seer/dataset_statistics.npz")

    step = 0
    done = False
    hand_state_list = []
    robot_state_list = []
    left_image_list = []
    right_image_list = []
    while not done:
        state, left_image, right_image = player.observe(mode=arm_mode)  # (41,), (240, 320, 3), (240, 320, 3)
        # print(state.shape, left_image.shape, right_image.shape)
        state = torch.from_numpy(state).to(torch.float32)
        norm_hand_state = normalizer.normalize_hand_state(state[:12])  # (12,)
        norm_robot_state = normalizer.normalize_robot_state(state[12:])[-14:]  # (14,)
        norm_robot_state = torch.cat([torch.zeros(15), norm_robot_state], dim=0)  # (29,)
        left_image = image_processor(Image.fromarray(left_image))  # (3, 224, 224)
        right_image = image_processor(Image.fromarray(right_image))  # (3, 224, 224)
        print(left_image.shape)

        hand_state_list.append(norm_hand_state)
        robot_state_list.append(norm_robot_state)
        left_image_list.append(left_image)
        right_image_list.append(right_image)

        if len(hand_state_list) >= seq_len:
            window_hand_state = torch.stack(hand_state_list, dim=0).unsqueeze(0)  # (1, seq_len, 12)
            window_robot_state = torch.stack(robot_state_list, dim=0).unsqueeze(0)  # (1, seq_len, 29)
            window_left_image = torch.stack(left_image_list, dim=0).unsqueeze(0)  # (1, seq_len, 3, 224, 224)
            window_right_image = torch.stack(right_image_list, dim=0).unsqueeze(0)  # (1, seq_len, 3, 224, 224)
            window_text_tensors = text_tensors.unsqueeze(1).repeat(1, seq_len, 1)  # (1, seq_len, 77)
            print(window_hand_state.shape, window_robot_state.shape, window_left_image.shape, window_right_image.shape, window_text_tensors.shape)
            
            hand_pred_action, pose_pred_action, robot_pred_action, image_pred, \
                hand_pred_state, pose_pred_state, robot_pred_state, loss_action = model(
                    image_left=window_left_image.to(device),
                    image_right=window_right_image.to(device),
                    hand_state=window_hand_state.to(device),
                    pose_state=None,
                    robot_state=window_robot_state.to(device),
                    text_token=window_text_tensors.to(device),
                    action=None,
                )
            
            if hand_pred_action is not None:
                hand_pred_action = normalizer.denormalize_hand_action(hand_pred_action)
                if action_pred_steps > 1:
                    hand_pred_action = hand_pred_action[:, :, 0, :]  # the first action step
                hand_pred_action = hand_pred_action[:, -1, :]  # the predicted next action
                # print("hand_pred_action shape: ", hand_pred_action.shape)
                # print("hand_pred_action: ", hand_pred_action)
            if pose_pred_action is not None:
                pose_pred_action = normalizer.denormalize_pose_action(pose_pred_action)
                if action_pred_steps > 1:
                    pose_pred_action = pose_pred_action[:, :, -1, :]  # the last action step
                pose_pred_action = pose_pred_action[:, -1, :]  # the predicted next action
                # print("hand_pred_action shape: ", pose_pred_action.shape)
                # print("pose_pred_action: ", pose_pred_action)
            if robot_pred_action is not None:
                robot_pred_action = normalizer.denormalize_robot_action(robot_pred_action)
                if action_pred_steps > 1:
                    robot_pred_action = robot_pred_action[:, :, -1, :]  # the last action step
                robot_pred_action = robot_pred_action[:, -1, :]  # the predicted next action
                # print("hand_pred_action shape: ", robot_pred_action.shape)
                # print("robot_pred_action: ", robot_pred_action)

            if control_type == "position":
                pred_action = torch.cat([hand_pred_action, robot_pred_action[:, 15:]], dim=1).detach().cpu().numpy()
                print("pred_action shape: ", pred_action.shape)
                print("pred_action: ", pred_action)

            hand_state_list.pop(0)
            robot_state_list.pop(0)
            left_image_list.pop(0)
            right_image_list.pop(0)

            break

        # import time
        # time.sleep(1.)
        step += 1
