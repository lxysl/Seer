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
import time
import clip
import h5py
from huggingface_hub import snapshot_download
from torchvision import transforms as v2
from omegaconf import OmegaConf
from collections import OrderedDict

from controller.gr2_player import GR2Player

from models.my_seer_model import SeerAgent
from models.normalize_utils_fix import StateActionNormalizer


np.set_printoptions(suppress=True, precision=11)


def preprocess_text_calvin(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text

def load_hdf5(path, offset=10):  # offset 10ms
    input_file = path
    file = h5py.File(input_file, "r")
    print(f"Total hdf5_frames: {file['timestamp'].shape[0]}")

    timestamps = np.array(file["timestamp"][:], dtype=np.int64) - offset
    states_hands = np.array(file["state/hand"][:])
    states_pose = np.array(file["state/pose"][:])
    states_robot = np.array(file["state/robot"][:])

    actions_hands = np.array(file["action/hand"][:])
    actions_pose = np.array(file["action/pose"][:])
    actions_robot = np.array(file["action/robot"][:])

    states = np.concatenate([states_hands, states_pose, states_robot], axis=1)
    actions = np.concatenate([actions_hands, actions_pose, actions_robot], axis=1)
    return timestamps, states, actions


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
        phase="finetune",
        control_type=control_type,
    )

    # load model weights
    checkpoint = torch.load("/home/gjt/lxy/Seer/checkpoints/pick_bottle_scratch_position_fix_wo_norm_resize/best_model.pth", map_location="cpu")
    model.to(device)
    model._init_model_type()

    state_dict = checkpoint["model_state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k   # 去掉 "module."
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, False)

    model.eval()

    _, image_processor = clip.load("ViT-B/32", device=device)
    preprocess_text_fn = functools.partial(
        preprocess_text_calvin, tokenizer=clip
    )
    lang = ["Pick up the bottle"]
    text_tensors = preprocess_text_fn(lang)

    # load normalizer
    normalizer = None
    # normalizer = StateActionNormalizer("/home/gjt/lxy/Seer/dataset_statistics_fix.npz")

    # hdf5_timestamps, states, actions = load_hdf5("/home/gjt/yanchi/teleoperation_dds/data/recordings/fix/training/2025-04-10_20-26-15/episode_000000000.hdf5")
    # timestamp, state, gt_action = hdf5_timestamps[0], states[0], actions[0]
    # player.step(np.concatenate([gt_action[-14:], gt_action[:12]], axis=0), mode=arm_mode, time=2.0)

    step = 0
    done = False
    save = True
    all_left_image_list = []
    all_right_image_list = []
    all_text_list = []
    all_collect_state_list = []
    all_collect_norm_state_list = []
    model_hand_pred_actions = []
    model_robot_pred_actions = []
    all_norm_window_hand_state_list = []
    all_norm_window_robot_state_list = []
    all_pred_action_list = []
    all_denorm_pred_action_list = []
    hand_state_list = []
    robot_state_list = []
    left_image_list = []
    right_image_list = []
    while not done:
        print(step)
        state, left_image, right_image = player.observe(mode=arm_mode)  # (41,), (240, 320, 3), (240, 320, 3)

        # resize images to (224, 224)
        left_image = Image.fromarray(left_image).resize((224, 224))  # (3, 224, 224)
        right_image = Image.fromarray(right_image).resize((224, 224))  # (3, 224, 224)

        # print(state.shape, left_image.shape, right_image.shape)
        # if step == 0:
        #     # save left and right image
        #     imageio.imwrite("./left_image.png", left_image)
        #     imageio.imwrite("./right_image.png", right_image)
        # print("save")

        state = torch.from_numpy(state).to(torch.float32)
        if normalizer:
            norm_hand_state = normalizer.normalize_hand_state(state[:12])  # (12,)
            norm_robot_state = normalizer.normalize_robot_state(state[-29:])  # (29,)
        else:
            hand_state = state[:12]
            robot_state = state[-29:]
        left_image = image_processor(left_image)
        right_image = image_processor(right_image)

        all_collect_state_list.append(state)
        if normalizer:
            all_collect_norm_state_list.append(np.concatenate([norm_hand_state, norm_robot_state], axis=0))
            hand_state_list.append(norm_hand_state)
            robot_state_list.append(norm_robot_state)
        else:
            hand_state_list.append(hand_state)
            robot_state_list.append(robot_state)
        left_image_list.append(left_image)
        right_image_list.append(right_image)

        if len(hand_state_list) >= seq_len:
            window_hand_state = torch.stack(hand_state_list, dim=0).unsqueeze(0)  # (1, seq_len, 12)
            window_robot_state = torch.stack(robot_state_list, dim=0).unsqueeze(0)  # (1, seq_len, 29)
            window_left_image = torch.stack(left_image_list, dim=0).unsqueeze(0)  # (1, seq_len, 3, 224, 224)
            window_right_image = torch.stack(right_image_list, dim=0).unsqueeze(0)  # (1, seq_len, 3, 224, 224)
            window_text_tensors = text_tensors.unsqueeze(1).repeat(1, seq_len, 1)  # (1, seq_len, 77)
            # print(window_hand_state.shape, window_robot_state.shape, window_left_image.shape, window_right_image.shape, window_text_tensors.shape)

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

            if step <= 100:
                all_left_image_list.append(window_left_image.detach().cpu().numpy())
                all_right_image_list.append(window_right_image.detach().cpu().numpy())
                all_text_list.append(window_text_tensors.detach().cpu().numpy())
            model_hand_pred_actions.append(hand_pred_action.detach().cpu().numpy())
            model_robot_pred_actions.append(robot_pred_action.detach().cpu().numpy())
            all_norm_window_hand_state_list.append(window_hand_state.detach().cpu().numpy())
            all_norm_window_robot_state_list.append(window_robot_state.detach().cpu().numpy())

            if hand_pred_action is not None:
                denorm_hand_pred_action = normalizer.denormalize_hand_action(hand_pred_action) if normalizer else hand_pred_action
                if action_pred_steps > 1:
                    hand_pred_action = hand_pred_action[:, :, 0, :]
                    denorm_hand_pred_action = denorm_hand_pred_action[:, :, 0, :]  # the first action step
                hand_pred_action = hand_pred_action[0, -1, :]
                denorm_hand_pred_action = denorm_hand_pred_action[0, -1, :]  # the predicted next action in the first batch
                # print("hand_pred_action shape: ", hand_pred_action.shape)
                # print("hand_pred_action: ", hand_pred_action)
            if pose_pred_action is not None:
                denorm_pose_pred_action = normalizer.denormalize_pose_action(pose_pred_action) if normalizer else pose_pred_action
                if action_pred_steps > 1:
                    pose_pred_action = pose_pred_action[:, :, 0, :]
                    denorm_pose_pred_action = denorm_pose_pred_action[:, :, 0, :]  # the first action step
                pose_pred_action = pose_pred_action[0, -1, :]
                denorm_pose_pred_action = denorm_pose_pred_action[0, -1, :]  # the predicted next action in the first batch
                # print("hand_pred_action shape: ", pose_pred_action.shape)
                # print("pose_pred_action: ", pose_pred_action)
            if robot_pred_action is not None:
                denorm_robot_pred_action = normalizer.denormalize_robot_action(robot_pred_action) if normalizer else robot_pred_action
                if action_pred_steps > 1:
                    robot_pred_action = robot_pred_action[:, :, 0, :]
                    denorm_robot_pred_action = denorm_robot_pred_action[:, :, 0, :]  # the first action step
                robot_pred_action = robot_pred_action[0, -1, :]
                denorm_robot_pred_action = denorm_robot_pred_action[0, -1, :]  # the predicted next action in the first batch
                # print("hand_pred_action shape: ", robot_pred_action.shape)
                # print("robot_pred_action: ", robot_pred_action)

            if control_type == "position":
                pred_action = torch.cat([robot_pred_action[15:], hand_pred_action], dim=0).detach().cpu().numpy()
                denorm_pred_action = torch.cat([denorm_robot_pred_action[15:], denorm_hand_pred_action], dim=0).detach().cpu().numpy()

            all_pred_action_list.append(pred_action)
            all_denorm_pred_action_list.append(denorm_pred_action)

            # print("denorm_pred_action shape: ", denorm_pred_action.shape)
            # print("denorm_pred_action: ", denorm_pred_action)
            # input("")
            if step < seq_len + 2:
                # Avoid sudden movements in the first few steps
                player.step(denorm_pred_action, mode=arm_mode, time=0.5)
            else:
                player.step(denorm_pred_action, mode=arm_mode, time=0.0)

            hand_state_list.pop(0)
            robot_state_list.pop(0)
            left_image_list.pop(0)
            right_image_list.pop(0)

        # time.sleep(0.015)
        step += 1

        # if step == 200:
        #     done = True

    # save to hdf5
    with h5py.File("gr2_data-2.hdf5", "w") as f:
        # f.create_dataset("left_image", data=np.array(all_left_image_list))
        # f.create_dataset("right_image", data=np.array(all_right_image_list))
        # f.create_dataset("text", data=np.array(all_text_list))
        f.create_dataset("collect_state", data=np.array(all_collect_state_list))
        f.create_dataset("collect_norm_state", data=np.array(all_collect_norm_state_list))
        # f.create_dataset("hand_state", data=np.array(all_hand_state_list))
        # f.create_dataset("norm_hand_state", data=np.array(all_norm_hand_state_list))
        f.create_dataset("norm_window_hand_state", data=np.array(all_norm_window_hand_state_list))
        # f.create_dataset("robot_state", data=np.array(all_robot_state_list))
        # f.create_dataset("norm_robot_state", data=np.array(all_norm_robot_state_list))
        f.create_dataset("norm_window_robot_state", data=np.array(all_norm_window_robot_state_list))
        f.create_dataset("pred_action", data=np.array(all_pred_action_list))
        f.create_dataset("denorm_pred_action", data=np.array(all_denorm_pred_action_list))
        # f.create_dataset("gt_action", data=np.array(all_gt_action_list))
        # f.create_dataset("video_timestamps", data=video_timestamps)
        # f.create_dataset("hdf5_timestamps", data=hdf5_timestamps)
        f.create_dataset("model_hand_pred_actions", data=model_hand_pred_actions)
        f.create_dataset("model_robot_pred_actions", data=np.array(model_robot_pred_actions))
