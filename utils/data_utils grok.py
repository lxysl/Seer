import logging

import numpy as np
try:
    from pytorch3d.transforms import (
        euler_angles_to_matrix,
        matrix_to_euler_angles,
        matrix_to_quaternion,
        quaternion_to_matrix,
    )
except:
    print('no pytorch3d')
import torch
from torch.cuda.amp import autocast
logger = logging.getLogger(__name__)
import functools
import math
import io
import os
import random
import re
import pickle
import glob
import cv2
from multiprocessing import Value
from functools import partial
import json
from itertools import chain
from dataclasses import dataclass
import numpy as np
from PIL import Image
import copy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Dataset
from torch.utils.data.distributed import DistributedSampler
try:
    from petrel_client.client import Client
except:
    pass 
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import bisect
from itertools import accumulate
import copy
from typing import List
from torchvision import transforms as torchtransforms
from PIL import Image
import clip
from pdb import set_trace
import h5py
from scipy.spatial.transform import Rotation as R
import time
import pyzed.sl as sl

Image.MAX_IMAGE_PIXELS = 1000000000
MAX_NUM_TOKENS = 256
MAX_NUM_IMAGES = 5
TINY_IMAGE_SIZE_THRESHOLD = 1
N_CHANNELS = 3
INTERLEAVED_IMAGE_SIZE = 224

_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

MIN_KB = 10
MAX_NUM_IMAGES = 5
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple, Callable, Union

obs_config = DictConfig(
    {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
)

prop_state = DictConfig(
    {
        "n_scene_obs": 24,
        "n_state_obs": 15,
        "keep_indices": [[0, 15]],
        "robot_orientation_idx": [3, 6],
        "normalize": True,
        "normalize_robot_orientation": True,
    }
)

def _6d_to_pose(pose6d, degrees=False):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose

def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] =  R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d

def get_state_info_dict(episode: Dict[str, np.ndarray]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create a dictionary with raw state observations for environment resets.

    Args:
        episode: Sequence dictionary.

    Returns:
         Info dict of full robot and scene state (for env resets).
    """
    return {
        "state_info": {
            "robot_obs": torch.from_numpy(episode["robot_obs"]),
            "scene_obs": torch.from_numpy(episode["scene_obs"]),
        }
    }

def process_state(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    proprio_state: DictConfig,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    state_obs_keys = observation_space["state_obs"]
    state_obs_list_normalized = []
    state_obs_list_unnormalized = []
    for state_ob in state_obs_keys:
        if window_size == 0 and seq_idx == 0:  # single file loader
            state_tensor = torch.from_numpy(episode[state_ob]).float()
        else:  # episode loader
            state_tensor = torch.from_numpy(episode[state_ob][seq_idx : seq_idx + window_size]).float()
        # expand dims for single environment obs
        if len(state_tensor.shape) != 2:
            state_tensor = state_tensor.unsqueeze(0)
        # shape: (BxN_state_obs)
        assert len(state_tensor.shape) == 2
        if state_ob in transforms:
            state_tensor_normalized = transforms[state_ob](state_tensor)
            state_obs_list_normalized.append(state_tensor_normalized)
        else:
            state_obs_list_normalized.append(state_tensor)
        state_obs_list_unnormalized.append(state_tensor)
    seq_state_obs = torch.cat(state_obs_list_normalized, dim=1)
    seq_state_obs_unnormalized = torch.cat(state_obs_list_unnormalized, dim=1)

    if not proprio_state.normalize_robot_orientation and "robot_orientation_idx" in proprio_state:
        seq_state_obs[:, slice(*proprio_state.robot_orientation_idx)] = seq_state_obs_unnormalized[
            :, slice(*proprio_state.robot_orientation_idx)
        ]

    if not proprio_state.normalize:
        seq_state_obs = seq_state_obs_unnormalized

    # slice the specified parts of the proprioception state
    state_obs_sliced = []
    for slice_ids in proprio_state.keep_indices:
        seq_state_obs_ = seq_state_obs[:, slice(*slice_ids)]
        state_obs_sliced.append(seq_state_obs_)
    seq_state_obs = torch.cat(state_obs_sliced, dim=1)

    return {"robot_obs": seq_state_obs}

def preprocess_image(sample, image_processor):
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    # apply random horizontal flip and color jitter
    return image

def preprocess_text_calvin(sample, tokenizer):
    text = tokenizer.tokenize(sample, truncate=True)
    return text

def process_rgb(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    rgb_obs_keys = observation_space["rgb_obs"]

    seq_rgb_obs_dict = {}
    for _, rgb_obs_key in enumerate(rgb_obs_keys):
        rgb_obs = episode[rgb_obs_key]
        # expand dims for single environment obs
        if len(rgb_obs.shape) != 4:
            rgb_obs = np.expand_dims(rgb_obs, axis=0)
        assert len(rgb_obs.shape) == 4
        if window_size == 0 and seq_idx == 0:  # single file loader
            # To Square image
            seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte().permute(0, 3, 1, 2)
        else:  # episode loader
            seq_rgb_obs_ = torch.from_numpy(rgb_obs[seq_idx : seq_idx + window_size]).byte().permute(0, 3, 1, 2)
        # we might have different transformations for the different cameras
        if rgb_obs_key in transforms:
            seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
        seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
    # shape: N_rgb_obs x (BxCxHxW)
    return {"rgb_obs": seq_rgb_obs_dict}

def process_depth(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # expand dims for single environment obs
    def exp_dim(depth_img):
        if len(depth_img.shape) != 3:
            depth_img = np.expand_dims(depth_img, axis=0)
        return depth_img

    depth_obs_keys = observation_space["depth_obs"]
    seq_depth_obs_dict = {}
    for _, depth_obs_key in enumerate(depth_obs_keys):
        depth_ob = exp_dim(episode[depth_obs_key])
        assert len(depth_ob.shape) == 3
        if window_size == 0 and seq_idx == 0:  # single file loader
            depth_ob_ = torch.from_numpy(depth_ob).float()
        else:  # episode loader
            depth_ob_ = torch.from_numpy(depth_ob[seq_idx : seq_idx + window_size]).float()
        # we might have different transformations for the different cameras
        if depth_obs_key in transforms:
            depth_ob_ = transforms[depth_obs_key](depth_ob_)
        seq_depth_obs_dict[depth_obs_key] = depth_ob_
    # shape: N_depth_obs x(BxHxW)
    return {"depth_obs": seq_depth_obs_dict}

def process_actions(
    episode: Dict[str, np.ndarray],
    observation_space: DictConfig,
    transforms: Dict,
    seq_idx: int = 0,
    window_size: int = 0,
) -> Dict[str, torch.Tensor]:
    # shape: (N_actions)
    action_keys = observation_space["actions"]
    if len(action_keys) != 1:
        raise NotImplementedError
    action_key = action_keys[0]
    if window_size == 0 and seq_idx == 0:  # single file loader
        action = episode[action_key]
        if "actions" in transforms:
            action = transforms["actions"]((action, episode["robot_obs"]))
        seq_acts = torch.from_numpy(action).float()
    else:  # episode loader
        seq_acts = torch.from_numpy(episode[action_key][seq_idx : seq_idx + window_size]).float()
    return {"actions": seq_acts}

def process_language(episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool) -> Dict[str, torch.Tensor]:
    seq_lang = {"lang": torch.empty(0)}
    if with_lang:
        lang = torch.from_numpy(episode["language"]).float()
        if "language" in transforms:
            lang = transforms["language"](lang)
        seq_lang["lang"] = lang
    return seq_lang

def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    """
    Check naming pattern of dataset files.

    Args:
        dataset_dir: Path to dataset.
        save_format: File format (CALVIN default is npz).

    Returns:
        naming_pattern: 'file_0000001.npz' -> ('file_', '.npz')
        n_digits: Zero padding of file enumeration.
    """
    it = os.scandir(dataset_dir)
    while True:
        filename = Path(next(it))
        if save_format in filename.suffix:
            break
    aux_naming_pattern = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / aux_naming_pattern[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    assert len(naming_pattern) == 2
    assert n_digits > 0
    return naming_pattern, n_digits

def load_partial_traj_data():
    with open('utils/partial_task_data.json', 'r') as f:
        data = json.load(f)
    return data

def subtract_ranges(rangeA, rangeB):
    def subtract_single_range(a, b):
        result = []
        a_start, a_end = a
        b_start, b_end = b

        if b_start > a_end or b_end < a_start:
            # No overlap
            return [a]
        if b_start > a_start:
            result.append((a_start, min(a_end, b_start - 1)))
        if b_end < a_end:
            result.append((max(a_start, b_end + 1), a_end))

        return [r for r in result if r[0] <= r[1]]

    result = rangeA
    for b in rangeB:
        new_result = []
        for a in result:
            new_result.extend(subtract_single_range(a, b))
        result = new_result

    return result

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

class BaseCalvinDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        *args: Any,
        proprio_state: DictConfig = prop_state,
        lang_folder: str = "lang_annotations",
        num_workers: int = 0,
        key: str = "lang",
        obs_space: DictConfig = obs_config,
        transforms: Dict = {},
        batch_size: int = 32,
        window_size: int = 16,
        min_window_size: int = 16,
        max_window_size: int = 16,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        rgb_pad=-1,
        gripper_pad=-1,
        traj_cons=False,
        text_aug=False,
        dif_ws=False,
        act_step=1,
        data_in_ceph=False,
        **kwargs: Any,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.except_lang = key == "except_lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
        self.act_step = act_step
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  
        self.aux_lang_loss_window = aux_lang_loss_window
        self.traj_cons = traj_cons
        self.data_in_ceph = data_in_ceph
        if self.data_in_ceph:
            self.conf_path = '~/petreloss.conf'
            self.client = Client(self.conf_path)
       
        with open('./utils/enrich_lang_annotations.json', 'r') as f:
            self.enrich_lang = json.load(f)
        self.text_aug = text_aug

        self.rgb_pad = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)

        if self.data_in_ceph:
            assert (
                "validation" in self.abs_datasets_dir
                or "training" in self.abs_datasets_dir
            )
            self.validation = "validation" in self.abs_datasets_dir
        else:
            assert (
                "validation" in self.abs_datasets_dir.as_posix()
                or "training" in self.abs_datasets_dir.as_posix()
            )
            self.validation = "validation" in self.abs_datasets_dir.as_posix()
        print(f"loading dataset at {self.abs_datasets_dir}")
        print("finished loading dataset")

    def process_rgb(
        self,
        episode: Dict[str, np.ndarray],
        observation_space: DictConfig,
        transforms: Dict,
        seq_idx: int = 0,
        window_size: int = 0,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        rgb_obs_keys = observation_space["rgb_obs"]
        seq_rgb_obs_dict = {}
        for _, rgb_obs_key in enumerate(rgb_obs_keys):
            rgb_obs = episode[rgb_obs_key]
            # expand dims for single environment obs
            if len(rgb_obs.shape) != 4:
                rgb_obs = np.expand_dims(rgb_obs, axis=0)
            assert len(rgb_obs.shape) == 4
            if window_size == 0 and seq_idx == 0:  # single file loader
                # To Square image
                seq_rgb_obs_ = torch.from_numpy(rgb_obs).byte()
            else:  # episode loader
                seq_rgb_obs_ = torch.from_numpy(
                    rgb_obs[seq_idx : seq_idx + window_size]
                ).byte()
            
            if rgb_obs_key in transforms:
                seq_rgb_obs_ = transforms[rgb_obs_key](seq_rgb_obs_)
            seq_rgb_obs_dict[rgb_obs_key] = seq_rgb_obs_
        # shape: N_rgb_obs x (BxHxWxC)
        return {"rgb_obs": seq_rgb_obs_dict}

    def process_language(
        self, episode: Dict[str, np.ndarray], transforms: Dict, with_lang: bool
    ):
        return {"lang": episode["language"]} if with_lang else {"lang": 'no lang'}

    def __getitem__(self, idx: Union[int, Tuple[int, int]], fixed_seed=False) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                print(
                    f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}"
                )
                raise ValueError
        else:
            idx, window_size = idx
        
        head = False
        sequence = self._get_sequences(idx, window_size, head=head)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size, head=head)
        
        import copy
        new_list = []
        np_rgb = copy.deepcopy(sequence["rgb_obs"]["rgb_static"].numpy())
        for i in range(np_rgb.shape[0]):
            new_list.append(Image.fromarray(np_rgb[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_static"] = new_list
        new_list = []
        np_gripper = copy.deepcopy(sequence["rgb_obs"]["rgb_gripper"].numpy())
        for i in range(np_gripper.shape[0]):
            new_list.append(Image.fromarray(np_gripper[i, :, :, :].astype(np.uint8)))
        sequence["rgb_obs"]["rgb_gripper"] = new_list
        return sequence

    def _get_sequences(self, idx: int, window_size: int, head: bool=False) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)
        seq_state_obs = process_state(
            episode, self.observation_space, self.transforms, self.proprio_state
        )
        seq_rgb_obs = self.process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = self.process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_depth_obs,
            **seq_acts,
            **info,
            **seq_lang,
        }  
        seq_dict["idx"] = idx  

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif (
            self.episode_lookup[idx + window_diff]
            != self.episode_lookup[idx] + window_diff
        ):
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(
                self.max_window_size, (self.min_window_size + steps_to_next_episode - 1)
            )
        else:
            max_window = self.max_window_size

        return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int, head: bool=False) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update(
            {
                "rgb_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["rgb_obs"].items()
                }
            }
        )
        seq.update(
            {
                "depth_obs": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["depth_obs"].items()
                }
            }
        )

        if not self.relative_actions:
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                # repeat action for world coordinates action space
                seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size, head)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            if head:
                seq_acts = self._pad_with_zeros(seq["actions"], pad_size, head)
            else:
                seq_acts = torch.cat(
                    [
                        self._pad_with_zeros(seq["actions"][..., :-1], pad_size, head),
                        self._pad_with_repetition(seq["actions"][..., -1:], pad_size, head),
                    ],
                    dim=-1,
                )
            seq.update({"actions": seq_acts})
        seq.update(
            {
                "state_info": {
                    k: self._pad_with_repetition(v, pad_size, head)
                    for k, v in seq["state_info"].items()
                }
            }
        )
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        if head:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[0], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((last_repeated, input_tensor))
        else:
            last_repeated = torch.repeat_interleave(
                torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
            )
            padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int, head: bool = False) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        if head:
            padded = torch.vstack((zeros_repeated, input_tensor))
        else:
            padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None
    dataset: Dataset = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class DiskCalvinDataset(BaseCalvinDataset):
    """
    Dataset that loads episodes as individual files from disk.
    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        image_fn: Callable,
        text_fn: Callable,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        partial_data=False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.partial_data = partial_data
        if self.save_format == "pkl":
            self.load_file = self.load_pkl
        elif self.save_format == "npz":
            self.load_file = partial(self.load_npz, data_in_ceph=self.data_in_ceph)
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        if self.with_lang:
            (
                self.episode_lookup,
                self.lang_lookup,
                self.lang_ann,
                self.lang_task
            ) = self._build_file_indices_lang()
        elif self.except_lang:
            self.episode_lookup = self._build_file_indices_except_lang()
        else:
            self.episode_lookup = self._build_file_indices()

        if self.data_in_ceph:
            self.naming_pattern, self.n_digits = self.ceph_lookup_naming_pattern()
        else:
            self.naming_pattern, self.n_digits = lookup_naming_pattern(
                self.abs_datasets_dir, self.save_format
            )
    
    def ceph_lookup_naming_pattern(self):
        filenames = self.client.list(self.abs_datasets_dir)
        for filename in filenames:
            if self.save_format in filename:
                break
        filename = self.abs_datasets_dir + f"/{filename}"
        suffix = "." + self.save_format
        stem_suffix = filename.split('/')[-1]
        stem = stem_suffix.replace(suffix, "")
        aux_naming_pattern = re.split(r"\d+", stem)
        naming_pattern = (filename.replace(stem_suffix, aux_naming_pattern[0]), suffix)
        n_digits = len(re.findall(r"\d+", stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        if self.data_in_ceph:
            return f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        else:
            return Path(
                f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
            )

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.
        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx)
        ]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]]
            if self.text_aug:
                task = self.lang_task[self.lang_lookup[idx]]
                enrich_lang = random.choice(self.enrich_lang[task] + [episode["language"]])
                episode["language"] = enrich_lang
        return episode

    def _build_file_indices_lang(
        self, # abs_datasets_dir: Path
    ):
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        abs_datasets_dir = self.abs_datasets_dir
        episode_lookup = []

        try:
            if self.data_in_ceph:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir +f"/{self.lang_folder}/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/{self.lang_folder}/auto_lang_ann.npy", enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
                    allow_pickle=True,
                ).item()
        except Exception:
            if self.data_in_ceph:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir + "/auto_lang_ann.npy",
                )
                lang_data_bytes = self.client.get(abs_datasets_dir+f"/auto_lang_ann.npy", enable_cache=True)
                lang_data = io.BytesIO(lang_data_bytes)
                lang_data = np.load(lang_data, allow_pickle=True).item()
            else:
                print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / "auto_lang_ann.npy",
                )
                lang_data = np.load(
                    abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True
                ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        partial_st_ed_list = load_partial_traj_data()
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.partial_data:
                if [start_idx, end_idx] not in partial_st_ed_list:
                    continue
            if self.pretrain:
                start_idx = max(
                    start_idx,
                    end_idx + 1 - self.min_window_size - self.aux_lang_loss_window,
                )
            assert end_idx >= self.max_window_size
            cnt = 0
            
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def _build_file_indices_except_lang(self) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        abs_datasets_dir = self.abs_datasets_dir
        lang_data = np.load(
            abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy",
            allow_pickle=True,
        ).item()
        lang_ep_start_end_ids = lang_data["info"]["indx"]

        episode_lookup = []

        if self.data_in_ceph:
            lang_data_bytes = self.client.get(abs_datasets_dir+f"ep_start_end_ids.npy", enable_cache=True)
            lang_data = io.BytesIO(lang_data_bytes)
            ep_start_end_ids = np.load(lang_data)
        else:
            ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print(
            f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.'
        )
        ep_start_end_ids = np.load(abs_datasets_dir / "except_lang_idx" / "except_lang_idx.npy").tolist()

        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def collator(self, sample):
        action_tensors = torch.from_numpy(np.array([np.stack(s["actions"]) for s in sample]))
        state_tensors = torch.from_numpy(np.array([np.stack(s["robot_obs"]) for s in sample]))
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        stacked_language = [s["lang"] for s in sample]
        text_tensors = self.text_fn(stacked_language)
        
        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs*seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)
        
        if self.act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            robot_obs = torch.cat([robot_obs[..., :6], robot_obs[..., [-1]]], dim=-1)
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
        
        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs

    def load_pkl(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def load_npz(self, filename, data_in_ceph=False):
        if not data_in_ceph:
            return np.load(filename.as_posix())
        else:
            data_bytes = self.client.get(filename, enable_cache=True)
            data = io.BytesIO(data_bytes)
            try:
                data = np.load(data, allow_pickle=True)
            except:
                data = np.load(data)
            return data

def get_calvin_dataset(args, image_processor, tokenizer, epoch=0, floor=False, except_lang=False):
    dataset_path = args.calvin_dataset
    # ann is dict including language and info
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/training"
    else:
        datasets_dir = Path(dataset_path) / "training"
    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph,
        key='except_lang' if except_lang else 'lang',
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  #
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)

def get_calvin_val_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    dataset_path = args.calvin_dataset
    shared_epoch = SharedEpoch(epoch=epoch)
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    if args.data_in_ceph:
        datasets_dir = dataset_path + "/validation"
    else:
        datasets_dir = Path(dataset_path) / "validation"
    calvin_dataset = DiskCalvinDataset(
        datasets_dir=datasets_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        text_aug=args.text_aug,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        partial_data=args.partial_data,
        data_in_ceph=args.data_in_ceph
    )
    round_fn = math.floor if floor else math.ceil
    num_samples = len(calvin_dataset)
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    sampler = DistributedSampler(
        calvin_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        calvin_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        shuffle=False,
        persistent_workers=True,
        collate_fn=calvin_dataset.collator,
        drop_last=True
    )
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=calvin_dataset)


class PickBottleDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        image_fn,
        text_fn,
        window_size=16,
        rgb_pad=-1,
        gripper_pad=-1,
        act_step=1,
        min_window_size=16,
        max_window_size=16,
        dif_ws=False,
        traj_cons=False,
        text_aug=False,
        pad=True,
        validation=False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.window_size = window_size
        self.act_step = act_step
        self.pad = pad
        self.validation = validation
        self.rgb_pad = rgb_pad
        self.gripper_pad = gripper_pad
        self.traj_cons = traj_cons
        
        if not dif_ws:
            self.min_window_size = window_size + act_step - 1
            self.max_window_size = window_size + act_step - 1
        else:
            self.min_window_size = min_window_size
            self.max_window_size = max_window_size
            
        self.timestamp_folders = sorted(glob.glob(str(self.dataset_dir / "20*-*-*_*-*-*")))
        print(f"Found {len(self.timestamp_folders)} timestamp folders")
        
        self.episode_lookup, self.folder_mapping = self._build_file_indices()
        print(f"Total number of trajectory windows: {len(self.episode_lookup)}")
        
        self.video_cache = {}
        self.language_instruction = "Pick up the bottle"
        
    def _build_file_indices(self):
        episode_lookup = []
        folder_mapping = {}
        
        for folder_idx, folder_path in enumerate(self.timestamp_folders):
            folder_mapping[folder_idx] = folder_path
            h5_files = sorted(glob.glob(os.path.join(folder_path, "episode_*.hdf5")))
            
            for episode_file in h5_files:
                episode_idx = int(os.path.basename(episode_file).split("_")[1].split(".")[0])
                with h5py.File(episode_file, "r") as f:
                    episode_length = len(f["timestamp"][:])
                
                if episode_length > self.max_window_size:
                    for start_idx in range(0, episode_length - self.min_window_size + 1):
                        episode_lookup.append((folder_idx, episode_idx, start_idx))
        
        return episode_lookup, folder_mapping
    
    def _load_episode(self, idx):
        folder_idx, episode_idx, start_frame_idx = self.episode_lookup[idx]
        folder_path = self.folder_mapping[folder_idx]
        
        h5_filename = os.path.join(folder_path, f"episode_{episode_idx:09d}.hdf5")
        video_filename = os.path.join(folder_path, f"episode_{episode_idx:09d}_top.svo2")
        
        with h5py.File(h5_filename, "r") as f:
            window_size = min(self.max_window_size, len(f["timestamp"][:]) - start_frame_idx)
            
            # Load all action dimensions
            actions_hand = f["action/hand"][start_frame_idx:start_frame_idx + window_size]    # (window_size, 12)
            actions_pose = f["action/pose"][start_frame_idx:start_frame_idx + window_size]    # (window_size, 24)
            actions_robot = f["action/robot"][start_frame_idx:start_frame_idx + window_size]  # (window_size, 29)
            actions = np.concatenate([actions_hand, actions_pose, actions_robot], axis=1)     # (window_size, 65)
            
            # Load all state dimensions
            state_hand = f["state/hand"][start_frame_idx:start_frame_idx + window_size]       # (window_size, 12)
            state_pose = f["state/pose"][start_frame_idx:start_frame_idx + window_size]       # (window_size, 27)
            state_robot = f["state/robot"][start_frame_idx:start_frame_idx + window_size]     # (window_size, 29)
            state = np.concatenate([state_hand, state_pose, state_robot], axis=1)             # (window_size, 68)
            
            timestamps = f["timestamp"][start_frame_idx:start_frame_idx + window_size]
        
        frames = self._load_video_frames(video_filename, timestamps)
        
        return {
            "actions": actions,              # (window_size, 65)
            "robot_obs": state,              # (window_size, 68)
            "scene_obs": np.zeros((window_size, 24)),  # Placeholder
            "rgb_static": frames["left"],    # List of left frames
            "rgb_gripper": frames["right"],  # List of right frames
            "language": self.language_instruction
        }
    
    def _load_video_frames(self, video_filename, timestamps):
        """
        Load video frames (left and right) from SVO2 file based on timestamps using pyzed.
        Args:
            video_filename: Path to the SVO2 file
            timestamps: Array of timestamps to match
        Returns:
            Dictionary containing lists of left and right frames as numpy arrays:
            {
                "left": [left_frame_1, left_frame_2, ...],
                "right": [right_frame_1, right_frame_2, ...]
            }
        """
        if video_filename not in self.video_cache:
            zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.set_from_svo_file(video_filename)
            init_params.svo_real_time_mode = False
            
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise ValueError(f"Could not open SVO file: {video_filename}. Error: {err}")
            
            frame_count = zed.get_svo_number_of_frames()
            left_frames = []
            right_frames = []
            video_timestamps = []
            
            left_image = sl.Mat()
            right_image = sl.Mat()
            
            for _ in range(frame_count):
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(left_image, sl.VIEW.LEFT)
                    zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                    left_frames.append(left_image.get_data())
                    right_frames.append(right_image.get_data())
                    video_timestamps.append(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds() / 1000.0)
            
            zed.close()
            self.video_cache[video_filename] = {
                "left": np.array(left_frames),
                "right": np.array(right_frames),
                "timestamps": np.array(video_timestamps)
            }
        
        cached_data = self.video_cache[video_filename]
        all_left_frames = cached_data["left"]
        all_right_frames = cached_data["right"]
        video_timestamps = cached_data["timestamps"]
        
        selected_left_frames = []
        selected_right_frames = []
        for ts in timestamps:
            closest_idx = np.argmin(np.abs(video_timestamps - ts))
            # Convert from BGR to RGB
            selected_left_frames.append(cv2.cvtColor(all_left_frames[closest_idx], cv2.COLOR_BGR2RGB))
            selected_right_frames.append(cv2.cvtColor(all_right_frames[closest_idx], cv2.COLOR_BGR2RGB))
        
        return {
            "left": selected_left_frames,
            "right": selected_right_frames
        }
    
    def __len__(self):
        return len(self.episode_lookup)
    
    def __getitem__(self, idx):
        episode = self._load_episode(idx)
        
        sequence = {
            "robot_obs": torch.tensor(episode["robot_obs"], dtype=torch.float32),
            "rgb_obs": {
                "rgb_static": [Image.fromarray(frame) for frame in episode["rgb_static"]],
                "rgb_gripper": [Image.fromarray(frame) for frame in episode["rgb_gripper"]]
            },
            "depth_obs": {},
            "actions": torch.tensor(episode["actions"], dtype=torch.float32),
            "state_info": {
                "robot_obs": torch.tensor(episode["robot_obs"], dtype=torch.float32),
                "scene_obs": torch.tensor(episode["scene_obs"], dtype=torch.float32)
            },
            "lang": episode["language"],
            "idx": idx
        }
        
        if self.pad and len(episode["actions"]) < self.max_window_size:
            pad_size = self.max_window_size - len(episode["actions"])
            sequence = self._pad_sequence(sequence, pad_size)
        
        return sequence
    
    def _pad_sequence(self, seq, pad_size):
        seq["robot_obs"] = self._pad_with_repetition(seq["robot_obs"], pad_size)
        for key in seq["rgb_obs"]:
            last_img = seq["rgb_obs"][key][-1]
            seq["rgb_obs"][key].extend([last_img] * pad_size)
        
        # Pad actions with zeros (assuming relative actions, adjust if absolute)
        actions = seq["actions"]
        pad_actions = torch.zeros((pad_size, actions.shape[1]), dtype=actions.dtype)
        seq["actions"] = torch.cat([actions, pad_actions], dim=0)
        
        for key in seq["state_info"]:
            seq["state_info"][key] = self._pad_with_repetition(seq["state_info"][key], pad_size)
        
        return seq
    
    @staticmethod
    def _pad_with_repetition(input_tensor, pad_size):
        last_repeated = torch.repeat_interleave(
            torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
        )
        return torch.vstack((input_tensor, last_repeated))
    
    def collator(self, sample):
        action_tensors = torch.stack([s["actions"] for s in sample])    # (batch, window_size, 65)
        state_tensors = torch.stack([s["robot_obs"] for s in sample])  # (batch, window_size, 68)
        
        image_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_static"]) for s in sample])
        gripper_tensors = torch.stack([self.image_fn(s["rgb_obs"]["rgb_gripper"]) for s in sample])
        
        stacked_language = [s["lang"] for s in sample]
        text_tensors = self.text_fn(stacked_language)
        
        # Handle augmentations (unchanged)
        if self.rgb_pad != -1:
            bs, seq_len = image_tensors.shape[:2]
            if self.traj_cons:
                image_tensors = self.rgb_shift.forward_traj(image_tensors)
            else:
                image_tensors = image_tensors.view(bs * seq_len, *image_tensors.shape[2:])
                image_tensors = self.rgb_shift(image_tensors)
                image_tensors = image_tensors.view(bs, seq_len, *image_tensors.shape[1:])
        
        if self.gripper_pad != -1:
            bs, seq_len = gripper_tensors.shape[:2]
            if self.traj_cons:
                gripper_tensors = self.gripper_shift.forward_traj(gripper_tensors)
            else:
                gripper_tensors = gripper_tensors.view(bs * seq_len, *gripper_tensors.shape[2:])
                gripper_tensors = self.gripper_shift(gripper_tensors)
                gripper_tensors = gripper_tensors.view(bs, seq_len, *gripper_tensors.shape[1:])
        
        robot_obs = torch.zeros(1)
        
        if self.act_step != 1:
            actions = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, action_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    actions[b, ix] = action_tensors[b, ix:ix+self.act_step]
            
            robot_obs = torch.zeros((action_tensors.shape[0], self.window_size, self.act_step, state_tensors.shape[-1]))
            for b in range(action_tensors.shape[0]):
                for ix in range(self.window_size):
                    robot_obs[b, ix] = state_tensors[b, ix:ix+self.act_step]
            
            action_tensors = actions
            image_tensors = image_tensors[:, :-(self.act_step-1)]
            gripper_tensors = gripper_tensors[:, :-(self.act_step-1)]
            state_tensors = state_tensors[:, :-(self.act_step-1)]
        
        return image_tensors, text_tensors, action_tensors, gripper_tensors, state_tensors, robot_obs

def get_pick_bottle_dataset(args, image_processor, tokenizer, epoch=0, validation=False):
    """
    Create a dataloader for the pick_bottle dataset.
    
    Args:
        args: Arguments containing dataset configuration
        image_processor: Function to process images
        tokenizer: Function to process text
        epoch: Current epoch
        validation: Whether to load validation data
        
    Returns:
        DataInfo object containing dataloader and metadata
    """
    shared_epoch = SharedEpoch(epoch=epoch)
    
    # Create preprocessing functions
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(
        preprocess_text_calvin, tokenizer=tokenizer
    )
    
    # Determine dataset directory
    dataset_dir = args.calvin_dataset
    if validation:
        dataset_dir = os.path.join(dataset_dir, "validation")
    else:
        dataset_dir = os.path.join(dataset_dir, "training")
    
    # Create dataset
    dataset = PickBottleDataset(
        dataset_dir=dataset_dir,
        image_fn=preprocess_image_fn,
        text_fn=preprocess_text_fn,
        window_size=args.window_size,
        rgb_pad=args.rgb_pad,
        gripper_pad=args.gripper_pad,
        traj_cons=args.traj_cons,
        dif_ws=args.dif_ws,
        min_window_size=args.min_window_size,
        max_window_size=args.max_window_size,
        act_step=args.multi_step_action,
        validation=validation
    )
    
    # Calculate batching details
    round_fn = math.floor if "floor" in args else math.ceil
    num_samples = len(dataset)
    world_size = args.world_size if "world_size" in args else 1
    global_batch_size = args.batch_size * world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    
    # Create sampler for distributed training
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=args.rank if "rank" in args else 0,
        shuffle=not validation,
        seed=args.seed,
        drop_last=True,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=num_workers,
        prefetch_factor=3,
        sampler=sampler,
        persistent_workers=True,
        collate_fn=dataset.collator,
        drop_last=True
    )
    
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, sampler=sampler, dataset=dataset)


if __name__ == "__main__":
    from arguments_utils import get_parser
    from tqdm import tqdm
    parser = get_parser()
    args = parser.parse_args()
    device='cuda'
    model, image_processor = clip.load("ViT-B/32", device=device)
    pick_bottle_dataset = get_pick_bottle_dataset(args, image_processor, clip, epoch=0)
    pick_bottle_dataset.set_epoch(epoch=0)
    pick_bottle_loader = pick_bottle_dataset.dataloader
    num_batches_per_epoch = pick_bottle_loader.num_batches
    rank = args.rank if "rank" in args else 0
    t = tqdm(
        enumerate(pick_bottle_loader),
        disable=rank != 0,
        total=num_batches_per_epoch,
    )
    t1 = time.time()
    mv_avg_loss = []
    for num_steps, batch in t:
        if num_steps > 0:
            torch.cuda.synchronize()
            t2 = time.time()
            print("t2 - t1", t2 - t1)
        torch.cuda.synchronize()
        t1 = time.time()
