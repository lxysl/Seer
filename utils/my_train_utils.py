import time
from contextlib import suppress

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from einops import rearrange
from pdb import set_trace
import numpy as np
import torch.distributed as dist


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
    return cast_dtype

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def get_ckpt_name(args, epoch=-1):
    return f'{epoch}.pth'

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """

    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))

    return x

def normalize_patchfied_image(patchfied_imgs):
    mean = patchfied_imgs.mean(dim=-1, keepdim=True)
    var = patchfied_imgs.var(dim=-1, keepdim=True)
    patchfied_imgs = (patchfied_imgs - mean) / (var + 1.e-6)**.5

    return patchfied_imgs

def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both pick_bottle (= 1 batch regardless of gradient accum)
    end = time.time()
    
    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        # images
        images_left = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_right = batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        # text tokens
        text_tokens = batch_calvin[1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)
        
        # states - now handles hand, pose, robot components
        states = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        # Assuming states contains concatenated [hand(12), pose(27), robot(29)] values
        # No need for special handling of gripper width as in the old code
        input_states = states  # Use the full state vector
        
        # self_key_point
        self_keypoints = None
        
        # actions - now handles hand, pose, robot components
        actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        # Prepare inputs for the model
        input_image_left = images_left[:, :args.sequence_length, :]
        input_image_right = images_right[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]

        input_hand_state = input_states[:, :args.sequence_length, :12] if args.control_type in ["position", "all"] else None
        input_pose_state = input_states[:, :args.sequence_length, 12:12+27] if args.control_type in ["pose", "all"] else None
        input_robot_state = input_states[:, :args.sequence_length, 12+27:] if args.control_type in ["position", "all"] else None
            
        # Prepare label actions for all components
        label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) 
                                 for j in range(args.action_pred_steps)], dim=-2)
        
        # Separate label actions for hand, pose, and robot components
        # Assuming actions are concatenated as [hand(12), pose(24), robot(29)]
        hand_dim, pose_dim, robot_dim = 12, 24, 29
        label_hand_actions = label_actions[..., :hand_dim]
        label_pose_actions = label_actions[..., hand_dim:hand_dim+pose_dim]
        label_robot_actions = label_actions[..., hand_dim+pose_dim:hand_dim+pose_dim+robot_dim]

        with autocast():
            # Call model with new parameter names and receive new outputs
            hand_pred_action, pose_pred_action, robot_pred_action, image_pred, \
            hand_pred_state, pose_pred_state, robot_pred_state, loss_action = model(
                image_left=input_image_left,
                image_right=input_image_right,
                hand_state=input_hand_state,
                pose_state=input_pose_state,
                robot_state=input_robot_state,
                text_token=input_text_token,
                action=actions[:, :args.sequence_length, :],
            )
        
        # 根据控制类型计算不同的损失
        # Initialize losses with zero tensors
        loss_hand_action = torch.tensor([0.0]).to(device_id)
        loss_pose_action = torch.tensor([0.0]).to(device_id)
        loss_robot_action = torch.tensor([0.0]).to(device_id)
        
        # Calculate action losses based on control_type
        if args.loss_action and args.action_pred_steps:
            # Hand action loss (for "position" or "all" control types)
            if args.control_type in ["position", "all"] and hand_pred_action is not None:
                loss_hand_action = torch.nn.functional.smooth_l1_loss(
                    hand_pred_action[:, :args.sequence_length-args.atten_goal], 
                    label_hand_actions[:, :args.sequence_length-args.atten_goal].detach())
            
            # Pose action loss (for "pose" or "all" control types)
            if args.control_type in ["pose", "all"] and pose_pred_action is not None:
                loss_pose_action = torch.nn.functional.smooth_l1_loss(
                    pose_pred_action[:, :args.sequence_length-args.atten_goal], 
                    label_pose_actions[:, :args.sequence_length-args.atten_goal].detach())
            
            # Robot action loss (for "position" or "all" control types)
            if args.control_type in ["position", "all"] and robot_pred_action is not None:
                loss_robot_action = torch.nn.functional.smooth_l1_loss(
                    robot_pred_action[:, :args.sequence_length-args.atten_goal], 
                    label_robot_actions[:, :args.sequence_length-args.atten_goal].detach())

        # Image loss calculation 
        if args.loss_image and args.obs_pred and image_pred is not None:
            label_image_left = images_left[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_right = images_right[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_left = patchify(label_image_left, patch_size=args.patch_size)
            label_image_right = patchify(label_image_right, patch_size=args.patch_size)
            label_image_left = normalize_patchfied_image(label_image_left)
            label_image_right = normalize_patchfied_image(label_image_right)
            
            image_pred = image_pred.reshape(-1, args.sequence_length, image_pred.shape[1], image_pred.shape[2], image_pred.shape[3])
            image_pred = image_pred[:, :args.sequence_length-args.atten_goal]
            image_pred = image_pred.reshape(-1, image_pred.shape[2], image_pred.shape[3], image_pred.shape[4])
            
            loss_image = 0.5 * (torch.nn.functional.mse_loss(
                            image_pred[:, 0, :, :], 
                            label_image_left.detach()) + 
                            torch.nn.functional.mse_loss(
                            image_pred[:, 1, :, :], 
                            label_image_right.detach()))
        else:
            loss_image = torch.tensor([0.0]).to(device_id)
        
        # Combined loss - now includes components based on control_type
        loss_action_combined = (
            args.loss_hand_action_ratio * loss_hand_action + 
            args.loss_pose_action_ratio * loss_pose_action + 
            args.loss_robot_action_ratio * loss_robot_action
        )
        
        loss_calvin = loss_action_combined + args.loss_image_ratio * loss_image

        # gradient_accumulation_steps        
        loss = loss_calvin / args.gradient_accumulation_steps
        loss_hand_action = loss_hand_action / args.gradient_accumulation_steps
        loss_pose_action = loss_pose_action / args.gradient_accumulation_steps
        loss_robot_action = loss_robot_action / args.gradient_accumulation_steps
        loss_image = loss_image / args.gradient_accumulation_steps
        mv_avg_loss.append(loss.item())

        ### backward pass ###
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": loss.item() * args.gradient_accumulation_steps,
                        "loss_hand_action": loss_hand_action.item() * args.gradient_accumulation_steps,
                        "loss_pose_action": loss_pose_action.item() * args.gradient_accumulation_steps,
                        "loss_robot_action": loss_robot_action.item() * args.gradient_accumulation_steps,
                        "loss_image": loss_image.item() * args.gradient_accumulation_steps,
                        "global_step": global_step,
                    },
                )

        avg_horizon = min(100, len(mv_avg_loss))
        # 根据控制类型显示不同的损失
        postfix_dict = {"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item()}
        
        if args.loss_image and args.obs_pred:
            postfix_dict["loss_image"] = loss_image.item()
            
        if args.control_type in ["position", "all"]:
            postfix_dict["loss_hand"] = loss_hand_action.item()
            postfix_dict["loss_robot"] = loss_robot_action.item()
            
        if args.control_type in ["pose", "all"]:
            postfix_dict["loss_pose"] = loss_pose_action.item()
            
        t.set_postfix(postfix_dict)

def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict

def get_checkpoint_all_param(model):
    state_dict = model.state_dict()

    return state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def eval_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    device_id,
    wandb,
):
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.eval()

    # setup logging
    step_time_m = AverageMeter()  # time for one step
    data_time_m = AverageMeter()  # avg time to load one batch
    end = time.time()
    
    # metrics
    val_loss_action = AverageMeter()
    val_loss_hand_action = AverageMeter()
    val_loss_pose_action = AverageMeter()
    val_loss_robot_action = AverageMeter()
    val_loss_image = AverageMeter()
    val_loss_calvin = AverageMeter()
    
    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=num_batches_per_epoch,
        desc=f"validation epoch {epoch+1}/{args.num_epochs}"
    )
    
    with torch.no_grad():
        for num_steps, batch_calvin in t:
            data_time_m.update(time.time() - end)
            global_step = num_steps + epoch * num_batches_per_epoch

            # images
            images_left = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True)
            images_right = batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True)
            
            # text tokens
            text_tokens = batch_calvin[1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)
            
            # states - handles hand, pose, robot components
            states = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
            input_states = states  # Use the full state vector
            
            # actions - handles hand, pose, robot components
            actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
            
            # Prepare inputs for the model
            input_image_left = images_left[:, :args.sequence_length, :]
            input_image_right = images_right[:, :args.sequence_length, :]
            input_text_token = text_tokens[:, :args.sequence_length, :]
            
            input_hand_state = input_states[:, :args.sequence_length, :12] if args.control_type in ["position", "all"] else None
            input_pose_state = input_states[:, :args.sequence_length, 12:12+27] if args.control_type in ["pose", "all"] else None
            input_robot_state = input_states[:, :args.sequence_length, 12+27:] if args.control_type in ["position", "all"] else None

            # Prepare label actions for all components
            label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) 
                                    for j in range(args.action_pred_steps)], dim=-2)
            
            # Separate label actions for hand, pose, and robot components
            hand_dim, pose_dim, robot_dim = 12, 24, 29
            label_hand_actions = label_actions[..., :hand_dim]
            label_pose_actions = label_actions[..., hand_dim:hand_dim+pose_dim]
            label_robot_actions = label_actions[..., hand_dim+pose_dim:hand_dim+pose_dim+robot_dim]

            with autocast():
                # Call model with parameter names and receive outputs
                hand_pred_action, pose_pred_action, robot_pred_action, image_pred, \
                hand_pred_state, pose_pred_state, robot_pred_state, loss_action = model(
                    image_left=input_image_left,
                    image_right=input_image_right,
                    hand_state=input_hand_state,
                    pose_state=input_pose_state,
                    robot_state=input_robot_state,
                    text_token=input_text_token,
                    action=actions[:, :args.sequence_length, :],
                )
            
            # 根据控制类型计算不同的损失
            # Initialize losses with zero tensors
            loss_hand_action = torch.tensor([0.0]).to(device_id)
            loss_pose_action = torch.tensor([0.0]).to(device_id)
            loss_robot_action = torch.tensor([0.0]).to(device_id)
            
            # Calculate action losses based on control_type
            if args.loss_action and args.action_pred_steps:
                # Hand action loss (for "position" or "all" control types)
                if args.control_type in ["position", "all"] and hand_pred_action is not None:
                    loss_hand_action = torch.nn.functional.smooth_l1_loss(
                        hand_pred_action[:, :args.sequence_length-args.atten_goal], 
                        label_hand_actions[:, :args.sequence_length-args.atten_goal].detach())
                
                # Pose action loss (for "pose" or "all" control types)
                if args.control_type in ["pose", "all"] and pose_pred_action is not None:
                    loss_pose_action = torch.nn.functional.smooth_l1_loss(
                        pose_pred_action[:, :args.sequence_length-args.atten_goal], 
                        label_pose_actions[:, :args.sequence_length-args.atten_goal].detach())
                
                # Robot action loss (for "position" or "all" control types)
                if args.control_type in ["position", "all"] and robot_pred_action is not None:
                    loss_robot_action = torch.nn.functional.smooth_l1_loss(
                        robot_pred_action[:, :args.sequence_length-args.atten_goal], 
                        label_robot_actions[:, :args.sequence_length-args.atten_goal].detach())

            # Image loss calculation 
            if args.loss_image and args.obs_pred and image_pred is not None:
                label_image_left = images_left[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
                label_image_right = images_right[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
                label_image_left = patchify(label_image_left, patch_size=args.patch_size)
                label_image_right = patchify(label_image_right, patch_size=args.patch_size)
                label_image_left = normalize_patchfied_image(label_image_left)
                label_image_right = normalize_patchfied_image(label_image_right)
                
                image_pred = image_pred.reshape(-1, args.sequence_length, image_pred.shape[1], image_pred.shape[2], image_pred.shape[3])
                image_pred = image_pred[:, :args.sequence_length-args.atten_goal]
                image_pred = image_pred.reshape(-1, image_pred.shape[2], image_pred.shape[3], image_pred.shape[4])
                
                loss_image = 0.5 * (torch.nn.functional.mse_loss(
                                image_pred[:, 0, :, :], 
                                label_image_left.detach()) + 
                                torch.nn.functional.mse_loss(
                                image_pred[:, 1, :, :], 
                                label_image_right.detach()))
            else:
                loss_image = torch.tensor([0.0]).to(device_id)
            
            # Combined loss - includes all three action components based on control_type
            loss_action_combined = (
                args.loss_hand_action_ratio * loss_hand_action + 
                args.loss_pose_action_ratio * loss_pose_action + 
                args.loss_robot_action_ratio * loss_robot_action
            )
            
            loss_calvin = loss_action_combined + args.loss_image_ratio * loss_image
            
            # Update metrics
            val_loss_hand_action.update(loss_hand_action.item())
            val_loss_pose_action.update(loss_pose_action.item())
            val_loss_robot_action.update(loss_robot_action.item())
            val_loss_image.update(loss_image.item())
            val_loss_calvin.update(loss_calvin.item())
            
            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()
            
            if args.rank == 0:
                # 根据控制类型显示不同的postfix
                postfix_dict = {'val_loss': f'{val_loss_calvin.avg:.4f}'}
                
                if args.control_type in ["position", "all"]:
                    postfix_dict['hand_loss'] = f'{val_loss_hand_action.avg:.4f}'
                    postfix_dict['robot_loss'] = f'{val_loss_robot_action.avg:.4f}'
                
                if args.control_type in ["pose", "all"]:
                    postfix_dict['pose_loss'] = f'{val_loss_pose_action.avg:.4f}'
                
                if args.loss_image and args.obs_pred:
                    postfix_dict['img_loss'] = f'{val_loss_image.avg:.4f}'
                
                t.set_postfix(postfix_dict)

    # Log validation metrics
    if args.rank == 0 and args.report_to_wandb:
        wandb_log_dict = {
            "val/loss_calvin": val_loss_calvin.avg,
            "val/loss_image": val_loss_image.avg,
            "val/epoch": epoch,
        }
        
        # 根据控制类型记录不同的wandb指标
        if args.control_type in ["position", "all"]:
            wandb_log_dict["val/loss_hand_action"] = val_loss_hand_action.avg
            wandb_log_dict["val/loss_robot_action"] = val_loss_robot_action.avg
            
        if args.control_type in ["pose", "all"]:
            wandb_log_dict["val/loss_pose_action"] = val_loss_pose_action.avg
        
        wandb.log(wandb_log_dict)
    
    return val_loss_calvin.avg
        