import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from models.seer_model import SeerAgent

# 导入测试目标函数
from utils.train_utils import train_one_epoch_calvin, AverageMeter

# 创建模拟数据集类
class MockCalvinDataset(Dataset):
    def __init__(self, batch_size=2, sequence_length=7, window_size=10):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.num_batches = 5  # 仅用于测试的小批量数据
    
    def __getitem__(self, idx):
        # 创建模拟数据，格式与CALVIN数据相同
        # [0]: images_left (B, window_size, 3, 224, 224)
        # [1]: text_tokens (B, 77)
        # [2]: actions (B, window_size, 65) - hand(12) + pose(24) + robot(29)  
        # [3]: images_right (B, window_size, 3, 224, 224)
        # [4]: states (B, window_size, 68) - hand(12) + pose(27) + robot(29)
        
        # 创建模拟图像数据
        images_left = torch.rand(self.batch_size, self.window_size, 3, 224, 224)
        images_right = torch.rand(self.batch_size, self.window_size, 3, 224, 224)
        
        # 创建模拟文本token
        text_tokens = torch.randint(0, 49408, (self.batch_size, 77))
        
        # 创建模拟动作数据
        actions = torch.rand(self.batch_size, self.window_size, 7)
        for i in range(self.window_size):
            actions[:, i, :].fill_(float(i))
        
        # 创建模拟状态数据
        states = torch.rand(self.batch_size, self.window_size, 7)
        for i in range(self.window_size):
            states[:, i, :].fill_(float(i) * -1.0)
        
        return images_left, text_tokens, actions, images_right, states
    
    def __len__(self):
        return self.num_batches

# 创建模拟的DataLoader类
class MockCalvinDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_batches = dataset.num_batches
        self.current_idx = 0
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx < self.num_batches:
            batch = self.dataset[self.current_idx]
            self.current_idx += 1
            return batch
        else:
            raise StopIteration

# 创建模拟的Args类
class MockArgs:
    def __init__(self):
        # 基本参数
        self.sequence_length = 7
        self.window_size = 10
        self.batch_size = 2
        self.gradient_accumulation_steps = 1
        self.num_epochs = 1
        self.precision = "fp32"
        self.rank = 0
        self.world_size = 1
        
        # 模型相关参数
        self.action_pred_steps = 3
        self.atten_goal = 0
        self.loss_action = True
        self.obs_pred = True
        self.loss_image = True
        self.future_steps = 3
        self.patch_size = 16
        self.atten_only_obs = False
        self.attn_robot_proprio_state = False

        self.gripper_width = False
        
        # 损失函数相关参数
        self.loss_hand_action_ratio = 1.0
        self.loss_pose_action_ratio = 1.0
        self.loss_robot_action_ratio = 1.0
        self.loss_image_ratio = 0.1
        
        # 路径相关参数
        self.vit_checkpoint_path = "ckpt/mae_pretrain_vit_base.pth"
        self.save_every_iter = 100
        self.save_checkpoint = False
        self.report_to_wandb = False

        # 控制类型
        self.control_type = "position"

        # 归一化数据
        self.normalize_data = False
        self.dataset_statistics_file = "dataset_statistics.npz"

def test_train_one_epoch_calvin():
    print("开始测试 train_one_epoch_calvin 函数")
    
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模拟参数
    args = MockArgs()
    
    # 创建模拟数据集和数据加载器
    calvin_dataset = MockCalvinDataset(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        window_size=args.window_size
    )
    calvin_loader = MockCalvinDataLoader(calvin_dataset)
    
    # 创建SeerAgent模型实例
    model = SeerAgent(
        finetune_type="bottle",
        clip_device=device,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=6,
        num_obs_token_per_image=9,
        calvin_input_image_size=224,
        patch_size=16,
        action_pred_steps=args.action_pred_steps,
        obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs,
        attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal,
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        transformer_layers=1,  # 使用较小的模型加快测试速度
        hidden_dim=384,
        transformer_heads=12,
        phase="finetune",
    ).to(device)
    
    # 初始化模型类型
    model._init_model_type()
    
    # 创建优化器和学习率调度器
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.1
    )
    
    # 创建学习率调度器
    total_training_steps = calvin_loader.num_batches * args.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps
    )
    
    # 如果使用wandb，在这里可以添加模拟的wandb对象
    if args.report_to_wandb:
        mock_wandb = wandb.init(project="seer", name="test")
    else:
        mock_wandb = None
    
    # 执行训练一个epoch
    try:
        print("开始训练一个epoch...")
        train_one_epoch_calvin(
            args=args,
            model=model,
            epoch=0,
            calvin_loader=calvin_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device_id=device,
            wandb=mock_wandb
        )
        print("训练完成！")
        
        # 输出模型参数梯度信息，确认训练正常进行
        total_params = 0
        params_with_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
        
        print(f"模型中需要梯度的参数总数: {total_params}")
        print(f"训练后有梯度的参数数量: {params_with_grad}")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_train_one_epoch_calvin()
