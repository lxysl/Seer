import numpy as np
import torch

class StateActionNormalizer:
    """
    用于状态和动作归一化的类。
    处理hand、pose、robot三个部分的归一化和反归一化操作。
    
    属性:
        state_mean: 状态的均值 [hand_mean, pose_mean, robot_mean]
        state_std: 状态的标准差 [hand_std, pose_std, robot_std]
        action_mean: 动作的均值 [hand_mean, pose_mean, robot_mean]
        action_std: 动作的标准差 [hand_std, pose_std, robot_std]
    """
    
    def __init__(self, stats_file_path):
        """
        初始化归一化器
        
        参数:
            stats_file_path: 包含均值和标准差的npz文件路径
        """
        try:
            data = np.load(stats_file_path)
            self.state_mean = torch.from_numpy(data["state_mean"].astype(np.float32))
            self.state_std = torch.from_numpy(data["state_std"].astype(np.float32))
            self.action_mean = torch.from_numpy(data["action_mean"].astype(np.float32))
            self.action_std = torch.from_numpy(data["action_std"].astype(np.float32))
            
            # 创建有效维度掩码，标出那些标准差不为0的维度
            self.valid_state_mask = self.state_std > 1e-6
            self.valid_action_mask = self.action_std > 1e-6
            
            # 打印各个部分的维度信息
            print("Loaded normalization statistics:")
            print(f"State mean shape: {self.state_mean.shape}")
            print(f"State std shape: {self.state_std.shape}")
            print(f"Action mean shape: {self.action_mean.shape}")
            print(f"Action std shape: {self.action_std.shape}")
            print(f"Valid state dimensions: {self.valid_state_mask.sum().item()}/{len(self.state_std)}")
            print(f"Valid action dimensions: {self.valid_action_mask.sum().item()}/{len(self.action_std)}")
            
            # 检查是否有无效维度并打印详细信息
            if not self.valid_state_mask.all():
                invalid_state_indices = torch.where(~self.valid_state_mask)[0].tolist()
                print(f"Warning: State dimensions with near-zero std detected: {invalid_state_indices}")
            
            if not self.valid_action_mask.all():
                invalid_action_indices = torch.where(~self.valid_action_mask)[0].tolist()
                print(f"Warning: Action dimensions with near-zero std detected: {invalid_action_indices}")
                
        except Exception as e:
            print(f"Error loading normalization statistics: {e}")
            raise
    
    def normalize_state(self, state):
        """
        对状态进行归一化处理，跳过标准差接近0的维度
        
        参数:
            state: 包含hand、pose、robot的状态张量 (batch_size, seq_len, state_dim)
            
        返回:
            normalized_state: 归一化后的状态张量
        """
        # 将均值和标准差移动到与state相同的设备上
        device = state.device
        mean = self.state_mean.to(device)
        std = self.state_std.to(device)
        valid_mask = self.valid_state_mask.to(device)
            
        # 创建归一化后的张量（初始值与原始状态相同）
        normalized_state = state.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(state.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行归一化
                normalized_state[..., i] = (state[..., i] - mean[i]) / std[i]
            
        return normalized_state
    
    def denormalize_state(self, normalized_state):
        """
        对归一化的状态进行反归一化处理，跳过标准差接近0的维度
        
        参数:
            normalized_state: 归一化的状态张量
            
        返回:
            state: 原始尺度的状态张量
        """
        # 将均值和标准差移动到与normalized_state相同的设备上
        device = normalized_state.device
        mean = self.state_mean.to(device)
        std = self.state_std.to(device)
        valid_mask = self.valid_state_mask.to(device)
            
        # 创建反归一化后的张量（初始值与归一化状态相同）
        denormalized_state = normalized_state.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(normalized_state.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行反归一化
                denormalized_state[..., i] = normalized_state[..., i] * std[i] + mean[i]
            
        return denormalized_state
    
    def normalize_action(self, action):
        """
        对动作进行归一化处理，跳过标准差接近0的维度
        
        参数:
            action: 包含hand、pose、robot的动作张量 (batch_size, seq_len, action_dim)
            
        返回:
            normalized_action: 归一化后的动作张量
        """
        # 将均值和标准差移动到与action相同的设备上
        device = action.device
        mean = self.action_mean.to(device)
        std = self.action_std.to(device)
        valid_mask = self.valid_action_mask.to(device)
            
        # 创建归一化后的张量（初始值与原始动作相同）
        normalized_action = action.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(action.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行归一化
                normalized_action[..., i] = (action[..., i] - mean[i]) / std[i]
            
        return normalized_action
    
    def denormalize_action(self, normalized_action):
        """
        对归一化的动作进行反归一化处理，跳过标准差接近0的维度
        
        参数:
            normalized_action: 归一化的动作张量
            
        返回:
            action: 原始尺度的动作张量
        """
        # 将均值和标准差移动到与normalized_action相同的设备上
        device = normalized_action.device
        mean = self.action_mean.to(device)
        std = self.action_std.to(device)
        valid_mask = self.valid_action_mask.to(device)
            
        # 创建反归一化后的张量（初始值与归一化动作相同）
        denormalized_action = normalized_action.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(normalized_action.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行反归一化
                denormalized_action[..., i] = normalized_action[..., i] * std[i] + mean[i]
            
        return denormalized_action

    def normalize_hand_action(self, hand_action):
        """
        仅对hand部分的动作进行归一化处理，跳过标准差接近0的维度
        
        参数:
            hand_action: hand动作张量 (batch_size, seq_len, 12)
            
        返回:
            normalized_hand_action: 归一化后的hand动作张量
        """
        # 将均值和标准差移动到与hand_action相同的设备上
        device = hand_action.device
        mean = self.action_mean[:12].to(device)
        std = self.action_std[:12].to(device)
        valid_mask = self.valid_action_mask[:12].to(device)
            
        # 创建归一化后的张量（初始值与原始动作相同）
        normalized_hand_action = hand_action.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(hand_action.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行归一化
                normalized_hand_action[..., i] = (hand_action[..., i] - mean[i]) / std[i]
            
        return normalized_hand_action
    
    def denormalize_hand_action(self, normalized_hand_action):
        """
        对归一化的hand动作进行反归一化处理，跳过标准差接近0的维度
        
        参数:
            normalized_hand_action: 归一化的hand动作张量
            
        返回:
            hand_action: 原始尺度的hand动作张量
        """
        # 将均值和标准差移动到与normalized_hand_action相同的设备上
        device = normalized_hand_action.device
        mean = self.action_mean[:12].to(device)
        std = self.action_std[:12].to(device)
        valid_mask = self.valid_action_mask[:12].to(device)
            
        # 创建反归一化后的张量（初始值与归一化动作相同）
        denormalized_hand_action = normalized_hand_action.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(normalized_hand_action.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行反归一化
                denormalized_hand_action[..., i] = normalized_hand_action[..., i] * std[i] + mean[i]
            
        return denormalized_hand_action
    
    def normalize_pose_action(self, pose_action):
        """
        仅对pose部分的动作进行归一化处理，跳过标准差接近0的维度
        
        参数:
            pose_action: pose动作张量 (batch_size, seq_len, 24)
            
        返回:
            normalized_pose_action: 归一化后的pose动作张量
        """
        # 将均值和标准差移动到与pose_action相同的设备上
        device = pose_action.device
        mean = self.action_mean[12:12+24].to(device)
        std = self.action_std[12:12+24].to(device)
        valid_mask = self.valid_action_mask[12:12+24].to(device)
            
        # 创建归一化后的张量（初始值与原始动作相同）
        normalized_pose_action = pose_action.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(pose_action.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行归一化
                normalized_pose_action[..., i] = (pose_action[..., i] - mean[i]) / std[i]
            
        return normalized_pose_action
    
    def denormalize_pose_action(self, normalized_pose_action):
        """
        对归一化的pose动作进行反归一化处理，跳过标准差接近0的维度
        
        参数:
            normalized_pose_action: 归一化的pose动作张量
            
        返回:
            pose_action: 原始尺度的pose动作张量
        """
        # 将均值和标准差移动到与normalized_pose_action相同的设备上
        device = normalized_pose_action.device
        mean = self.action_mean[12:12+24].to(device)
        std = self.action_std[12:12+24].to(device)
        valid_mask = self.valid_action_mask[12:12+24].to(device)
            
        # 创建反归一化后的张量（初始值与归一化动作相同）
        denormalized_pose_action = normalized_pose_action.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(normalized_pose_action.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行反归一化
                denormalized_pose_action[..., i] = normalized_pose_action[..., i] * std[i] + mean[i]
            
        return denormalized_pose_action
    
    def normalize_robot_action(self, robot_action):
        """
        仅对robot部分的动作进行归一化处理，跳过标准差接近0的维度
        
        参数:
            robot_action: robot动作张量 (batch_size, seq_len, 29)
            
        返回:
            normalized_robot_action: 归一化后的robot动作张量
        """
        # 将均值和标准差移动到与robot_action相同的设备上
        device = robot_action.device
        mean = self.action_mean[12+24:].to(device)
        std = self.action_std[12+24:].to(device)
        valid_mask = self.valid_action_mask[12+24:].to(device)
            
        # 创建归一化后的张量（初始值与原始动作相同）
        normalized_robot_action = robot_action.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(robot_action.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行归一化
                normalized_robot_action[..., i] = (robot_action[..., i] - mean[i]) / std[i]
            
        return normalized_robot_action
    
    def denormalize_robot_action(self, normalized_robot_action):
        """
        对归一化的robot动作进行反归一化处理，跳过标准差接近0的维度
        
        参数:
            normalized_robot_action: 归一化的robot动作张量
            
        返回:
            robot_action: 原始尺度的robot动作张量
        """
        # 将均值和标准差移动到与normalized_robot_action相同的设备上
        device = normalized_robot_action.device
        mean = self.action_mean[12+24:].to(device)
        std = self.action_std[12+24:].to(device)
        valid_mask = self.valid_action_mask[12+24:].to(device)
            
        # 创建反归一化后的张量（初始值与归一化动作相同）
        denormalized_robot_action = normalized_robot_action.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(normalized_robot_action.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行反归一化
                denormalized_robot_action[..., i] = normalized_robot_action[..., i] * std[i] + mean[i]
            
        return denormalized_robot_action
    
    def normalize_hand_state(self, hand_state):
        """
        仅对hand部分的状态进行归一化处理，跳过标准差接近0的维度
        
        参数:
            hand_state: hand状态张量 (batch_size, seq_len, 12)
            
        返回:
            normalized_hand_state: 归一化后的hand状态张量
        """
        # 将均值和标准差移动到与hand_state相同的设备上
        device = hand_state.device
        mean = self.state_mean[:12].to(device)
        std = self.state_std[:12].to(device)
        valid_mask = self.valid_state_mask[:12].to(device)
            
        # 创建归一化后的张量（初始值与原始状态相同）
        normalized_hand_state = hand_state.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(hand_state.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行归一化
                normalized_hand_state[..., i] = (hand_state[..., i] - mean[i]) / std[i]
            
        return normalized_hand_state
    
    def denormalize_hand_state(self, normalized_hand_state):
        """
        对归一化的hand状态进行反归一化处理，跳过标准差接近0的维度
        
        参数:
            normalized_hand_state: 归一化的hand状态张量
            
        返回:
            hand_state: 原始尺度的hand状态张量
        """
        # 将均值和标准差移动到与normalized_hand_state相同的设备上
        device = normalized_hand_state.device
        mean = self.state_mean[:12].to(device)
        std = self.state_std[:12].to(device)
        valid_mask = self.valid_state_mask[:12].to(device)
            
        # 创建反归一化后的张量（初始值与归一化状态相同）
        denormalized_hand_state = normalized_hand_state.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(normalized_hand_state.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行反归一化
                denormalized_hand_state[..., i] = normalized_hand_state[..., i] * std[i] + mean[i]
            
        return denormalized_hand_state
    
    def normalize_pose_state(self, pose_state):
        """
        仅对pose部分的状态进行归一化处理，跳过标准差接近0的维度
        
        参数:
            pose_state: pose状态张量 (batch_size, seq_len, 27)
            
        返回:
            normalized_pose_state: 归一化后的pose状态张量
        """
        # 将均值和标准差移动到与pose_state相同的设备上
        device = pose_state.device
        mean = self.state_mean[12:12+27].to(device)
        std = self.state_std[12:12+27].to(device)
        valid_mask = self.valid_state_mask[12:12+27].to(device)
            
        # 创建归一化后的张量（初始值与原始状态相同）
        normalized_pose_state = pose_state.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(pose_state.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行归一化
                normalized_pose_state[..., i] = (pose_state[..., i] - mean[i]) / std[i]
            
        return normalized_pose_state
    
    def denormalize_pose_state(self, normalized_pose_state):
        """
        对归一化的pose状态进行反归一化处理，跳过标准差接近0的维度
        
        参数:
            normalized_pose_state: 归一化的pose状态张量
            
        返回:
            pose_state: 原始尺度的pose状态张量
        """
        # 将均值和标准差移动到与normalized_pose_state相同的设备上
        device = normalized_pose_state.device
        mean = self.state_mean[12:12+27].to(device)
        std = self.state_std[12:12+27].to(device)
        valid_mask = self.valid_state_mask[12:12+27].to(device)
            
        # 创建反归一化后的张量（初始值与归一化状态相同）
        denormalized_pose_state = normalized_pose_state.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(normalized_pose_state.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行反归一化
                denormalized_pose_state[..., i] = normalized_pose_state[..., i] * std[i] + mean[i]
            
        return denormalized_pose_state
    
    def normalize_robot_state(self, robot_state):
        """
        仅对robot部分的状态进行归一化处理，跳过标准差接近0的维度
        
        参数:
            robot_state: robot状态张量 (batch_size, seq_len, 29)
            
        返回:
            normalized_robot_state: 归一化后的robot状态张量
        """
        # 将均值和标准差移动到与robot_state相同的设备上
        device = robot_state.device
        mean = self.state_mean[12+27:].to(device)
        std = self.state_std[12+27:].to(device)
        valid_mask = self.valid_state_mask[12+27:].to(device)
            
        # 创建归一化后的张量（初始值与原始状态相同）
        normalized_robot_state = robot_state.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(robot_state.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行归一化
                normalized_robot_state[..., i] = (robot_state[..., i] - mean[i]) / std[i]
            
        return normalized_robot_state
    
    def denormalize_robot_state(self, normalized_robot_state):
        """
        对归一化的robot状态进行反归一化处理，跳过标准差接近0的维度
        
        参数:
            normalized_robot_state: 归一化的robot状态张量
            
        返回:
            robot_state: 原始尺度的robot状态张量
        """
        # 将均值和标准差移动到与normalized_robot_state相同的设备上
        device = normalized_robot_state.device
        mean = self.state_mean[12+27:].to(device)
        std = self.state_std[12+27:].to(device)
        valid_mask = self.valid_state_mask[12+27:].to(device)
            
        # 创建反归一化后的张量（初始值与归一化状态相同）
        denormalized_robot_state = normalized_robot_state.clone()
        
        # 对每个维度单独处理，避免形状广播问题
        for i in range(normalized_robot_state.shape[-1]):
            if valid_mask[i]:
                # 只对有效维度进行反归一化
                denormalized_robot_state[..., i] = normalized_robot_state[..., i] * std[i] + mean[i]
            
        return denormalized_robot_state 