import os
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm

def calculate_dataset_statistics():
    # 数据集根目录，根据实际情况修改
    root_dir = "/home/gjt/yanchi/teleoperation_dds/data/recordings/pick_bottle"
    
    # 查找所有时间命名的文件夹
    folders = glob(os.path.join(root_dir, "????-??-??_??-??-??"))
    
    # 初始化数据收集列表
    all_actions = []
    all_states = []
    
    # 遍历所有文件夹
    for folder_path in tqdm(folders, desc="Processing folders"):
        # 查找该文件夹下的所有episode文件
        episode_files = glob(os.path.join(folder_path, "episode_*.hdf5"))
        
        # 遍历每个episode文件
        for h5_filename in episode_files:
            try:
                with h5py.File(h5_filename, "r") as f:
                    # 获取该文件中的所有数据点
                    # 加载所有action维度
                    actions_hand = f["action/hand"][:]    
                    actions_pose = f["action/pose"][:]    
                    actions_robot = f["action/robot"][:]  
                    
                    # 加载所有state维度
                    state_hand = f["state/hand"][:]       
                    state_pose = f["state/pose"][:]       
                    state_robot = f["state/robot"][:]     
                    
                    # 合并为单一向量
                    actions = np.concatenate([actions_hand, actions_pose, actions_robot], axis=1)
                    states = np.concatenate([state_hand, state_pose, state_robot], axis=1)
                    
                    # 将数据添加到收集列表
                    all_actions.append(actions)
                    all_states.append(states)
            except Exception as e:
                print(f"Error processing {h5_filename}: {e}")
    
    # 合并所有数据
    print("Concatenating all data...")
    actions = np.concatenate(all_actions, axis=0)
    states = np.concatenate(all_states, axis=0)
    
    # 计算合并后向量的均值和方差
    print("Calculating statistics...")
    stats = {
        "action": {
            "mean": np.mean(actions, axis=0),
            "std": np.std(actions, axis=0)
        },
        "state": {
            "mean": np.mean(states, axis=0),
            "std": np.std(states, axis=0)
        }
    }
    
    # 保存统计结果
    output_file = "dataset_statistics.npz"
    print(f"Saving statistics to {output_file}")
    
    np.savez(output_file, 
             action_mean=stats["action"]["mean"], 
             action_std=stats["action"]["std"],
             state_mean=stats["state"]["mean"], 
             state_std=stats["state"]["std"])
    
    # 打印一些统计信息
    print("\nDataset Statistics Summary:")
    print(f"Total samples: {len(actions)}")
    print(f"Action dimensions: {actions.shape[1]}")
    print(f"State dimensions: {states.shape[1]}")
    
    # 检查是否有标准差为0的维度（这些维度在归一化时可能会有问题）
    zero_std_action = np.where(stats["action"]["std"] == 0)[0]
    zero_std_state = np.where(stats["state"]["std"] == 0)[0]
    
    if len(zero_std_action) > 0:
        print(f"\nWARNING: Found {len(zero_std_action)} action dimensions with zero standard deviation!")
        print(f"Zero std action indices: {zero_std_action}")
    
    if len(zero_std_state) > 0:
        print(f"\nWARNING: Found {len(zero_std_state)} state dimensions with zero standard deviation!")
        print(f"Zero std state indices: {zero_std_state}")
    
    return stats


def generate_mock_statistics():
    # Create mock statistics with the specified dimensions
    action_mean = np.random.normal(0, 1, size=(65,))
    action_std = np.abs(np.random.normal(1, 0.3, size=(65,)))  # Using abs to ensure positive std values
    state_mean = np.random.normal(0, 1, size=(68,))
    state_std = np.abs(np.random.normal(1, 0.3, size=(68,)))   # Using abs to ensure positive std values
    
    # Save the mock statistics to a file
    output_file = "dataset_statistics.npz"
    print(f"Saving mock statistics to {output_file}")
    
    np.savez(output_file, 
             action_mean=action_mean, 
             action_std=action_std,
             state_mean=state_mean, 
             state_std=state_std)
    
    # Print information about the generated data
    print("\nMock Statistics Summary:")
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Action std shape: {action_std.shape}")
    print(f"State mean shape: {state_mean.shape}")
    print(f"State std shape: {state_std.shape}")
    
    # Load and verify the saved data
    data = np.load("dataset_statistics.npz")
    print("\nVerifying saved data:")
    print(f"Keys in saved file: {list(data.keys())}")
    print(f"action_mean shape: {data['action_mean'].shape}")
    print(f"action_std shape: {data['action_std'].shape}")
    print(f"state_mean shape: {data['state_mean'].shape}")
    print(f"state_std shape: {data['state_std'].shape}")
    
    return {
        "action": {
            "mean": action_mean,
            "std": action_std
        },
        "state": {
            "mean": state_mean,
            "std": state_std
        }
    }


if __name__ == "__main__":
    # stats = calculate_dataset_statistics()

    stats = generate_mock_statistics()

    # 打印部分统计结果示例
    print("\nExample statistics:")
    print(f"Action mean: {stats['action']['mean']}")
    print(f"Action std: {stats['action']['std']}")
    print(f"State mean: {stats['state']['mean']}")
    print(f"State std: {stats['state']['std']}")
    
    # 打印每个部分的维度范围信息
    print("\nDimension ranges:")
    print("Action:")
    print(f"  Hand: 0-11 (12 dims)")
    print(f"  Pose: 12-35 (24 dims)")
    print(f"  Robot: 36-64 (29 dims)")
    print("State:")
    print(f"  Hand: 0-11 (12 dims)")
    print(f"  Pose: 12-38 (27 dims)")
    print(f"  Robot: 39-67 (29 dims)")

    # load
    data = np.load("dataset_statistics.npz")
    print(data.keys())
    print(data["action_mean"].shape)
    print(data["action_std"].shape)
    print(data["state_mean"].shape)
    print(data["state_std"].shape)
