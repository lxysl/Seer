### NEED TO CHANGE ###
save_checkpoint_path="./checkpoints"
root_dir="your_path_to_the_parent_folder_of_real_data"
real_dataset_names="your_real_dataset_name"
vit_checkpoint_path="ckpt/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
### NEED TO CHANGE ###

### EXAMPLE ###
# - root_dir
#   - real_dataset_names
#       - 0000
#           - 000000
#           - ......
#           - xxxxxx
#       - ....
#       - 00xx 
### EXAMPLE ###

node=1
node_num=1
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset "/home/gjt/yanchi/teleoperation_dds/data/recordings/pick_bottle" \
    --workers 4 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 50 \
    --seed 42 \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --save_checkpoint \
    --finetune_type bottle \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name pick_bottle_scratch \
    --save_checkpoint_path ${save_checkpoint_path} \
    --except_lang \
    --transformer_layers 24 \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --obs_pred \
    --loss_action \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint 1 \
    --warmup_epochs 5 \
    --use_aug_data \
    --report_to_wandb \
    # --loss_image \

