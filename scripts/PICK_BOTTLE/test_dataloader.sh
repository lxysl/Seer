### NEED TO CHANGE ###
save_checkpoint_path="./checkpoints"
root_dir="your_path_to_the_parent_folder_of_real_data"
real_dataset_names="your_real_dataset_name"
vit_checkpoint_path="xxx/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
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

python utils/data_utils.py \
    --calvin_dataset "/home/gjt/yanchi/teleoperation_dds/data/recordings/pick_bottle" \
    --window_size 10 \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --traj_cons \
    --batch_size 16 \
    --workers 8 \
    --phase "finetune" \
    --save_checkpoint_path "./checkpoints" \

    # --gradient_accumulation_steps 4 \
    # --bf16_module "vision_encoder" \
    # --vit_checkpoint_path ${vit_checkpoint_path} \
    # --lr_scheduler cosine \
    # --save_every_iter 100000 \
    # --num_epochs 40 \
    # --seed 42 \
    # --precision fp32 \
    # --learning_rate 1e-3 \
    # --save_checkpoint \
    # --finetune_type real \
    # --wandb_project seer \
    # --weight_decay 1e-4 \
    # --num_resampler_query 6 \
    # --run_name sn_scratch \
    # --except_lang \
    # --transformer_layers 24 \
    # --action_pred_steps 3 \
    # --sequence_length 7 \
    # --future_steps 3 \
    # --obs_pred \
    # --loss_action \
    # --loss_image \
    # --save_checkpoint_seq 1 \
    # --start_save_checkpoint 15 \
    # --warmup_epochs 5 \
    # --real_dataset_names ${real_dataset_names} \
    # --use_aug_data \
    # --report_to_wandb \


# image_primary: torch.Size([batch_size, 10, 3, 224, 224])
# image_wrist: torch.Size([batch_size, 10, 3, 224, 224])
# state: torch.Size([batch_size, 10, 68])
# text_token: torch.Size([batch_size, 77])
# action: torch.Size([batch_size, 10, 65])
