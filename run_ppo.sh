CUDA_VISIBLE_DEVICES=0 deepspeed --module openrlhf.cli.train_ppo \
  --pretrain ../qwen-satori \
  --fixed_rm MATH \
  --save_path ./checkpoint/qwen-satori-math-rlhf \
  --save_steps 25 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data ../ScaleQuest-QwQ \
  --eval_data data/math \
  --input_key query \
  --answer_key answer \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb $WANDB_API_KEY

# Support remote reward model (HTTP)
# --remote_rm_url http://localhost:5000/get_reward
# --apply_chat_template \
# --advantage_estimator reinforce \