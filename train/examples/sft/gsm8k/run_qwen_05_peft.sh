# Tested with 2 & 4 GPUs

set -x

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=datasets/gsm8k/train-00000-of-00001.parquet \
    data.val_files=datasets/gsm8k/test-00000-of-00001.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    optim.lr=1e-4 \
    data.prompt_dict_keys=null \
    +data.response_dict_keys=null \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=saves/qwen-2.5-0.5b-gsm8k \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.logger=console \
    trainer.total_epochs=1 \
    model.lora_rank=32\
    model.lora_alpha=16 \
    model.target_modules=all-linear

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \