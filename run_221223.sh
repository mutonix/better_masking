python -m spacy download en_core_web_lg
python -m torch.distributed.launch --nproc_per_node=8 bert_ptw_pretrain.py \
    --steps 200000 \
    --base_run_name bert_wt_hfwwm_200k \
    --ckpt_output_dir ./checkpoints \
    --dsets_cache_dir ./datasets \
    --size large \
    --datas wikipedia \
    --probe \
    --pos_wt