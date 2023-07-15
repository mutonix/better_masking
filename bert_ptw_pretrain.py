import logging
import os, random
from pathlib import Path
from functools import partial
from datetime import datetime, timezone, timedelta
import math
from dataclasses import dataclass, field
import argparse
from multiprocessing import Manager

import wandb
import numpy as np
import torch
import torch.nn as nn

import datasets
from transformers import BertConfig, BertTokenizerFast, BertForMaskedLM
from transformers import TrainingArguments, Trainer
from transformers.integrations import WandbCallback, rewrite_logs

from data_utils import MyConfig, BertDataProcessor, BertDataCollator

os.environ['WANDB_PROJECT'] = 'bert_pretrain'
os.environ['WANDB_API_KEY'] = '5fb826182752249e4b2d5cae0ab77acb5814211e'
os.environ['WANDB_WATCH'] = 'false'

c = MyConfig({})
c.update({
    'base_run_name': 'bert', # run_name = {base_run_name}_{seed}
    'seed': 11081, # 11081 36 1188 76 1 4 4649 7 # None/False to randomly choose seed from [0,999999
    'size': 'base',
    'datas': ['my_text'],

    'logger': 'wandb',
    'preprocess_dsets_num_proc': 16,
    'num_workers': 16,

    'n_gpus' : torch.cuda.device_count(),
    'max_grad_norm': 1.0,

    'logging_steps': 10,
    'from_pretrained': False,
    'grad_acc_steps': 1,  # `bert-base` for 8 Tesla-V100 GPUs RAM 32GB
    'deepspeed': None,
})


"""Pass arguments"""
parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--base_run_name", default=c.base_run_name, type=str)
parser.add_argument("--size", default=c.size, choices=["small", "base", "large"])
parser.add_argument("--datas", default=c.datas, nargs='+', type=str)
parser.add_argument("--num_workers", default=c.num_workers, type=int)

parser.add_argument('--steps', default=0, type=int)
parser.add_argument("--bs", default=0, type=int)
parser.add_argument("--lr", default=0., type=float)
parser.add_argument("--max_length", default=0, type=int)
parser.add_argument("--grad_acc_steps", default=c.grad_acc_steps, type=int)
parser.add_argument("--max_grad_norm", default=c.max_grad_norm, type=float)
parser.add_argument("--mask_prob", default=0., type=float)
parser.add_argument("--preprocess_dsets_num_proc", default=c.preprocess_dsets_num_proc, type=int, help="n cpus for preprocessing datasets")
parser.add_argument("--deepspeed", default=c.deepspeed)
parser.add_argument("--dsets_cache_dir", default='./datasets', type=str, help="directory for cache of datasets")
parser.add_argument("--ckpt_output_dir", default='./checkpoints', type=str, help="directory for checkpoints")
parser.add_argument("--probe", action="store_true", default=False)
parser.add_argument("--from_pretrained", action="store_true", default=False)
parser.add_argument("--pos_wt", action="store_true", default=False)
parser.add_argument("--only_wwm", action="store_true", default=False)
parser.add_argument("--masking_mode", type=str, default='')
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()
for k, v in vars(args).items():
    c.update({k: v})

# Setting of different sizes
i = ['small', 'base', 'large'].index(c.size)

if not c.steps: c.steps = [10**6, 10**6, 400*1000][i]
if not c.bs: c.bs = [128, 256, 256][i]
if not c.mask_prob: c.mask_prob = [0.15, 0.15, 0.15][i]
if not c.max_length: c.max_length = [128, 512, 512][i]
if not c.lr: c.lr = [5e-4, 2e-4, 1e-4][i]

# Check and Default
# for data in c.datas: assert data in ['wikipedia', 'wiki'bookcorpus', 'openwebtext', 'my_text']
assert c.logger in ['wandb', 'neptune', None, False]
if not c.base_run_name: c.base_run_name = str(datetime.now(timezone(timedelta(hours=+8))))[6:-13].replace(' ','').replace(':','').replace('-','')
if not c.seed: c.seed = random.randint(0, 999999)
c.run_name = f'{c.base_run_name}_{c.seed}'

bert_config = BertConfig.from_pretrained(f'bert-{c.size}-uncased')
hf_tokenizer = BertTokenizerFast.from_pretrained(f"bert-{c.size}-uncased")

# Path to data
#Path(c.dsets_cache_dir).mkdir(parent=True, exist_ok=True)
# Path to checkpoints
#Path(c.ckpt_output_dir + '/pretrain').mkdir(exist_ok=True, parents=True)

dsets = []
BertProcessor = partial(BertDataProcessor, hf_tokenizer=hf_tokenizer, max_length=c.max_length)

if 'wikipedia' in c.datas:
    print('load/download wiki dataset')
    wiki = datasets.load_dataset('wikipedia', '20200501.en', cache_dir=c.dsets_cache_dir)['train']
    print('load/create data from wiki dataset for BERT')
    e_wiki = BertProcessor(wiki).map(cache_file_name=f"wiki_probe_{c.max_length}.arrow", num_proc=c.preprocess_dsets_num_proc)
    dsets.append(e_wiki)

if 'wikitext' in c.datas:
    print('load/download wikitext')
    wiki = datasets.load_dataset('wikitext', 'wikitext-103-v1', cache_dir=c.dsets_cache_dir)['train']
    print('load/create data from wiki dataset for BERT')
    e_wiki = BertProcessor(wiki).map(cache_file_name=f"bert_wikitext_{c.max_length}.arrow", num_proc=c.preprocess_dsets_num_proc)
    dsets.append(e_wiki)

# OpenWebText
if 'openwebtext' in c.datas:
    print('load/download OpenWebText Corpus')
    owt = datasets.load_dataset('openwebtext', cache_dir=c.dsets_cache_dir)['train']
    print('load/create data from OpenWebText Corpus for BERT')
    e_owt = BertProcessor(owt).map(cache_file_name=f"bert_owt_{c.max_length}.arrow", num_proc=c.preprocess_dsets_num_proc)
    dsets.append(e_owt)

if 'my_text' in c.datas:
    print('load/download my text')
    mt = datasets.load_dataset("text", data_files={"train": "urlsf_subset00-944_data.txt"}, cache_dir=c.dsets_cache_dir)['train']
    print('load/create data from my text for BERT')
    e_mytext = BertProcessor(mt).map(cache_file_name=f"bert_mt_{c.max_length}.arrow", num_proc=c.preprocess_dsets_num_proc)
    dsets.append(e_mytext)

assert len(dsets) == len(c.datas)

bert_dset = datasets.concatenate_datasets(dsets)
if c.probe:
    bert_dset.set_format(type='torch', columns=['input_ids', 'sentA_length', 'pos_subword_info'])
else:
    bert_dset.set_format(type='torch', columns=['input_ids', 'sentA_length'])

"""Model & Loss"""
class BertModelWithLoss(nn.Module): 
    def __init__(self, bert, hf_tokenizer, mask_prob, max_steps, global_dict=None, probe=False, pos_wt=False, masking_mode=''):
        super().__init__()
        self.bert = bert
        self.hf_tokenizer = hf_tokenizer
        self.mask_prob = float(mask_prob)
        self.max_steps = max_steps
        self.masking_mode = masking_mode
        self.bert_loss_func = nn.CrossEntropyLoss(reduction='none')
        self.global_dict = global_dict
        self.probe = probe
        self.pos_wt = pos_wt
        if self.probe:
            self.all_pos_score = torch.tensor(self.global_dict["all_pos_score"])
            self.pos_classes = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

    def forward(self, input_ids, labels, **kwargs):
        """
        masked_inputs (Tensor[int]): (N, L)
        sentA_lenths (Tensor[int]): (N, )
        """
        pos_subword_info = kwargs.pop('pos_subword_info', None)
        steps = kwargs.pop('steps', None)

        with torch.no_grad():
            attention_mask = input_ids != self.hf_tokenizer.pad_token_id
            token_type_ids = (~attention_mask).long()
            mlm_mask = labels != -100
        
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        bert_loss = self.bert_loss_func(bert_output.logits.transpose(1, 2).float(), labels)

        bert_mlm_loss = bert_loss[mlm_mask].clone()
        bert_loss = bert_loss[mlm_mask].mean()

        metrics_for_wandb = {}
        m = 0.999
        if self.probe:
            with torch.no_grad():
                device = input_ids.device

                # pos tagging
                pos_info_collect = pos_subword_info[mlm_mask]
                pos_classes = torch.tensor(list(set(pos_info_collect.tolist())), device=device)
                current_pos_score = []

                self.all_pos_score = self.all_pos_score.to(device)
                for c in pos_classes:
                    current_pos_score.append(bert_mlm_loss[pos_info_collect==c].mean())
                self.all_pos_score[pos_classes] = (1 - m) * torch.tensor(current_pos_score, device=device) + m * self.all_pos_score[pos_classes]

                self.global_dict["all_pos_score"] = self.all_pos_score.tolist()
            metrics_for_wandb = {c: self.all_pos_score[i] for i, c in enumerate(self.pos_classes)}
        metrics_for_wandb.update({'mlm_ratio': mlm_mask.sum() / attention_mask.sum()})

        return bert_loss, metrics_for_wandb


class BertMaskedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metrics = TrainMetric()

    def compute_loss(self, model, inputs, return_outputs=False):
        
        inputs.update({'steps': self.state.global_step})
        loss, model_metrics = model(**inputs)
        self.train_metrics.update(model_metrics)

        self.state.train_metrics = self.train_metrics

        return loss

@dataclass
class TrainMetric():
    mlm_ratio: torch.FloatTensor = field(default=torch.tensor(0.))

    def update(self, metric_dict):
        for k, v in metric_dict.items(): setattr(self, k, v)

class BertWandbCallback(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            logs.update(vars(state.train_metrics))
            self._wandb.log({**logs, "train/global_step": state.global_step})

if __name__ == '__main__':
    """Pre-Training"""
    torch.backends.cudnn.benchmark = True
    random.seed(c.seed)
    np.random.seed(c.seed)
    torch.manual_seed(c.seed)


    m = Manager()
    global_dict = m.dict()
    global_dict["all_pos_score"] = [0.] * 17

    if c.size in ['base', 'large'] and c.from_pretrained:
        bert = BertForMaskedLM.from_pretrained(f"bert-{c.size}-uncased")
    else:
        bert = BertForMaskedLM(bert_config)

    bert_mlm_model = BertModelWithLoss(bert, hf_tokenizer, c.mask_prob, 
                                    max_steps=c.steps, 
                                    probe=c.probe,
                                    pos_wt=c.pos_wt,
                                    masking_mode=c.masking_mode,
                                    global_dict=global_dict)
    # bert_wwm_collator = DataCollatorForWholeWordMask(tokenizer=hf_tokenizer, mlm_probability=c.mask_prob)
    bert_data_collator = BertDataCollator(hf_tokenizer, c.max_length, probe=c.probe, pos_wt=c.pos_wt, global_dict=global_dict, only_wwm=c.only_wwm, max_steps=c.steps, mask_prob=c.mask_prob, masking_mode=c.masking_mode)

    print('Initialize args')
    training_args = TrainingArguments(
        run_name=f'{c.base_run_name}-{float(c.mask_prob)}-{c.size}',
        output_dir=c.ckpt_output_dir + f'/pretrain/{c.base_run_name}-{float(c.mask_prob)}-{c.size}',          # output directory
        logging_dir='./logs',            # directory for storing logs
        logging_steps=c.logging_steps,
        save_steps=5000,     # Number of updates steps before two checkpoint saves. default: 500
        save_total_limit=10, # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
        dataloader_num_workers=c.num_workers,
        remove_unused_columns=False,
        gradient_accumulation_steps=c.grad_acc_steps,
        per_device_train_batch_size=c.bs // c.n_gpus // c.grad_acc_steps,  # batch size per device during training
        max_grad_norm=c.max_grad_norm,
        lr_scheduler_type='linear' if c.masking_mode != 'cosine' else 'cosine',
        warmup_steps=10000,
        max_steps=c.steps,   # 100k
        seed=c.seed,
        fp16=True,
        local_rank=c.local_rank,
        deepspeed=c.deepspeed, # only works on 5 <= gcc <= 7
        learning_rate=c.lr,
        weight_decay=0.01,
        ddp_find_unused_parameters=False,
        report_to='none'
    )

    print('Initialize trainer')
    trainer = BertMaskedTrainer(
        model=bert_mlm_model,                 # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=bert_dset,          # training dataset
        data_collator=bert_data_collator,
        callbacks=[BertWandbCallback]
    )

    bert_data_collator.bind_trainer(trainer)

    print('Start training at ', datetime.now())
    try:
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        trainer.train()
