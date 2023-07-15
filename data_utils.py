import os
import re
import math
import random

import datasets
import spacy
import tokenizations
from collections.abc import Mapping

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import  DataCollatorForWholeWordMask
from transformers.data.data_collator import tolist, _torch_collate_batch


class MyConfig(dict):
  def __getattr__(self, name): return self[name]
  def __setattr__(self, name, value): self[name] = value


pos_tagger = spacy.load('en_core_web_lg')

class BertDataProcessor():
  def __init__(self, hf_dset, hf_tokenizer, max_length, text_col='text', lines_delimiter='\n', minimize_data_size=True, apply_cleaning=True):
    self.hf_tokenizer = hf_tokenizer
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length

    self.hf_dset = hf_dset
    self.text_col = text_col
    self.lines_delimiter = lines_delimiter
    self.minimize_data_size = minimize_data_size
    self.apply_cleaning = apply_cleaning
    pos_classes = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    self.pos_hash = {c: i for i, c in enumerate(pos_classes)}

  def map(self, **kwargs) -> datasets.arrow_dataset.Dataset:
    num_proc = kwargs.pop('num_proc', os.cpu_count())
    cache_file_name = kwargs.pop('cache_file_name', None)
    if cache_file_name is not None:
        if not cache_file_name.endswith('.arrow'): 
            cache_file_name += '.arrow'        
        if '/' not in cache_file_name: 
            cache_dir = os.path.abspath(os.path.dirname(self.hf_dset.cache_files[0]['filename']))
            cache_file_name = os.path.join(cache_dir, cache_file_name)

    return self.hf_dset.map(
        function=self,
        batched=True,
        cache_file_name=cache_file_name,
        remove_columns=self.hf_dset.column_names,
        disable_nullable=True,
        input_columns=[self.text_col],
        writer_batch_size=10**4,
        num_proc=num_proc,
        **kwargs     
    )

  def __call__(self, texts):
    if self.minimize_data_size: new_example = {'input_ids':[], 'sentA_length':[], 'pos_subword_info':[]}
    else: new_example = {'input_ids':[], 'input_mask': [], 'segment_ids': []}

    for text in texts: # for every doc
      
      for line in re.split(self.lines_delimiter, text): # for every paragraph
        
        if re.fullmatch(r'\s*', line): continue # empty string or string with all space characters
        if self.apply_cleaning and self.filter_out(line): continue
        
        example = self.add_line(line)
        if example:
          for k,v in example.items(): new_example[k].append(v)
      
      if self._current_length != 0:
        example = self._create_example()
        for k,v in example.items(): new_example[k].append(v)

    return new_example

  def filter_out(self, line):
    if len(line) < 80: return True
    return False 

  def clean(self, line):
    # () is remainder after link in it filtered out
    return line.strip().replace("\n", " ").replace("()","")

  def add_line(self, line):
    """Adds a line of text to the current example being built."""
    line = self.clean(line)
    tokens = self.hf_tokenizer.tokenize(line, max_length=512, truncation=True)
    tokids = self.hf_tokenizer.convert_tokens_to_ids(tokens)
    self._current_sentences.append(tokids)
    self._current_length += len(tokids)
    if self._current_length >= self._target_length:
      return self._create_example()
    return None

  def _create_example(self):
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    if random.random() < 0.1:
      first_segment_target_length = 100000
    else:
      # -3 due to not yet having [CLS]/[SEP] tokens in the input text
      first_segment_target_length = (self._target_length - 3) // 2

    first_segment = []
    second_segment = []
    for sentence in self._current_sentences:
      # the sentence goes to the first segment if (1) the first segment is
      # empty, (2) the sentence doesn't put the first segment over length or
      # (3) 50% of the time when it does put the first segment over length
      if (len(first_segment) == 0 or
          len(first_segment) + len(sentence) < first_segment_target_length or
          (len(second_segment) == 0 and
           len(first_segment) < first_segment_target_length and
           random.random() < 0.5)):
        first_segment += sentence
      else:
        second_segment += sentence

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[:self._max_length - 2]
    second_segment = second_segment[:max(0, self._max_length -
                                         len(first_segment) - 3)]

    # prepare to start building the next example
    self._current_sentences = []
    self._current_length = 0
    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
      self._target_length = random.randint(5, self._max_length)
    else:
      self._target_length = self._max_length

    return self._make_example(first_segment, second_segment)

  def _make_example(self, first_segment, second_segment):
    """Converts two "segments" of text into a tf.train.Example."""
    input_ids = [self.hf_tokenizer.cls_token_id] + first_segment + [self.hf_tokenizer.sep_token_id]

    bert_tokens = self.hf_tokenizer.convert_ids_to_tokens(first_segment)
    sentence = self.hf_tokenizer.decode(first_segment)

    with pos_tagger.select_pipes(enable=['morphologizer', 'tok2vec', 'tagger', 'attribute_ruler']):
      spacy_doc = pos_tagger(sentence)
    spacy_tokens = [t.text for t in spacy_doc]
    pos = torch.tensor([self.pos_hash[t.pos_] for t in spacy_doc])

    # align spacy_tokens to bert_tokens
    a2b, b2a = tokenizations.get_alignments(spacy_tokens, bert_tokens)

    count = 0
    align_index = []
    token_top = -1
    for i in range(len(spacy_tokens)):
      for j in a2b[i]:
        if j > token_top:
          align_index.append(count)
      count += 1
      token_top = a2b[i][-1]
    
    align_index = torch.tensor(align_index)
    # assign pos to bert_tokens
    pos_subword_info = torch.index_select(pos, dim=0, index=align_index)
    pos_subword_info = [-1] + pos_subword_info.tolist() + [-1]

    sentA_length = len(input_ids)
    segment_ids = [0] * sentA_length
    assert len(input_ids) == len(pos_subword_info)

    # if second_segment:
    #   input_ids += second_segment + [self.hf_tokenizer.sep_token_id]
    #   segment_ids += [1] * (len(second_segment) + 1)

    if self.minimize_data_size:
      return {
        'input_ids': input_ids,
        'sentA_length': sentA_length,
        'pos_subword_info': pos_subword_info
      }
    else:
      input_mask = [1] * len(input_ids)
      input_ids += [0] * (self._max_length - len(input_ids))
      input_mask += [0] * (self._max_length - len(input_mask))
      segment_ids += [0] * (self._max_length - len(segment_ids))
      return {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
      }


class BertDataCollator(DataCollatorForWholeWordMask):
    def __init__(self, hf_tokenizer, max_length, global_dict=None, probe=False, pos_wt=False, only_wwm=False, max_steps=None, mask_prob=None, masking_mode=None):
        self.tokenizer = hf_tokenizer
        self._max_length = max_length
        self.global_dict = global_dict
        self.probe = probe
        self.pos_wt_flag = pos_wt
        self.only_wwm = only_wwm
        self.max_steps = max_steps
        self.mask_prob = mask_prob
        self.masking_mode = masking_mode
        # if self.pos_wt_flag:
        #   # ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        #   self.all_pos_score = self.global_dict["all_pos_score"]

    def bind_trainer(self, trainer):
      self.trainer = trainer

    def __call__(self, samples):
        input_ids, sentA_length, pos_subword_info = [], [], []
        
        if self.probe:
          for s in samples:
              input_ids.append(s['input_ids'])
              sentA_length.append(s['sentA_length'].unsqueeze(0))
              pos_subword_info.append(s['pos_subword_info'])

          input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).long()
          sentA_length = torch.cat(sentA_length)
          pos_subword_info = pad_sequence(pos_subword_info, batch_first=True, padding_value=-1).long()

          if self.pos_wt_flag:
            self.all_pos_score = torch.tensor(self.global_dict["all_pos_score"])

            if not self.only_wwm:   
              std = 1e-4 if self.all_pos_score.std() == 0 else self.all_pos_score.std()
              self.all_pos_wt = torch.sigmoid((self.all_pos_score - self.all_pos_score.mean()) / std * 2)
            else:
                self.all_pos_wt = torch.ones_like(self.all_pos_score)
            
            batch = self.torch_call(samples, pos_subword_info)
            return {
                'input_ids':  batch['input_ids'], 
                'labels': batch['labels'],
                'pos_subword_info': pos_subword_info
            }
          else:
            return {
                'input_ids':  input_ids, 
                'sentA_length':  sentA_length, 
                'pos_subword_info':  pos_subword_info
            }

        else:
          for s in samples:
              input_ids.append(s['input_ids'])
              sentA_length.append(s['sentA_length'].unsqueeze(0))


          input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).long()
          sentA_length = torch.cat(sentA_length)

          return {
              'input_ids':  input_ids, 
              'sentA_length':  sentA_length, 
          }

    def torch_call(self, examples, pos_subword_info):
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for i, e in enumerate(examples):
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            # if "chinese_ref" in e:
            #     ref_pos = tolist(e["chinese_ref"])
            #     len_seq = len(e["input_ids"])
            #     for i in range(len_seq):
            #         if i in ref_pos:
            #             ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens, pos_subword_info[i]))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(self, input_tokens, pos_seq, max_predictions=512):
            """
            Get 0/1 labels for masked tokens with whole word mask proxy
            """
            # if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            #     warnings.warn(
            #         "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
            #         "Please refer to the documentation for more information."
            #     )

            cand_indexes = []
            for (i, token) in enumerate(input_tokens):
                if token == "[CLS]" or token == "[SEP]":
                    continue

                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([pos_seq[i]])
                    cand_indexes[-1].append(i)

            random.shuffle(cand_indexes)
            self.mlm_probability = self.get_mask_prob(self.trainer.state.global_step)
            num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
            masked_lms = []
            covered_indexes = set()
            for index_set in cand_indexes:
                pos = index_set[0]
                index_set = index_set[1:]
                if random.random() <= self.all_pos_wt[pos]:
                  if len(masked_lms) >= num_to_predict:
                      break
                  # If adding a whole-word mask would exceed the maximum number of
                  # predictions, then just skip this candidate.
                  if len(masked_lms) + len(index_set)> num_to_predict:
                      continue
                  is_any_index_covered = False
                  for index in index_set:
                      if index in covered_indexes:
                          is_any_index_covered = True
                          break
                  if is_any_index_covered:
                      continue

                  for index in index_set:
                      covered_indexes.add(index)
                      masked_lms.append(index)

            if len(covered_indexes) != len(masked_lms):
                raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
            mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
            return mask_labels

    def get_mask_prob(self, steps):
        if self.masking_mode:
            if self.masking_mode == 'asc': mask_prob = steps / self.max_steps * 2 * self.mask_prob
            elif self.masking_mode == 'desc': mask_prob = (1 - steps / self.max_steps) * 2 * self.mask_prob
            elif self.masking_mode == 'square_above': mask_prob = - (steps / self.max_steps * math.sqrt(self.mask_prob * 2)) ** 2 + self.mask_prob * 2 + 0.01
            elif self.masking_mode == 'square_below': mask_prob = self.mask_prob * 2 / (self.max_steps ** 2) * (steps - self.max_steps) ** 2
            elif self.masking_mode == 'poly_asc': 
                mask_prob = steps / self.max_steps * 4 * self.mask_prob if steps < self.max_steps / 2 \
                                    else (1 - steps / self.max_steps) * 4 * self.mask_prob
            elif self.masking_mode == 'lin_warm_decay':
                warmup_steps = 80000
                mask_prob = steps / warmup_steps * 2 * self.mask_prob if steps < warmup_steps \
                                else (1 + (warmup_steps - steps) / (self.max_steps - warmup_steps)) * 2 * self.mask_prob
            elif self.masking_mode == 'poly_desc': 
                mask_prob = (1 - 2 * steps / self.max_steps) * 2 * self.mask_prob if steps < self.max_steps / 2 \
                                    else (2 * steps  - self.max_steps) / self.max_steps * 2 * self.mask_prob
                mask_prob += 0.01
            elif self.masking_mode == 'cosine':
                mask_prob = self.mask_prob * (1 + math.cos(math.pi / self.max_steps * steps)) + 0.02
            else:
                turn = self.max_steps * 0.4
                mask_prob = 0.2 + 0.1 * math.cos(math.pi / turn * steps) if steps < turn \
                        else -(0.5 / self.max_steps) ** 2 * (steps - turn) ** 2 + 0.1
        else:
            mask_prob = self.mask_prob

        return mask_prob







