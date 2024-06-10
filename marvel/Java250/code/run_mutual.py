# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification,
                          T5Config, T5ForConditionalGeneration)
from tqdm import tqdm, trange
import multiprocessing
from model import CodeBERTnoise, GraphCodeBERTnoise, UniXCodernoise, CodeBERT_twoContact, GraphCodeBERT_twoContact, UniXCoder_twoContact

cpu_cont = 16
logger = logging.getLogger(__name__)

sys.path.append("../../../")

from language_parser.run_parser import get_identifiers_from_tokens

from tree_sitter import Language, Parser
from datetime import datetime
from load_data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

MODEL_CLASSES = {
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer, CodeBERTnoise, CodeBERT_twoContact),
    'graphcodebert': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, GraphCodeBERTnoise, GraphCodeBERT_twoContact),
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer, UniXCodernoise, UniXCoder_twoContact),
}


def single_train(model1, model2, noise, inputs_ids, attn_mask, position_idx, labels, args, former_noise=True):
    # 互相学习, model1向model2学习
    if former_noise:
        if args.model_type == 'codebert':
            loss1, c_prob1, c_outputs1, attetnions1 = model1(inputs_ids,labels,noise=noise)
            loss2, c_prob2, c_outputs2, attentions2 = model2(inputs_ids,labels)
        elif args.model_type == 'graphcodebert':
            loss1, c_prob1, c_outputs1, attentions1 = model1(inputs_ids,attn_mask,position_idx,labels, noise=noise)
            loss2, c_prob2, c_outputs2, attentions2 = model2(inputs_ids,attn_mask,position_idx,labels)    
        elif args.model_type == 'unixcoder':
            loss1, c_prob1, c_outputs1, attetnions1 = model1(inputs_ids,labels,noise=noise)
            loss2, c_prob2, c_outputs2, attentions2 = model2(inputs_ids,labels)   
    else:
        if args.model_type == 'codebert':
            loss1, c_prob1, c_outputs1, attetnions1 = model1(inputs_ids,labels)
            loss2, c_prob2, c_outputs2, attentions2 = model2(inputs_ids,labels,noise=noise)
        elif args.model_type == 'graphcodebert':
            loss1, c_prob1, c_outputs1, attentions1 = model1(inputs_ids,attn_mask,position_idx,labels)
            loss2, c_prob2, c_outputs2, attentions2 = model2(inputs_ids,attn_mask,position_idx,labels, noise=noise)        
        elif args.model_type == 'unixcoder':
            loss1, c_prob1, c_outputs1, attetnions1 = model1(inputs_ids,labels)
            loss2, c_prob2, c_outputs2, attentions2 = model2(inputs_ids,labels,noise=noise)
    # return loss1
    loss_hard = loss1

    loss_soft = 0 
    kl_div2 = nn.functional.kl_div(c_prob1.log(), c_prob2, reduction='sum')
    loss_soft += kl_div2
    
    if args.alpha > 0:
        loss = args.alpha * loss_soft + (1 - args.alpha) * loss_hard
        # loss = (loss / (1 - args.alpha))
    else:
        loss = loss_hard + loss_soft
    return loss

def calculate_noise(model, inputs_ids, attentions, args):
    if args.model_type == 'codebert':
        embeddings = model.encoder.embeddings.word_embeddings(inputs_ids)
    elif args.model_type == 'graphcodebert':
        embeddings = model.encoder.roberta.embeddings.word_embeddings(inputs_ids)
    elif args.model_type == 'unixcoder':
        embeddings = model.encoder.embeddings.word_embeddings(inputs_ids)
    bsz, seqlen, hsz = embeddings.shape
    noise = torch.zeros_like(embeddings).to(args.device) # (bsz, seqlen, hsz)

    code_tokens = [model.tokenizer.convert_ids_to_tokens(inputs_ids[i]) for i in range(bsz)]

    identifiers = [get_identifiers_from_tokens(code_token, args.language_type) for code_token in code_tokens]
    
    token_index_list = [] #(bsz, token_num)
    token_index_map_list = [] #(bsz, token_num)
    attention_rank_list = [] #(bsz, token_num)，存储对应 token 位置的 attention 排名

    for batch_id in range(bsz):
        # 获取每个标识符的位置
        token_index_list.append([])
        token_index_map_list.append({})
        attention_rank_list.append({})
        temp_code_token_list = code_tokens[batch_id]
        code_token_list = []
        for code_token in temp_code_token_list:
            if len(code_token) > 1 and code_token[0] == 'Ġ':
                code_token_list.append(code_token[1:])
            else:
                code_token_list.append(code_token)

        identifiers_list = identifiers[batch_id]
        identifiers_list.remove('Ġ') if 'Ġ' in identifiers_list else None
        
        for identifier in identifiers_list:
            indexex = [i for i in range(len(code_token_list) - 1) if code_token_list[i] == identifier]
            token_index_map_list[batch_id][identifier] = indexex
                                        
        for identifier in identifiers_list:
            attention_val = 0
            for layer in attentions[batch_id]:
                temp_val = 0
                for idn in token_index_map_list[batch_id][identifier]:
                    temp_val += layer[idn][idn].item()
                attention_val += (temp_val / len(token_index_map_list[batch_id][identifier]))
            for idn in token_index_map_list[batch_id][identifier]:
                attention_rank_list[batch_id][idn] = attention_val
    
        attention_rank_list[batch_id] = sorted(attention_rank_list[batch_id].items(), key=lambda x: x[1], reverse=True)
    random_noise = torch.randn(hsz).to(args.device)
    for batch_id in range(bsz):
        attentions_values = [t[1] for t in attention_rank_list[batch_id]]
        if len(attentions_values) > 0:
            average = sum(attentions_values) / len(attentions_values)
            for id, weight in attention_rank_list[batch_id]:
                cur_noise = random_noise * (weight * 1 / average + 1)
                noise[batch_id][id] += cur_noise   
    return noise

def train(args, train_dataset, tokenizer, model1, model2,pool):
    """Train the model by our method"""
    #build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // 2
    args.warmup_steps =args.max_steps // 5
    model1.to(args.device)
    model2.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in model1.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model1.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in model2.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model2.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        model1 = torch.nn.DataParallel(model1)
        model2 = torch.nn.DataParallel(model2)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    
    best_acc1 = 0.0
    best_acc2 = 0.0

    model1.zero_grad()
    model2.zero_grad()

    loss_list = []
    avg_loss = 0
    for idx in range(args.epochs):
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            if args.model_type == 'codebert':
                inputs_ids = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                model1.train()
                model2.train()

                _,_,_, attentions = model1(inputs_ids,labels)
                
            elif args.model_type == 'graphcodebert':
                inputs_ids = batch[0].to(args.device)
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                labels=batch[3].to(args.device)
                model1.train()
                model2.train()

                _,_,_, attentions = model1(inputs_ids,attn_mask,position_idx,labels)

            elif args.model_type == 'unixcoder':
                inputs_ids = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                model1.train()
                model2.train()

                _,_,_, attentions = model1(inputs_ids,labels)  
            noise = calculate_noise(model1, inputs_ids, attentions, args)

            if args.model_type == 'codebert' or args.model_type == 'unixcoder':
                attn_mask = None
                position_idx = None

            for step_adv in range(args.max_adv_step):
                noise.requires_grad_()
                
                # model1 加噪, model1 向 model2 学习
                train_loss1 = single_train(model1, model2, noise, inputs_ids, attn_mask, position_idx, labels, args, former_noise=True)
                
                if step_adv == args.max_adv_step - 1:
                    break

                if args.n_gpu > 1:
                    train_loss1 = train_loss1.mean()

                if args.gradient_accumulation_steps > 1:
                    train_loss1 = train_loss1 / args.gradient_accumulation_steps

                train_loss1.backward()
                torch.nn.utils.clip_grad_norm_(model1.parameters(), args.max_grad_norm)                 
                
                noise_grad = noise.grad.clone().detach()
                denorm = torch.norm(noise_grad.view(noise_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                noise = (noise + args.adv_lr * noise_grad / denorm).detach()
                

            if args.n_gpu > 1:
                train_loss1 = train_loss1.mean()

            if args.gradient_accumulation_steps > 1:
                train_loss1 = train_loss1 / args.gradient_accumulation_steps

            train_loss1.backward()
            torch.nn.utils.clip_grad_norm_(model1.parameters(), args.max_grad_norm)
            
            optimizer1.step()
            optimizer1.zero_grad()
            scheduler1.step()
                
            # 还是给 model1 加噪声, model2 向 model1 学习
            train_loss2 = single_train(model2, model1, noise, inputs_ids, attn_mask, position_idx, labels, args, former_noise=False)

            if args.n_gpu > 1:
                train_loss2 = train_loss2.mean()

            if args.gradient_accumulation_steps > 1:
                train_loss2 = train_loss2 / args.gradient_accumulation_steps

            train_loss2.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), args.max_grad_norm)

            optimizer2.step()
            optimizer2.zero_grad()
            scheduler2.step()
            
            cur_loss = (train_loss1 + train_loss2).item() / 2

            tr_num += 1
            train_loss += cur_loss
            avg_loss = round(train_loss/tr_num,5)           
            bar.set_description("epoch {}, avg_loss {}, cur_loss {}".format(idx,avg_loss,cur_loss))
            
            global_step += 1

            if global_step % args.save_steps == 0:
                results1 = evaluate_model(args, model1, tokenizer, eval_when_training=True,new_infer=True,pool=pool)
                results2 = evaluate_model(args, model2, tokenizer, eval_when_training=True,pool=pool)
                # Save model checkpoint
                if results1['eval_acc'] > best_acc1:
                    best_acc1=results1['eval_acc']
                    logger.info("  "+"*"*20)
                    logger.info(" Model1 Best Acc:%s",round(best_acc1,4))
                    logger.info("  "+"*"*20)

                    checkpoint_prefix = 'checkpoint-best-acc'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model_to_save1 = model1.module if hasattr(model1,'module') else model1
                    output_dir1 = os.path.join(output_dir, '{}_{}_model1.bin'.format(args.model_type, args.save_name))
                    torch.save(model_to_save1.state_dict(), output_dir1)
                    logger.info("Saving model1 checkpoint to %s", output_dir1)

                if results2['eval_acc'] > best_acc2:
                    best_acc2=results2['eval_acc']
                    logger.info("  "+"*"*20)
                    logger.info(" Model2 Best Acc:%s",round(best_acc2,4))
                    logger.info("  "+"*"*20)  
                    
                    checkpoint_prefix = 'checkpoint-best-acc'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                            
                    model_to_save2 = model2.module if hasattr(model2,'module') else model2
                    output_dir2 = os.path.join(output_dir, '{}_{}_model2.bin'.format(args.model_type, args.save_name))
                    torch.save(model_to_save2.state_dict(), output_dir2)
                    logger.info("Saving model2 checkpoint to %s", output_dir2)

        loss_list.append(avg_loss)


def evaluate_model(args, model, tokenizer,eval_when_training=False,new_infer=False,contact=False,pool=None):
    #build dataloader
    if args.model_type == 'codebert':
        eval_dataset = CodeBertTextDataset(tokenizer, args, file_path=args.eval_data_file,pool=pool)
    elif args.model_type == 'graphcodebert':
        eval_dataset = GraphCodeBertTextDataset(tokenizer, args, file_path=args.eval_data_file,pool=pool)
    elif args.model_type == 'unixcoder':
        eval_dataset = UniXCoderTextDataset(tokenizer, args, file_path=args.eval_data_file,pool=pool)        
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=4)
    model.to(args.device)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    nb_eval_steps = 0
    model.eval()
    
    model.zero_grad()
    
    logits=[]
    y_trues=[]
    # bar = tqdm(eval_dataloader,total=len(eval_dataloader))
    # for step, batch in enumerate(bar):
    for batch in eval_dataloader:
        if args.model_type == 'codebert':
            inputs_ids = batch[0].to(args.device)
            label = batch[1].to(args.device)
            if new_infer:
                if contact:
                    _,_,_, attentions = model.model1(inputs_ids,label)
                    noise = calculate_noise(model.model1, inputs_ids, attentions, args)
                else:
                    _,_,_, attentions = model(inputs_ids,label)
                    noise = calculate_noise(model, inputs_ids, attentions, args)
            with torch.no_grad():
                prob, c_logit = model(inputs_ids, noise=noise if new_infer else None)
        elif args.model_type == 'graphcodebert':
            inputs_ids = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            label = batch[3].to(args.device)
            if new_infer:
                if contact:
                    _,_,_, attentions = model.model1(inputs_ids,attn_mask,position_idx,label)
                    noise = calculate_noise(model.model1, inputs_ids, attentions, args)
                else:
                    _,_,_, attentions = model(inputs_ids,attn_mask,position_idx,label)
                    noise = calculate_noise(model, inputs_ids, attentions, args)
            with torch.no_grad():
                prob, c_logit = model(inputs_ids,attn_mask,position_idx,noise=noise if new_infer else None)
        elif args.model_type == 'unixcoder':
            inputs_ids = batch[0].to(args.device)
            label = batch[1].to(args.device)
            if new_infer:
                if contact:
                    _,_,_, attentions = model.model1(inputs_ids,label)
                    noise = calculate_noise(model.model1, inputs_ids, attentions, args)
                else:
                    _,_,_, attentions = model(inputs_ids,label)
                    noise = calculate_noise(model, inputs_ids, attentions, args)
            with torch.no_grad():
                prob, c_logit = model(inputs_ids, noise=noise if new_infer else None)                           
        logits.append(prob.cpu().numpy())
        y_trues.append(label.cpu().numpy())
        nb_eval_steps += 1

    #calculate scores
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)

    y_preds = []
    for logit in logits:
        y_preds.append(np.argmax(logit))

    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds, average='macro')   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds, average='macro') 
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_trues,y_preds)

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_acc": float(acc)
    }
    
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer,eval_when_training=False,new_infer=False, pool=None):
    #build dataloader
    if args.model_type == 'codebert':
        eval_dataset = CodeBertTextDataset(tokenizer, args, file_path=args.test_data_file, pool=pool)
    elif args.model_type == 'graphcodebert':
        eval_dataset = GraphCodeBertTextDataset(tokenizer, args, file_path=args.test_data_file, pool=pool)
    elif args.model_type == 'unixcoder':
        eval_dataset = UniXCoderTextDataset(tokenizer, args, file_path=args.test_data_file, pool=pool)
                
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=4)
    model.to(args.device)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    nb_eval_steps = 0
    model.eval()
    
    model.zero_grad()
    
    logits=[]
    y_trues=[]
    # bar = tqdm(eval_dataloader,total=len(eval_dataloader))
    # for step, batch in enumerate(bar):
    for batch in eval_dataloader:
        if args.model_type == 'codebert':
            inputs_ids = batch[0].to(args.device)
            label=batch[1].to(args.device)
            _,_,_, attentions = model.model1(inputs_ids,label)
            noise = calculate_noise(model.model1, inputs_ids, attentions, args)
            with torch.no_grad():
                prob, c_logit = model(inputs_ids, noise=noise)    
        elif args.model_type == 'graphcodebert':
            inputs_ids = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            label=batch[3].to(args.device)
            _,_,_, attentions = model.model1(inputs_ids,attn_mask,position_idx,label)
            noise = calculate_noise(model.model1, inputs_ids, attentions, args)
            with torch.no_grad():
                prob, c_logit = model(inputs_ids,attn_mask,position_idx,noise=noise)           
        elif args.model_type == 'unixcoder':
            inputs_ids = batch[0].to(args.device)
            label=batch[1].to(args.device)
            _,_,_, attentions = model.model1(inputs_ids,label)
            noise = calculate_noise(model.model1, inputs_ids, attentions, args)
            with torch.no_grad():
                prob, c_logit = model(inputs_ids, noise=noise) 
                
        logits.append(prob.cpu().numpy())
        y_trues.append(label.cpu().numpy())
        nb_eval_steps += 1

    #calculate scores
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)

    y_preds = []
    for logit in logits:
        y_preds.append(np.argmax(logit))

    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds, average='macro')   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds, average='macro') 
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_trues,y_preds)

    result = {
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_acc": float(acc)
    }
    
    logger.info("***** test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    print("Accuracy in test set:", result['test_acc'])

    return result


def train_contact(args, train_dataset, tokenizer, model_contact, pool=None):
    """Train the model by our method"""
    #build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps =args.max_steps // 5
    model_contact.to(args.device)

    for name, parameter in model_contact.model1.named_parameters():
        parameter.requires_grad = False
    for name, parameter in model_contact.model2.named_parameters():
        parameter.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model_contact.parameters()), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    # multi-gpu training
    if args.n_gpu > 1:
        model_contact = torch.nn.DataParallel(model_contact)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    
    best_acc1 = 0.0

    model_contact.zero_grad()

    avg_loss = 0
    for idx in range(args.epochs):
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            if args.model_type == 'codebert':
                inputs_ids = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                model_contact.train()
                _,_,_, attentions = model_contact.model1(inputs_ids,labels)
                noise = calculate_noise(model_contact.model1, inputs_ids, attentions, args)
                loss, probs = model_contact(inputs_ids,labels,noise)
            elif args.model_type == 'graphcodebert':
                inputs_ids = batch[0].to(args.device)
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                labels=batch[3].to(args.device)
                model_contact.train()
                _,_,_, attentions = model_contact.model1(inputs_ids,attn_mask,position_idx,labels)
                noise = calculate_noise(model_contact.model1, inputs_ids, attentions, args)
                loss, probs = model_contact(inputs_ids,attn_mask,position_idx,labels,noise)
            elif args.model_type == 'unixcoder':
                inputs_ids = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                model_contact.train()
                _,_,_, attentions = model_contact.model1(inputs_ids,labels)
                noise = calculate_noise(model_contact.model1, inputs_ids, attentions, args)
                loss, probs = model_contact(inputs_ids,labels,noise)
                
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_contact.parameters(), args.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            cur_loss = loss.item()

            tr_num += 1
            train_loss += cur_loss
            avg_loss = round(train_loss/tr_num,5)           
            bar.set_description("epoch {}, avg_loss {}, cur_loss {}".format(idx,avg_loss,cur_loss))
            
            global_step += 1

            if global_step % args.save_steps == 0:
                results1 = evaluate_model(args, model_contact, tokenizer, eval_when_training=True, new_infer=True, contact=True, pool=pool)
                # Save model checkpoint
                if results1['eval_acc'] >= best_acc1:
                    best_acc1=results1['eval_acc']
                    logger.info("  "+"*"*20)
                    logger.info(" Model Best Acc:%s",round(best_acc1,4))
                    logger.info("  "+"*"*20)

                    checkpoint_prefix = 'checkpoint-best-acc'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model_to_save1 = model_contact.module if hasattr(model_contact,'module') else model_contact
                    output_dir1 = os.path.join(output_dir, '{}_{}_model.bin'.format(args.model_type, args.save_name))
                    torch.save(model_to_save1.state_dict(), output_dir1)
                    logger.info("Saving model checkpoint to %s", output_dir1)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default='../dataset/train.txt', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='../model/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default='../dataset/valid.txt', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default='../dataset/test.txt', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default='graphcodebert', type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--number_labels", default=250, type=int,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--code_length", default=384, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=128, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--language_type", type=str, default="python",
                        help="The programming language type of dataset")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--adv_lr', type=float, default=1e-4)

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--save_name", type=str,
                        help="Whether learn attention or not.")
    parser.add_argument("--max_adv_step", default=5, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--alpha", default=-1, type=float,
                        help="mutual learning.")
    
    pool = multiprocessing.Pool(cpu_cont)
    
    args = parser.parse_args()

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S', 
        level=logging.INFO
    )

    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    # Set seed
    set_seed(args)
    args.block_size = 512
    args.output_dir += args.model_type
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.model_type == 'codebert':
        args.config_name = "microsoft/codebert-base"
        args.model_name_or_path = "microsoft/codebert-base"
        args.tokenizer_name = "microsoft/codebert-base"
        args.learning_rate = 5e-5
        args.code_length = 512
        args.data_flow_length = 0    
    elif args.model_type == "graphcodebert":
        args.config_name = "microsoft/graphcodebert-base"
        args.model_name_or_path = "microsoft/graphcodebert-base"
        args.tokenizer_name = "microsoft/graphcodebert-base"
        args.code_length = 384
        args.data_flow_length = 128
        args.learning_rate = 2e-5
    elif args.model_type == "unixcoder":
        args.config_name = "microsoft/unixcoder-base"
        args.model_name_or_path = "microsoft/unixcoder-base"
        args.tokenizer_name = "microsoft/unixcoder-base"
        args.code_length = 512
        args.data_flow_length = 0    
        args.learning_rate = 2e-5  
        
    config_class, model_class, tokenizer_class, Model, Model_Contact = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels=250
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model_ = model_class.from_pretrained(args.model_name_or_path,config=config)   

    model=Model(model_,config,tokenizer,args)
    model1=Model(model_,config,tokenizer,args)
    model2=Model(model_,config,tokenizer,args)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.model_type == 'codebert':
            train_dataset = CodeBertTextDataset(model.tokenizer, args, args.train_data_file, pool=pool)
        elif args.model_type == 'graphcodebert':
            train_dataset = GraphCodeBertTextDataset(model.tokenizer, args, args.train_data_file, pool=pool)
        elif args.model_type == 'unixcoder':
            train_dataset = UniXCoderTextDataset(model.tokenizer, args, args.train_data_file, pool=pool)
                        
        print("Begin training")
        train(args, train_dataset, model.tokenizer, model1, model2, pool=pool)
        model_contact = Model_Contact(config, model1, model2, args)
        train_contact(args, train_dataset, model.tokenizer, model_contact, pool=pool)

    # Evaluation
    if args.do_eval:
        model_contact = Model_Contact(config, model1, model2, args)
        checkpoint_prefix = 'checkpoint-best-acc/{}_{}_model.bin'.format(args.model_type, args.save_name)
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model_contact.load_state_dict(torch.load(output_dir))
        model_contact.to(args.device)
        result=evaluate_model(args, model_contact, model.tokenizer, pool=pool)

    if args.do_test:
        model_contact = Model_Contact(config, model1, model2, args)
        checkpoint_prefix = 'checkpoint-best-acc/{}_{}_model.bin'.format(args.model_type, args.save_name)
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model_contact.load_state_dict(torch.load(output_dir, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")),strict=False)
        model_contact.to(args.device)
        test(args, model_contact, tokenizer, new_infer=True, pool=pool)



if __name__ == "__main__":
    main()
