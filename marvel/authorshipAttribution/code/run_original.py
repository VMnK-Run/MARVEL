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
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification,
                          T5Config, T5ForConditionalGeneration)
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,)
from tqdm import tqdm, trange
import multiprocessing
from model import CodeBERT, GraphCodeBERT, UniXCoder

cpu_cont = 16
logger = logging.getLogger(__name__)

import sys
sys.path.append("../../../")

from tree_sitter import Language, Parser

from load_data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

MODEL_CLASSES = {
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer, CodeBERT),
    'graphcodebert': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, GraphCodeBERT),
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer, UniXCoder)
}


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    
    #build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    args.max_steps=args.epochs*len(train_dataloader)
    args.save_steps=len(train_dataloader)
    args.warmup_steps=args.max_steps//5
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

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
    best_f1=0

    model.zero_grad()
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            if args.model_type == 'codebert':
                inputs = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                model.train()
                loss, logits = model(inputs, labels)
            elif args.model_type == 'graphcodebert':
                inputs_ids = batch[0].to(args.device)
                attn_mask = batch[1].to(args.device) 
                position_idx = batch[2].to(args.device) 
                labels=batch[3].to(args.device) 
                model.train()
                loss,logits = model(inputs_ids,attn_mask,position_idx,labels)
            elif args.model_type == 'unixcoder':
                inputs = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                model.train()
                loss, logits = model(inputs, labels)                

            if args.n_gpu > 1:
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
                
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1

                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_acc']>best_f1:
                        best_f1=results['eval_acc']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best acc:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}_{}_model.bin'.format(args.model_type, args.save_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        

def evaluate(args, model, tokenizer,eval_when_training=False):
    #build dataloader
    if args.model_type == 'codebert':
        eval_dataset = CodeBertTextDataset(tokenizer, args, file_path=args.eval_data_file)
    elif args.model_type == 'graphcodebert':
        eval_dataset = GraphCodeBertTextDataset(tokenizer, args, file_path=args.eval_data_file)
    elif args.model_type == 'unixcoder':
        eval_dataset = UniXCoderTextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        if args.model_type == 'codebert':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit = model(inputs, labels)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())
        elif args.model_type == 'graphcodebert':            
            inputs_ids = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device) 
            position_idx = batch[2].to(args.device) 
            label=batch[3].to(args.device) 
            with torch.no_grad():
                lm_loss,logit = model(inputs_ids, attn_mask, position_idx, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_trues.append(label.cpu().numpy())
        elif args.model_type == 'unixcoder':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit = model(inputs, labels)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
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

    # logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer):
    #build dataloader
    if args.model_type == 'codebert':
        eval_dataset = CodeBertTextDataset(tokenizer, args, file_path=args.eval_data_file)
    elif args.model_type == 'graphcodebert':
        eval_dataset = GraphCodeBertTextDataset(tokenizer, args, file_path=args.eval_data_file)
    elif args.model_type == 'unixcoder':
        eval_dataset = UniXCoderTextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in tqdm(eval_dataloader):
        if args.model_type == 'codebert':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit = model(inputs, labels)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())
        elif args.model_type == 'graphcodebert':            
            inputs_ids = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device) 
            position_idx = batch[2].to(args.device) 
            label=batch[3].to(args.device) 
            with torch.no_grad():
                lm_loss,logit = model(inputs_ids, attn_mask, position_idx, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_trues.append(label.cpu().numpy())
        elif args.model_type == 'unixcoder':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit = model(inputs, labels)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    #output result
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
        "test_acc":float(acc)
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

                   
                        
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
    parser.add_argument("--test_data_file", default='../dataset/valid.txt', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default=None, type=str,
                        help="The model type.")
    parser.add_argument("--save_name", default=None, type=str,
                        help="The model type.")        
    parser.add_argument("--number_labels", default=66, type=int,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=448, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--language_type", type=str, default='python',
                        help="The programming language type of dataset")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")


    args = parser.parse_args()

    args.output_dir += args.model_type
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    # Set seed  
    set_seed(args)

    args.block_size = 512
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
        args.code_length = 448
        args.data_flow_length = 64
        args.learning_rate = 2e-5
    elif args.model_type == "unixcoder":
        args.config_name = "microsoft/unixcoder-base"
        args.model_name_or_path = "microsoft/unixcoder-base"
        args.tokenizer_name = "microsoft/unixcoder-base"
        args.code_length = 512
        args.data_flow_length = 0      
        args.learning_rate = 2e-5

    config_class, model_class, tokenizer_class, Model = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name)
    config.num_labels=args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model = model_class.from_pretrained(args.model_name_or_path,config=config)
    
    model=Model(model,config,tokenizer,args)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.model_type == 'codebert':
            train_dataset = CodeBertTextDataset(tokenizer, args, args.train_data_file)
        elif args.model_type == 'graphcodebert':
            train_dataset = GraphCodeBertTextDataset(tokenizer, args,args.train_data_file)
        elif args.model_type == 'unixcoder':
            train_dataset = UniXCoderTextDataset(tokenizer, args,args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-acc/{}_{}_model.bin'.format(args.model_type, args.save_name)
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))    
        model.load_state_dict(torch.load(output_dir))         
        model.to(args.device)
        result=evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/{}_{}_model.bin'.format(args.model_type, args.save_name)
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))                                   
        model.to(args.device)
        result=test(args, model, tokenizer)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))


if __name__ == "__main__":
    main()
