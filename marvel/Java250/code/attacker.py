import sys
import os

import numpy as np
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../language_parser')
import time
import copy
import torch
import random
from load_data import GraphCodeBertInputFeatures, CodeBertInputFeatures, UniXCoderInputFeatures
from utils.utils_alert import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, isUID, get_substitues, is_valid_substitue, set_seed
from utils.utils_alert import CodeDataset, GraphCodeDataset, UniXCoderDataset #, get_edits_similarity

from language_parser import DFG_python, DFG_java, DFG_c
from language_parser import get_identifiers, get_example
from language_parser import get_identifiers_c, get_example_batch,get_example_batch_coda, get_code_style, change_code_style, remove_comments_and_docstrings
from language_parser.parse_util import extract_dataflow

from scipy.spatial.distance import cosine as cosine_distance
from tree_sitter import Language, Parser
from torch.utils.data import Dataset
dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c
}

#load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('../../../language_parser/parser_folder/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


class GraphCodeDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args=args
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length,
                            self.args.code_length+self.args.data_flow_length),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True
        
        code_tokens_return = self.examples[item].code_tokens
        if len(code_tokens_return) < self.args.code_length + self.args.data_flow_length:
            padding = [0] * (self.args.code_length + self.args.data_flow_length - len(code_tokens_return))
            code_tokens_return += padding

        return (torch.tensor(self.examples[item].code_ids),
              torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx),
              torch.tensor(self.examples[item].label))

def graphcodebert_compute_fitness(chromesome, codebert_tgt, tokenizer_tgt, orig_prob, orig_label, true_label ,code, names_positions_dict, args):
    # 计算fitness function.
    # words + chromesome + orig_label + current_prob
    temp_code = map_chromesome(chromesome, code, "java")
    new_feature = graphcodebert_convert_code_to_features(temp_code, tokenizer_tgt, true_label, args)
    new_dataset = GraphCodeDataset([new_feature], args)
    new_logits, preds = codebert_tgt.get_results(new_dataset, args.eval_batch_size)
    # 计算fitness function
    fitness_value = orig_prob - new_logits[0][orig_label]
    return fitness_value, preds[0]

def codebert_compute_fitness(chromesome, codebert_tgt, tokenizer_tgt, orig_prob, orig_label, true_label ,code, names_positions_dict, args):
    # 计算fitness function.
    # words + chromesome + orig_label + current_prob
    temp_code = map_chromesome(chromesome, code, "java")
    new_feature = codebert_convert_code_to_features(temp_code, tokenizer_tgt, true_label, args)
    new_dataset = CodeDataset([new_feature])
    new_logits, preds = codebert_tgt.get_results(new_dataset, args.eval_batch_size)
    # 计算fitness function
    fitness_value = orig_prob - new_logits[0][orig_label]
    return fitness_value, preds[0]

def unixcoder_compute_fitness(chromesome, codebert_tgt, tokenizer_tgt, orig_prob, orig_label, true_label ,code, names_positions_dict, args):
    # 计算fitness function.
    # words + chromesome + orig_label + current_prob
    temp_code = map_chromesome(chromesome, code, "java")
    new_feature = unixcoder_convert_code_to_features(temp_code, tokenizer_tgt, true_label, args)
    new_dataset = UniXCoderDataset([new_feature])
    new_logits, preds = codebert_tgt.get_results(new_dataset, args.eval_batch_size)
    # 计算fitness function
    fitness_value = orig_prob - new_logits[0][orig_label]
    return fitness_value, preds[0]

def graphcodebert_convert_code_to_features(code, tokenizer, label, args):
    # 这里要被修改..
    parser = parsers["java"]
    code_tokens, dfg = extract_dataflow(code, parser, "java")
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    return GraphCodeBertInputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, label)

def codebert_convert_code_to_features(code, tokenizer, label, args):
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return CodeBertInputFeatures(source_tokens, source_ids,label)

def unixcoder_convert_code_to_features(code, tokenizer, label, args):
    """convert examples to token ids"""
    code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
    source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return UniXCoderInputFeatures(source_tokens, source_ids, 0, label)

def graphcodebert_get_importance_score(args, example, code, words_list: list, sub_words: list, variable_names: list, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification', invocation_number=0):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None, invocation_number

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.
    for index, tokens in enumerate([words_list] + masked_token_list):
        invocation_number += 1
        new_code = ' '.join(tokens)
        new_feature = graphcodebert_convert_code_to_features(new_code, tokenizer, example[3].item(), args)
        new_example.append(new_feature)
    new_dataset = GraphCodeDataset(new_example, args)
    # 3. 将他们转化成features
    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.

    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])
    return importance_score, replace_token_positions, positions, invocation_number

def codebert_get_importance_score(args, example, code, words_list: list, sub_words: list, variable_names: list, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification', invocation_number=0):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None, invocation_number

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.


    for index, tokens in enumerate([words_list] + masked_token_list):
        invocation_number += 1
        new_code = ' '.join(tokens)
        new_feature = codebert_convert_code_to_features(new_code, tokenizer, example[1].item(), args)
        new_example.append(new_feature)
    new_dataset = CodeDataset(new_example)
    # 3. 将他们转化成features
    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.
    
    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions, invocation_number

def unixcoder_get_importance_score(args, example, code, words_list: list, sub_words: list, variable_names: list, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification', invocation_number=0):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None, invocation_number

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.


    for index, tokens in enumerate([words_list] + masked_token_list):
        invocation_number += 1
        new_code = ' '.join(tokens)
        new_feature = unixcoder_convert_code_to_features(new_code, tokenizer, example[1].item(), args)
        new_example.append(new_feature)
    new_dataset = UniXCoderDataset(new_example)
    # 3. 将他们转化成features
    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.
    
    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions, invocation_number

class AlertAttacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.model_mlm = model_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score

    def ga_attack(self, example, code, subs, initial_replace=None):
        # 先得到tgt_model针对原始Example的预测信息.
        invocation_number = 1
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)
        if self.args.model_name == 'codebert':
            true_label = example[1].item()
        elif self.args.model_name == 'graphcodebert':
            true_label = example[3].item()
        elif self.args.model_name == 'unixcoder':
            true_label = example[1].item()
        adv_code = ''
        temp_label = None

        identifiers, code_tokens = get_identifiers(code, 'java')
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)
        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        # 这里经过了小写处理..

        variable_names = list(subs.keys())

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, invocation_number, None
            
        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, invocation_number, None

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)
        nb_changed_var = 0 # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        most_gap = 0.0

        # 我们可以先生成所有的substitues
        variable_substitue_dict = {}

        for tgt_word in names_positions_dict.keys():
            variable_substitue_dict[tgt_word] = subs[tgt_word]

        fitness_values = []
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        # 关于chromesome的定义: {tgt_word: candidate, tgt_word_2: candidate_2, ...}
        for tgt_word in variable_substitue_dict.keys():
            # 这里进行初始化
            if initial_replace is None:
                # 对于每个variable: 选择"影响最大"的substitues
                replace_examples = []
                substitute_list = []
                
                current_prob = max(orig_prob)
                most_gap = 0.0
                initial_candidate = tgt_word
                tgt_positions = names_positions_dict[tgt_word]
                
                # 原来是随机选择的，现在要找到改变最大的.
                for a_substitue in variable_substitue_dict[tgt_word]:
                    # a_substitue = a_substitue.strip()
                    
                    substitute_list.append(a_substitue)
                    # 记录下这次换的是哪个substitue
                    temp_code = get_example(code, tgt_word, a_substitue, "java")
                    if self.args.model_name == 'codebert':
                        new_feature = codebert_convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
                    elif self.args.model_name == 'graphcodebert':
                        new_feature = graphcodebert_convert_code_to_features(temp_code, self.tokenizer_tgt, example[3].item(), self.args)
                    elif self.args.model_name == 'unixcoder':
                        new_feature = unixcoder_convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
                    replace_examples.append(new_feature)

                if len(replace_examples) == 0:
                    # 并没有生成新的mutants，直接跳去下一个token
                    continue
                if self.args.model_name == 'codebert':
                    new_dataset = CodeDataset(replace_examples)
                elif self.args.model_name == 'graphcodebert':
                    new_dataset = GraphCodeDataset(replace_examples, self.args)
                elif self.args.model_name == 'unixcoder':
                    new_dataset = UniXCoderDataset(replace_examples)
                # 3. 将他们转化成features
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

                _the_best_candidate = -1
                for index, temp_prob in enumerate(logits):
                    invocation_number += 1
                    temp_label = preds[index]
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        _the_best_candidate = index
                if _the_best_candidate == -1:
                    initial_candidate = tgt_word
                else:
                    initial_candidate = substitute_list[_the_best_candidate]
            else:
                initial_candidate = initial_replace[tgt_word]

            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)

            if self.args.model_name == 'codebert':
                invocation_number += 1
                temp_fitness, temp_label = codebert_compute_fitness(temp_chromesome, self.model_tgt, self.tokenizer_tgt, max(orig_prob), orig_label, true_label ,code, names_positions_dict, self.args)
            elif self.args.model_name == 'graphcodebert':
                invocation_number += 1
                temp_fitness, temp_label = graphcodebert_compute_fitness(temp_chromesome, self.model_tgt, self.tokenizer_tgt,
                                                           max(orig_prob), orig_label, true_label, code,
                                                           names_positions_dict, self.args)
            elif self.args.model_name == 'unixcoder':
                invocation_number += 1
                temp_fitness, temp_label = unixcoder_compute_fitness(temp_chromesome, self.model_tgt, self.tokenizer_tgt,
                                                                    max(orig_prob), orig_label, true_label, code,
                                                                    names_positions_dict, self.args)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 10)
        # 这里的超参数还是的调试一下.

        for i in range(max_iter):
            _temp_mutants = []
            for j in range(self.args.eval_batch_size):
                p = random.random()
                chromesome_1, index_1, chromesome_2, index_2 = select_parents(population)
                if p < cross_probability: # 进行crossover
                    if chromesome_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                        continue
                    child_1, child_2 = crossover(chromesome_1, chromesome_2)
                    if child_1 == chromesome_1 or child_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                else: # 进行mutates
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
                _temp_mutants.append(child_1)
            # compute fitness in batch
            feature_list = []
            for mutant in _temp_mutants:
                _temp_code = map_chromesome(mutant, code, "java")
                if self.args.model_name == 'codebert':
                    _tmp_feature = codebert_convert_code_to_features(_temp_code, self.tokenizer_tgt, true_label, self.args)
                elif self.args.model_name == 'graphcodebert':
                    _tmp_feature = graphcodebert_convert_code_to_features(_temp_code, self.tokenizer_tgt, true_label, self.args)
                elif self.args.model_name == 'unixcoder':
                    _tmp_feature = unixcoder_convert_code_to_features(_temp_code, self.tokenizer_tgt, true_label, self.args)
                feature_list.append(_tmp_feature)
            if len(feature_list) == 0:
                continue
            if self.args.model_name == 'codebert':
                new_dataset = CodeDataset(feature_list)
            elif self.args.model_name == 'graphcodebert':
                new_dataset = GraphCodeDataset(feature_list, self.args)
            elif self.args.model_name == 'unixcoder':
                new_dataset = UniXCoderDataset(feature_list)
            mutate_logits, mutate_preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            mutate_fitness_values = []
            for index, logits in enumerate(mutate_logits):
                invocation_number += 1
                if mutate_preds[index] != orig_label:
                    if is_success < 1:
                        is_success = 1
                    else:
                        is_success += 1
                    adv_code = map_chromesome(_temp_mutants[index], code, "java")
                    for old_word in _temp_mutants[index].keys():
                        if old_word == _temp_mutants[index][old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])
                    # edits, sim = get_edits_similarity(code, adv_code, self.fasttext_model, self.args.language_type)
                    return code, prog_length, adv_code, true_label, orig_label, mutate_preds[index], 1, variable_names, None, nb_changed_var, nb_changed_pos, _temp_mutants[index], invocation_number, most_gap
                _tmp_fitness = max(orig_prob) - logits[orig_label]
                mutate_fitness_values.append(_tmp_fitness)
            
            # 现在进行替换.
            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    # 替换.
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, None, invocation_number, most_gap

    def greedy_attack(self, example, code, subs):
        # 先得到tgt_model针对原始Example的预测信息.
        invocation_number = 1
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)
        if self.args.model_name == 'codebert':
            true_label = example[1].item()
        elif self.args.model_name == 'graphcodebert':
            true_label = example[3].item()
        elif self.args.model_name == 'unixcoder':
            true_label = example[1].item()
        adv_code = ''
        temp_label = None
        most_gap = 0.0

        identifiers, code_tokens = get_identifiers(code, 'java')
        prog_length = len(code_tokens)
        processed_code = " ".join(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        # 这里经过了小写处理..

        variable_names = list(subs.keys())

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, invocation_number, None
            
        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, invocation_number, None

        sub_words = [self.tokenizer_tgt.cls_token] + sub_words[:self.args.block_size - 2] + [self.tokenizer_tgt.sep_token]
        # 计算importance_score.

        if self.args.model_name == 'codebert':
            importance_score, replace_token_positions, names_positions_dict, invocation_number = codebert_get_importance_score(self.args, example,
                                                processed_code,
                                                words,
                                                sub_words,
                                                variable_names,
                                                self.model_tgt, 
                                                self.tokenizer_tgt, 
                                                [0,1], 
                                                batch_size=self.args.eval_batch_size, 
                                                max_length=self.args.block_size, 
                                                model_type='classification', invocation_number=invocation_number)
        elif self.args.model_name == 'graphcodebert':
            importance_score, replace_token_positions, names_positions_dict, invocation_number = graphcodebert_get_importance_score(self.args, example,
                                                                                                   processed_code,
                                                                                                   words,
                                                                                                   sub_words,
                                                                                                   variable_names,
                                                                                                   self.model_tgt,
                                                                                                   self.tokenizer_tgt,
                                                                                                   [0, 1],
                                                                                                   batch_size=self.args.eval_batch_size,
                                                                                                   max_length=self.args.code_length,
                                                                                                   model_type='classification', invocation_number=invocation_number)
        elif self.args.model_name == 'unixcoder':
            importance_score, replace_token_positions, names_positions_dict, invocation_number = unixcoder_get_importance_score(self.args, example,
                                                processed_code,
                                                words,
                                                sub_words,
                                                variable_names,
                                                self.model_tgt,
                                                self.tokenizer_tgt,
                                                [0,1],
                                                batch_size=self.args.eval_batch_size,
                                                max_length=self.args.block_size,
                                                model_type='classification', invocation_number=invocation_number)

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None, invocation_number, None

        token_pos_to_score_pos = {}
        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                # print(len(importance_score), token_pos_to_score_pos[token_pos])
                total_score += importance_score[token_pos_to_score_pos[token_pos]]
            
            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        # 根据importance_score进行排序

        final_code = copy.deepcopy(code)
        nb_changed_var = 0 # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]

            all_substitues = subs[tgt_word]

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            # 依次记录了被加进来的substitue
            # 即，每个temp_replace对应的substitue.
            for substitute in all_substitues:
                substitute_list.append(substitute)
                # 记录了替换的顺序
                # 需要将几个位置都替换成 sustitue_
                temp_code = get_example(final_code, tgt_word, substitute, "java")
                if self.args.model_name == 'codebert':
                    new_feature = codebert_convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(temp_code, self.tokenizer_tgt, example[3].item(), self.args)
                elif self.args.model_name == 'unixcoder':
                    new_feature = unixcoder_convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                # 并没有生成新的mutants，直接跳去下一个token
                continue
            if self.args.model_name == 'codebert':
                new_dataset = CodeDataset(replace_examples)
            elif self.args.model_name == 'graphcodebert':
                new_dataset = GraphCodeDataset(replace_examples, self.args)
            elif self.args.model_name == 'unixcoder':
                new_dataset = UniXCoderDataset(replace_examples)
                # 3. 将他们转化成features
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert(len(logits) == len(substitute_list))
            for index, temp_prob in enumerate(logits):
                invocation_number += 1
                temp_label = preds[index]

                if temp_label != orig_label:
                    # 如果label改变了，说明这个mutant攻击成功
                    if is_success < 1:
                        is_success = 1
                    else:
                        is_success += 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate, "java")
                    # run_log = open('../log/attack_log/alert_run.log', 'a')
                    # run_log.write("%s SUC! %s => %s (%.5f => %.5f)\n"%('>>', tgt_word, candidate,current_prob,temp_prob[orig_label]))
                    print("%s SUC! %s => %s (%.5f => %.5f)"%('>>', tgt_word, candidate,current_prob,temp_prob[orig_label]), flush=True)
                    # run_log.close()
                    # edits, sim = get_edits_similarity(code, adv_code, self.fasttext_model, self.args.language_type)
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words, invocation_number, current_prob - temp_prob[temp_label]
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            if most_gap > 0:
                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                final_code = get_example(final_code, tgt_word, candidate, "java")
                replaced_words[tgt_word] = candidate
                # run_log = open('../log/attack_log/alert_run.log', 'a')
                # run_log.write("%s ACC! %s => %s (%.5f => %.5f)\n"%('>>', tgt_word, candidate, current_prob + most_gap, current_prob))
                print("%s ACC! %s => %s (%.5f => %.5f)"%('>>', tgt_word, candidate, current_prob + most_gap, current_prob), flush=True)
                # run_log.close()
            else:
                replaced_words[tgt_word] = tgt_word

            adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words, invocation_number, most_gap


def get_embeddings(code, variables, tokenizer_mlm, codebert_mlm, args):
    new_code = copy.deepcopy(code)
    chromesome = {}
    for i in variables:
        chromesome[i] = '<unk>'
    # print(chromesome, flush=True)
    new_code = get_example_batch_coda(new_code, chromesome, "java")
    # print(new_code, flush=True)
    _, _, code_tokens = get_identifiers_c(new_code, "java")
    processed_code = " ".join(code_tokens)
    words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)
    if args.model_type == 'unixcoder':
        sub_words = [tokenizer_mlm.cls_token,"<encoder_only>",tokenizer_mlm.sep_token] + sub_words[:512 - 2] + [tokenizer_mlm.sep_token]
    else:
        sub_words = [tokenizer_mlm.cls_token] + sub_words[:512 - 2] + [tokenizer_mlm.sep_token]
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    with torch.no_grad():
        embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]

    return embeddings

class CodaAttacker:
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, codebert_mlm, fasttext_model, generated_substitutions) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.codebert_mlm = codebert_mlm
        self.fasttext_model = fasttext_model
        self.substitutions = generated_substitutions

    def attack(self, example, code):
        NUMBER_1 = 256
        NUMBER_2 = 64
        invocation_number = 1
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)
        if self.args.model_name == 'codebert':
            true_label = example[1].item()
        elif self.args.model_name == 'graphcodebert':
            true_label = example[3].item()
        elif self.args.model_name == 'unixcoder':
            true_label = example[1].item()
            
        variable_names, function_names, code_tokens = get_identifiers_c(code, "java")
        if (not orig_label == true_label) or len(variable_names)+len(function_names) == 0:
            return -2, invocation_number, 0, None, None
        start_time = time.time()
        all_variable_name = []
        random_subs = []
        all_code = [code] * NUMBER_2
        all_code_csc = [code] * NUMBER_2

        while len(random_subs) < NUMBER_1 and np.max(orig_prob) >= 0:
            # 概率第二大的 label
            orig_prob[np.argmax(orig_prob)] = -1
            topn_label = np.argmax(orig_prob)
            for i in np.random.choice(self.substitutions[str(topn_label)], size=len(self.substitutions[str(topn_label)]), replace=False):
                if len(i['variable_name']) < len(variable_names) or len(i['function_name']) < len(function_names):
                    continue
                all_variable_name.extend(i['variable_name'])
                temp = copy.deepcopy(i)
                temp['label'] = str(topn_label)
                random_subs.append(temp)
                if len(random_subs) >= NUMBER_1:
                    break
        end_time = time.time()

        substituions = []
        ori_embeddings = get_embeddings(code, variable_names+function_names, self.tokenizer_mlm, self.codebert_mlm, self.args)
        ori_embeddings = torch.nn.functional.pad(ori_embeddings, [0, 0, 0, 512 - np.shape(ori_embeddings)[1]])
        # print("end embedding", flush=True)
        embeddings_leng = np.shape(ori_embeddings)[-1]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for sub in random_subs:
            embeddings_index = sub['embeddings_index']
            embeddings = np.load('../dataset/substitutions/coda_%s_all_subs/%s_%s.npy' % (self.args.model_name, sub['label'], embeddings_index))
            embeddings = torch.from_numpy(embeddings).cuda()
            embeddings = torch.nn.functional.pad(embeddings, [0, 0, 0, 512 - np.shape(embeddings)[1]])
            substituions.append(([sub['variable_name'], sub['function_name'], sub['code']],
                                    np.sum(cos(ori_embeddings, embeddings).cpu().numpy()) / embeddings_leng))
        substituions = sorted(substituions, key=lambda x: x[1], reverse=True)
        substituions = [x[0] for x in substituions[:NUMBER_2]]
        max_number = len(substituions)

        temp_subs_variable_name = set()
        temp_subs_function_name = set()
        subs_code = []
        for subs in substituions:
            for i in subs[0]:
                temp_subs_variable_name.add(i)
            for i in subs[1]:
                temp_subs_function_name.add(i)
            subs_code.append(subs[2])

        min_prob = current_prob

        all_code_new = []
        # print("code style changes", flush=True)
        code_style = get_code_style(subs_code, 'java')
        replace_examples = []
        for temp in all_code_csc:
            try:
                temp_code = change_code_style(temp, "java", all_variable_name, code_style)[-1]
            except:
                temp_code = temp
            all_code_new.append(temp_code)
            if self.args.model_name == 'codebert':
                new_feature = codebert_convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
            elif self.args.model_name == 'graphcodebert':
                new_feature = graphcodebert_convert_code_to_features(temp_code, self.tokenizer_tgt, example[3].item(), self.args)
            elif self.args.model_name == 'unixcoder':
                new_feature = unixcoder_convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
            replace_examples.append(new_feature)

        if self.args.model_name == 'codebert':
            new_dataset = CodeDataset(replace_examples)
        elif self.args.model_name == 'graphcodebert':
            new_dataset = GraphCodeDataset(replace_examples, self.args)
        elif self.args.model_name == 'unixcoder':
            new_dataset = UniXCoderDataset(replace_examples)
            
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

        for index, temp_prob in enumerate(logits):
            invocation_number += 1
            temp_label = preds[index]
            if temp_label != orig_label:
                print("%s SUC! (%.5f => %.5f) (%d)" % ('>>', current_prob, temp_prob[orig_label], index + 1),
                        flush=True)
                return 2, invocation_number, end_time - start_time, all_code_new[index], current_prob - min(min_prob, temp_prob[orig_label])
            else:
                if min_prob > temp_prob[orig_label]:
                    min_prob = temp_prob[orig_label]
                    code = all_code_new[index]
        print("%s ACC! (%.5f => %.5f)" % ('>>', current_prob, min_prob), flush=True)
        
        subs_variable_name = []
        subs_function_name = []
        for i in temp_subs_variable_name:
            subs_variable_name.append([i, self.fasttext_model.get_word_vector(i)])
        for i in temp_subs_function_name:
            subs_function_name.append([i, self.fasttext_model.get_word_vector(i)])

        substituions = {}
        for i in variable_names:
            temp = []
            i_vec = self.fasttext_model.get_word_vector(i)
            for j in subs_variable_name:
                if i == j[0]:
                    continue
                temp.append([j[0], 1 - cosine_distance(i_vec, j[1])])
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            substituions[i] = [x[0] for x in temp]
        for i in function_names:
            temp = []
            i_vec = self.fasttext_model.get_word_vector(i)
            for j in subs_function_name:
                if i == j[0]:
                    continue
                temp.append([j[0], 1 - cosine_distance(i_vec, j[1])])
            temp = sorted(temp, key=lambda x: x[1], reverse=True)
            substituions[i] = [x[0] for x in temp]

        all_code = []
        all_code_csc = []
        replace_examples = []
        current_subs = ['' for i in range(len(variable_names) + len(function_names))]
        for i in range(max_number):
            temp_code = copy.deepcopy(code)
            for j, tgt_word in enumerate(variable_names):
                if i >= len(substituions[tgt_word]):
                    continue
                if substituions[tgt_word][i] in current_subs:
                    continue
                current_subs[j] = substituions[tgt_word][i]
                temp_code = get_example(temp_code, tgt_word, substituions[tgt_word][i], "java")

                all_code.append(temp_code)
                if self.args.model_name == 'codebert':
                    new_feature = codebert_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                    example[1].item(), self.args)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                            example[3].item(), self.args)
                elif self.args.model_name == 'unixcoder':
                    new_feature = unixcoder_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                    example[1].item(), self.args)
                replace_examples.append(new_feature)
            for j, tgt_word in enumerate(function_names):
                if i >= len(substituions[tgt_word]):
                    continue
                if substituions[tgt_word][i] in current_subs:
                    continue
                current_subs[j + len(variable_names)] = substituions[tgt_word][i]
                temp_code = get_example(temp_code, tgt_word, substituions[tgt_word][i], "java")

                all_code.append(temp_code)
                if self.args.model_name == 'codebert':
                    new_feature = codebert_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                    example[1].item(), self.args)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                            example[3].item(), self.args)
                elif self.args.model_name == 'unixcoder':
                    new_feature = unixcoder_convert_code_to_features(temp_code, self.tokenizer_tgt,
                                                                  example[1].item(), self.args)

                replace_examples.append(new_feature)

            all_code_csc.append(all_code[-1])


        if len(replace_examples) == 0:
            return -3, invocation_number, end_time - start_time, None, None
        if self.args.model_name == 'codebert':
            new_dataset = CodeDataset(replace_examples)
        elif self.args.model_name == 'graphcodebert':
            new_dataset = GraphCodeDataset(replace_examples, self.args)
        elif self.args.model_name == 'unixcoder':
            new_dataset = UniXCoderDataset(replace_examples)
            
        logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
        final_code = None
        for index, temp_prob in enumerate(logits):
            invocation_number += 1
            temp_label = preds[index]
            if temp_label != orig_label:
                print("%s SUC! (%.5f => %.5f) (%d)" % ('>>', current_prob, temp_prob[orig_label], index + 1),
                        flush=True)
                return 1, invocation_number, end_time - start_time, all_code[index], current_prob - temp_prob[orig_label]
            else:
                if min_prob >= temp_prob[orig_label]:
                    min_prob = temp_prob[orig_label]
                    final_code = all_code[index]
        print("%s ACC! (%.5f => %.5f)" % ('>>', current_prob, min_prob), flush=True)
        
        
        return -1, invocation_number, end_time - start_time, final_code, current_prob - min_prob

class MHMAttacker():
    def __init__(self, args, model_tgt, model_mlm, tokenizer, _token2idx, _idx2token) -> None:
        self.classifier = model_tgt
        self.model_mlm = model_mlm
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args
        self.tokenizer = tokenizer
    
    def mcmc(self, tokenizer, code=None, _label=None, _n_candi=30,
             _max_iter=100, _prob_threshold=0.95, subs = {}):
        identifiers, code_tokens = get_identifiers(code, 'java')
        prog_length = len(code_tokens)
        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, tokenizer)
        raw_tokens = copy.deepcopy(words)
        variable_names = list(subs.keys())
        
        uid = get_identifier_posistions_from_code(words, variable_names)

        if len(uid) <= 0: # 是有可能存在找不到变量名的情况的.
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        # 还需要得到substitues

        variable_substitue_dict = {}
        
        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = subs[tgt_word]
        
        old_uids = {}
        old_uid = ""
        for iteration in range(1, 1+_max_iter):
            # 这个函数需要tokens
            res = self.__replaceUID(_tokens=code, _label=_label, _uid=uid,
                                    substitute_dict=variable_substitue_dict,
                                    _n_candi=_n_candi,
                                    _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")
            
            if res['status'].lower() in ['s', 'a']:

                if iteration == 1:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]
                flag = 0
                for k in old_uids.keys():
                    if res["old_uid"] == old_uids[k][-1]:
                        flag = 1
                        old_uids[k].append(res["new_uid"])
                        old_uid = k
                        break
                if flag == 0:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                code = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid']) # 替换key，但保留value.
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])
                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        replace_info[uid_] = old_uids[uid_][-1]
                        # nb_changed_pos += len(uid[old_uids[uid_][-1]])
                    return {'succ': True, 'tokens': code,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"], "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0]-res["new_prob"][0], "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM"}
        replace_info = {}
        nb_changed_pos = 0
        for uid_ in old_uids.keys():
            replace_info[uid_] = old_uids[uid_][-1]
            # nb_changed_pos += len(uid[old_uids[uid_][-1]])
        return {'succ': False, 'tokens': res['tokens'], 'raw_tokens': None, "prog_length": prog_length, "new_pred": res["new_pred"], "is_success": -1, "old_uid": old_uid, "score_info": res["old_prob"][0]-res["new_prob"][0], "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM"}

    def mcmc_random(self, tokenizer, code=None, _label=None, _n_candi=30,
             _max_iter=100, _prob_threshold=0.95, subs = {}):
        identifiers, code_tokens = get_identifiers(code, 'java')
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, tokenizer)
        raw_tokens = copy.deepcopy(words)
        variable_names = list(subs.keys())
        
        uid = get_identifier_posistions_from_code(words, variable_names)

        if len(uid) <= 0: # 是有可能存在找不到变量名的情况的.
            return {'succ': None, 'tokens': None, 'raw_tokens': None}


        variable_substitue_dict = {}
        for tgt_word in uid.keys():
    
            variable_substitue_dict[tgt_word] = subs[tgt_word]

        old_uids = {}
        old_uid = ""
        for iteration in range(1, 1+_max_iter):
            # 这个函数需要tokens
            res = self.__replaceUID_random(_tokens=code, _label=_label, _uid=uid,
                                    substitute_dict=variable_substitue_dict,
                                    _n_candi=_n_candi,
                                    _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")
            
            if res['status'].lower() in ['s', 'a']:
                if iteration == 1:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]
                flag = 0
                for k in old_uids.keys():
                    if res["old_uid"] == old_uids[k][-1]:
                        flag = 1
                        old_uids[k].append(res["new_uid"])
                        old_uid = k
                        break
                if flag == 0:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]
                    
                code = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid']) # 替换key，但保留value.
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])
                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        replace_info[uid_] = old_uids[uid_][-1]
                        # nb_changed_pos += len(uid[old_uids[uid_][-1]])
                    return {'succ': True, 'tokens': code,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"], "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0]-res["new_prob"][0], "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM-Origin"}
        replace_info = {}
        nb_changed_pos = 0
        for uid_ in old_uids.keys():
            replace_info[uid_] = old_uids[uid_][-1]
            # nb_changed_pos += len(uid[old_uids[uid_][-1]])
        return {'succ': False, 'tokens': res['tokens'], 'raw_tokens': None, "prog_length": prog_length, "new_pred": res["new_pred"], "is_success": -1, "old_uid": old_uid, "score_info": res["old_prob"][0]-res["new_prob"][0], "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM-Origin"}
    
    def __replaceUID(self, _tokens, _label=None, _uid={}, substitute_dict={},
                     _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):
        
        assert _candi_mode.lower() in ["random", "nearby"]
        
        selected_uid = random.sample(substitute_dict.keys(), 1)[0] # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(substitute_dict[selected_uid], min(_n_candi, len(substitute_dict[selected_uid]))): # 选出_n_candi数量的候选.
                if c in _uid.keys():
                    continue
                if isUID(c): # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c, "java")

            new_example = []
            for tmp_tokens in candi_tokens:
                tmp_code = tmp_tokens
                if self.args.model_name == 'codebert':
                    new_feature = codebert_convert_code_to_features(tmp_code, self.tokenizer, _label, self.args)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(tmp_code, self.tokenizer, _label, self.args)
                elif self.args.model_name == 'unixcoder':
                    new_feature = unixcoder_convert_code_to_features(tmp_code, self.tokenizer, _label, self.args)
                new_example.append(new_feature)
            
            if self.args.model_name == 'codebert':
                new_dataset = CodeDataset(new_example)
            elif self.args.model_name == 'graphcodebert':
                new_dataset = GraphCodeDataset(new_example, self.args)
            elif self.args.model_name == 'unixcoder':
                new_dataset = UniXCoderDataset(new_example)
            prob, pred = self.classifier.get_results(new_dataset, self.args.eval_batch_size)

            for i in range(len(candi_token)):   # Find a valid example
                if pred[i] != _label: # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

            candi_idx = 0
            min_prob = 1.0

            for idx, a_prob in enumerate(prob[1:]):
                if a_prob[_label] < min_prob:
                    candi_idx = idx + 1
                    min_prob = a_prob[_label]

            # 找到Ground_truth对应的probability最小的那个mutant
            # At last, compute acceptance rate.
            alpha = (1-prob[candi_idx][_label]+1e-10) / (1-prob[0][_label]+1e-10)
            # 计算这个id对应的alpha值.
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
            else:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
        else:
            pass

    def __replaceUID_random(self, _tokens, _label=None, _uid={}, substitute_dict={},
                     _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):
        
        assert _candi_mode.lower() in ["random", "nearby"]
        selected_uid = random.sample(substitute_dict.keys(), 1)[0] # 选择需要被替换的变量名
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(self.idx2token, _n_candi): # 选出_n_candi数量的候选.
                if isUID(c): # 判断是否是变量名.
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c, "java")
                    # for i in _uid[selected_uid]: # 依次进行替换.

            new_example = []
            for tmp_tokens in candi_tokens:
                tmp_code = tmp_tokens
                if self.args.model_name == 'codebert':
                    new_feature = codebert_convert_code_to_features(tmp_code, self.tokenizer, _label, self.args)
                elif self.args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(tmp_code, self.tokenizer, _label, self.args)
                elif self.args.model_name == 'unixcoder':
                    new_feature = unixcoder_convert_code_to_features(tmp_code, self.tokenizer, _label, self.args)
                new_example.append(new_feature)
                
            if self.args.model_name == 'codebert':
                new_dataset = CodeDataset(new_example)
            elif self.args.model_name == 'graphcodebert':
                new_dataset = GraphCodeDataset(new_example, self.args)
            elif self.args.model_name == 'unixcoder':
                new_dataset = UniXCoderDataset(new_example)
                            
            prob, pred = self.classifier.get_results(new_dataset, self.args.eval_batch_size)
            for i in range(len(candi_token)):   # Find a valid example
                if pred[i] != _label: # 如果有样本攻击成功
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

            candi_idx = 0
            min_prob = 1.0

            for idx, a_prob in enumerate(prob[1:]):
                if a_prob[_label] < min_prob:
                    candi_idx = idx + 1
                    min_prob = a_prob[_label]

            # 找到Ground_truth对应的probability最小的那个mutant
            # At last, compute acceptance rate.
            alpha = (1-prob[candi_idx][_label]+1e-10) / (1-prob[0][_label]+1e-10)
            # 计算这个id对应的alpha值.
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
            else:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
        else:
            pass

    def __printRes(self, _iter=None, _res=None, _prefix="  => "):
        if _res['status'].lower() == 's':   # Accepted & successful
            print("%s iter %d, SUCC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'r': # Rejected
            print("%s iter %d, REJ. %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'a': # Accepted
            print("%s iter %d, ACC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)