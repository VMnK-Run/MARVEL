from __future__ import absolute_import, division, print_function
import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../../../')

from language_parser.parse_util import extract_dataflow
from language_parser import parsers
from language_parser import get_identifiers_c, get_example, get_identifiers

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.model_type == 'codebert':
        os.environ['PYHTONHASHSEED'] = str(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    elif args.model_type == 'graphcodebert':
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    elif args.model_type == 'unixcoder':
        os.environ['PYHTONHASHSEED'] = str(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True


class CodeBertInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

class GraphCodeBertInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, input_tokens, input_ids, position_idx, dfg_to_code, dfg_to_dfg, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.label = label
        
class UniXCoderInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

def codebert_convert_examples_to_features(code, label, tokenizer, args):
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return CodeBertInputFeatures(source_tokens, source_ids, 0, label)

def graphcodebert_convert_examples_to_features(code, tokenizer, label, args):
    # print(args.code_length, args.data_flow_length)
    parser = parsers["python"]
    code_tokens,dfg = extract_dataflow(code, parser, "python")
    code_tokens = [tokenizer.tokenize('@ '+x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1]+len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]
    code_tokens = code_tokens[:args.code_length+args.data_flow_length-2-min(len(dfg), args.data_flow_length)]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:args.code_length+args.data_flow_length-len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    # print(source_tokens.shape)
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length+args.data_flow_length-len(source_ids)
    position_idx += [tokenizer.pad_token_id]*padding_length
    source_ids += [tokenizer.pad_token_id]*padding_length
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]
    return GraphCodeBertInputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg,label)

def unixcoder_convert_examples_to_features(code, label, tokenizer, args):
    code_tokens = tokenizer.tokenize(code)[:args.block_size-4]
    source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return UniXCoderInputFeatures(source_tokens, source_ids, 0, label)

class GraphCodeBertTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        self.args=args
        file_type = file_path.split('/')[-1].split('.')[0]
        folder = '/'.join(file_path.split('/')[:-1]) + '/cached'

        if not os.path.exists(folder):
            os.makedirs(folder)
        cache_file_path = os.path.join(folder, '{}_cached_{}'.format(args.model_type,file_type))

        print('\n cached_features_file: ',cache_file_path)
        try:
            self.examples = torch.load(cache_file_path)
        except:
            code_files = []
            with open(file_path) as f:
                for line in f:
                    code = line.split(" <CODESPLIT> ")[0]
                    code = code.replace("\\n", "\n").replace('\"','"')
                    label = line.split(" <CODESPLIT> ")[1]
                    # 将这俩内容转化成input.
                    self.examples.append(graphcodebert_convert_examples_to_features(code, tokenizer, int(label), args))
                    code_files.append(code)
                    # 这里每次都是重新读取并处理数据集，能否cache然后load
            assert(len(self.examples) == len(code_files))

            torch.save(self.examples, cache_file_path)

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
        for idx,i in enumerate(self.examples[item].input_ids):
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
              
        return (torch.tensor(self.examples[item].input_ids),
              torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx),
              torch.tensor(self.examples[item].label))
            
class CodeBertTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        file_type = file_path.split('/')[-1].split('.')[0]
        folder = '/'.join(file_path.split('/')[:-1]) + '/cached'
        if not os.path.exists(folder):
            os.makedirs(folder)
        cache_file_path = os.path.join(folder, '{}_cached_{}'.format(args.model_type, file_type))
        print('\n cached_features_file: ', cache_file_path)
        try:
            self.examples = torch.load(cache_file_path)
        except:
            code_files = []
            with open(file_path) as f:
                for line in f:
                    code = line.split(" <CODESPLIT> ")[0]
                    code = code.replace("\\n", "\n").replace('\"', '"')
                    label = line.split(" <CODESPLIT> ")[1]
                    # 将这俩内容转化成input.
                    self.examples.append(codebert_convert_examples_to_features(code, int(label), tokenizer,args))
                    code_files.append(code)
                    # 这里每次都是重新读取并处理数据集，能否cache然后load
            assert(len(self.examples) == len(code_files))
            torch.save(self.examples, cache_file_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)

class UniXCoderTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        file_type = file_path.split('/')[-1].split('.')[0]
        folder = '/'.join(file_path.split('/')[:-1]) + '/cached'
        if not os.path.exists(folder):
            os.makedirs(folder)
        cache_file_path = os.path.join(folder, '{}_cached_{}'.format(args.model_type, file_type))
        print('\n cached_features_file: ',cache_file_path)
        try:
            self.examples = torch.load(cache_file_path)
        except:
            code_files = []
            with open(file_path) as f:
                for line in f:
                    code = line.split(" <CODESPLIT> ")[0]
                    code = code.replace("\\n", "\n").replace('\"', '"')
                    label = line.split(" <CODESPLIT> ")[1]
                    # 将这俩内容转化成input.
                    self.examples.append(unixcoder_convert_examples_to_features(code, int(label), tokenizer,args))
                    code_files.append(code)
                    # 这里每次都是重新读取并处理数据集，能否cache然后load
            assert(len(self.examples) == len(code_files))
            torch.save(self.examples, cache_file_path)
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):  
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)