import os
import json
import sys
import argparse
from tqdm import tqdm
sys.path.append('../../../')

from language_parser.run_parser import get_identifiers_c, remove_comments_and_docstrings,get_example_batch, get_example_batch_coda
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer, 
                          RobertaForMaskedLM, RobertaForSequenceClassification,
                          T5Config, T5ForConditionalGeneration)
from model import CodeBERT, GraphCodeBERT, UniXCoder
from load_data import CodeBertTextDataset, GraphCodeBertTextDataset, UniXCoderTextDataset
import torch
import numpy as np
import copy
from utils.utils_alert import _tokenize
import multiprocessing
cpu_cont = 16


MODEL_CLASSES = {
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'graphcodebert': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer)
}


def get_embeddings(code, variables, tokenizer_mlm, codebert_mlm, args):
    new_code = copy.deepcopy(code)
    chromesome = {}
    for i in variables:
        chromesome[i] = '<unk>'
    # print(chromesome)
    new_code = get_example_batch(new_code, chromesome, "java")

    try:
        _, _, code_tokens = get_identifiers_c(remove_comments_and_docstrings(new_code, "java"), "java")
    except:
        _, _, code_tokens = get_identifiers_c(new_code, 'java')
    processed_code = " ".join(code_tokens)
    words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)
    if args.model_name == 'unixcoder':
        sub_words = [tokenizer_mlm.cls_token,"<encoder_only>",tokenizer_mlm.sep_token]  + sub_words[:args.block_size - 2] + [tokenizer_mlm.sep_token]
    else:
        sub_words = [tokenizer_mlm.cls_token] + sub_words[:args.block_size - 2] + [tokenizer_mlm.sep_token]
    input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
    with torch.no_grad():
        embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_data_file", default='./all.txt', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--model_name", default="", type=str,
                        help="model name")

    args = parser.parse_args()
    args.device = torch.device("cuda")
    args.seed = 123456
    args.block_size = 512
    args.eval_batch_size = 32
    args.number_labels = 250
    args.language_type = 'java'
    args.store_path = './substitutions/coda_%s_all_subs.json' % args.model_name

    if args.model_name == 'codebert':
        args.output_dir = '../model/codebert/'
        args.model_type = 'codebert'
        args.config_name = "microsoft/codebert-base"
        args.model_name_or_path = "microsoft/codebert-base"
        args.tokenizer_name = "microsoft/codebert-base"
        args.base_model = 'microsoft/codebert-base-mlm'
        args.block_size = 512
    if args.model_name == 'graphcodebert':
        args.output_dir = '../model/graphcodebert/'
        args.model_type = 'graphcodebert'
        args.config_name = "microsoft/graphcodebert-base"
        args.model_name_or_path = "microsoft/graphcodebert-base"
        args.tokenizer_name = "microsoft/graphcodebert-base"
        args.base_model = 'microsoft/graphcodebert-base'
        args.code_length = 384
        args.data_flow_length = 128  
    if args.model_name == 'unixcoder':
        args.output_dir = '../model/unixcoder/'
        args.model_type = 'unixcoder'
        args.config_name = "microsoft/unixcoder-base"
        args.model_name_or_path = "microsoft/unixcoder-base"
        args.tokenizer_name = "microsoft/unixcoder-base"
        args.base_model = 'microsoft/unixcoder-base'
        args.block_size = 512
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, 510)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    if args.model_name == 'codebert':
        model = CodeBERT(model, config, tokenizer, args)
    elif args.model_name == 'graphcodebert':
        model = GraphCodeBERT(model, config, tokenizer, args) 
    elif args.model_name == 'unixcoder':
        model = UniXCoder(model, config, tokenizer, args)
    checkpoint_prefix = 'checkpoint-best-acc/%s_original_model.bin' % args.model_name
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    codebert_mlm.to(args.device)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    pool = multiprocessing.Pool(cpu_cont)
    if args.model_name == 'codebert':
        all_dataset = CodeBertTextDataset(tokenizer, args, args.all_data_file, pool=pool)
    elif args.model_name == 'graphcodebert':
        all_dataset = GraphCodeBertTextDataset(tokenizer, args, args.all_data_file, pool=pool)
    elif args.model_name == 'unixcoder':
        all_dataset = UniXCoderTextDataset(tokenizer, args, args.all_data_file, pool=pool)

    source_codes = []
    with open(args.all_data_file) as rf:
        for line in rf:
            temp = line.split('<CODESPLIT>')[0].replace("\\n", "\n").replace('\"', '"')
            source_codes.append(temp)
    assert (len(source_codes) == len(all_dataset))
    print('length of all data', len(source_codes))

    all_labels = {}
    count = 0
    with open(args.store_path, "w") as wf:
        for index, example in tqdm(enumerate(all_dataset), total=len(all_dataset)):            
            logits, preds = model.get_results([example], args.eval_batch_size)
            if args.model_name == 'codebert':
                true_label = str(int(example[1].item()))
            elif args.model_name == 'graphcodebert':
                true_label = str(int(example[3].item()))
            elif args.model_name == 'unixcoder':
                true_label = str(int(example[1].item()))
                
            orig_prob = np.max(logits[0])
            orig_label = str(int(preds[0]))
            code = source_codes[index]

            if not true_label == orig_label:
                continue

            if true_label not in all_labels.keys():
                all_labels[true_label] = []

            try:
                variable_name, function_name, _ = get_identifiers_c(remove_comments_and_docstrings(code, "java"), "java")
            except:
                variable_name, function_name, _ = get_identifiers_c(code, "java")

            # print(variable_name)
            # print(function_name)
            variables = []
            variables.extend(variable_name)
            variables.extend(function_name)

            try:
                embeddings = get_embeddings(code, variables, tokenizer_mlm, codebert_mlm, args)
            except:
                continue
            if not os.path.exists('./substitutions/coda_%s_all_subs' % args.model_name):
                os.makedirs('./substitutions/coda_%s_all_subs' % args.model_name)
            np.save('./substitutions/coda_%s_all_subs/%s_%s' % (args.model_name, str(orig_label), str(index)), embeddings.cpu().numpy())
            all_labels[true_label].append({'code': code, 'embeddings_index': index, 'variable_name': variable_name, 'function_name': function_name})
            count += 1
        print(count, len(all_dataset), count/len(all_dataset))
        wf.write(json.dumps(all_labels) + '\n')


if __name__ == "__main__":
    main()
