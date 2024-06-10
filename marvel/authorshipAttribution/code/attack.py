from tree_sitter import Language, Parser
import sys, os

sys.path.append('../../../')
sys.path.append('../../')
# print(sys.path)
from load_data import *
import fasttext
import torch
import json
import argparse
import time
import random
import numpy as np
from model import *
from utils.utils_alert import build_vocab
from attacker import AlertAttacker, CodaAttacker, MHMAttacker
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification,
                          T5Config, T5ForConditionalGeneration)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = {
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer, CodeBERT, CodeBERTnoise, CodeBERT_twoContact),
    'graphcodebert': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, GraphCodeBERT, GraphCodeBERTnoise, GraphCodeBERT_twoContact),
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer, UniXCoder, UniXCodernoise, UniXCoder_twoContact)
}

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

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--eval_data_file", default='../dataset/valid.txt', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--model_name", default="", type=str,
                        help="model name")
    parser.add_argument("--model_type", default="", type=str,
                        help="model name")
    parser.add_argument("--attack_name", default="", type=str,
                        help="attack name")
    parser.add_argument("--info",default="", type=str,
                        help="info")
    parser.add_argument("--model_dir", default=None, type=str,
                        help="model_dir")
    parser.add_argument("--subs_path", default=None, type=str,
                        help="model_dir")
    parser.add_argument("--output_code", default=0, type=int,
                        help="model_dir")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="eval batch size")
    parser.add_argument("--save_name", default='model.bin', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    args = parser.parse_args()
    args.device = torch.device(device)
    
    # Set seed
    args.seed = 123456
    args.number_labels = 66
    # args.eval_batch_size = 32
    args.language_type = 'python'
    args.n_gpu = torch.cuda.device_count()
    args.block_size = 512
    args.use_ga = True
    args.code_length = 448
    args.data_flow_length = 64
    

    set_seed(args)
    
    print("==========Loading Model===========", flush=True)

    if args.model_type == "codebert":
        args.config_name = "microsoft/codebert-base"
        args.model_name_or_path = "microsoft/codebert-base"
        args.tokenizer_name = "microsoft/codebert-base"
        args.base_model = 'microsoft/codebert-base-mlm'
        args.model_name = 'codebert'
        args.code_length = 512
        args.data_flow_length = 0
    elif args.model_type == "graphcodebert":
        args.config_name = "microsoft/graphcodebert-base"
        args.model_name_or_path = "microsoft/graphcodebert-base"
        args.tokenizer_name = "microsoft/graphcodebert-base"
        args.base_model = 'microsoft/graphcodebert-base'
        args.model_name = 'graphcodebert'
    elif args.model_type == "unixcoder":
        args.config_name = "microsoft/unixcoder-base"
        args.model_name_or_path = "microsoft/unixcoder-base"
        args.tokenizer_name = "microsoft/unixcoder-base"
        args.base_model = "microsoft/unixcoder-base"
        args.model_name = 'unixcoder'
        args.code_length = 512
        args.data_flow_length = 0      
        

    config_class, model_class, tokenizer_class, Model, Model_Noise, Model_Contact = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name)
    config.num_labels=args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model = model_class.from_pretrained(args.model_name_or_path,config=config)


    outputdir = "../model/{}/checkpoint-best-acc/{}_{}_model.bin".format(args.model_type, args.model_type, args.save_name)
    
    if args.save_name == 'original':
        model = Model(model,config,tokenizer,args)
        model.config.ouput_attentions=True
        model.load_state_dict(torch.load(outputdir, map_location=torch.device(device)),strict=False)     
        model.to(device)    
    elif args.save_name == 'marvel':
        model1 = Model_Noise(model,config,tokenizer,args)
        model2 = Model_Noise(model,config,tokenizer,args)

        model1.config.ouput_attentions=True
        model2.config.ouput_attentions=True
        
        model = Model_Contact(config, model1, model2, args)
        model.load_state_dict(torch.load(outputdir, map_location=torch.device(device)),strict=False)     
        model.to(device)       
    print("=======Loading Model Finished=====", flush=True)
    
    config_class, model_class, tokenizer_class,_,_,_ = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)

    if args.attack_name == 'alert':
        attacker = AlertAttacker(args, model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
        print("======Get Alert Attacker======", flush=True)
        source_codes = []
        substs = []
        subs_path = "../dataset/substitutions/{}_valid_subs.jsonl".format(args.model_type)
        with open(subs_path) as rf:
            for line in rf:
                item = json.loads(line.strip())
                source_codes.append(item["code"].replace("\\n", "\n").replace("\"", '"'))
                substs.append(item["substitutes"])

    elif args.attack_name == 'mhm':
        codes_file_path = "../dataset/substitutions/{}_valid_subs.jsonl".format(args.model_type)
        print(codes_file_path)
        source_codes = []
        substs = []
        with open(codes_file_path) as rf:
            for line in rf:
                item = json.loads(line.strip())
                source_codes.append(item["code"].replace("\\n", "\n").replace("\"", '"'))
                substs.append(item["substitutes"])

        code_tokens = []
        for index, code in enumerate(source_codes):
            code_tokens.append(get_identifiers(code, "python")[1])

        id2token, token2id = build_vocab(code_tokens, 5000)
        print("======Get MHM Attacker======", flush=True)
        attacker = MHMAttacker(args, model, codebert_mlm, tokenizer, token2id, id2token)
        
    elif args.attack_name == 'coda':
        fasttext_model = fasttext.load_model("../../../utils/fasttext_model.bin")
        codebert_mlm.to('cuda')
        generated_substitutions = json.load(open('../dataset/substitutions/coda_%s_all_subs.json' % (args.model_name), 'r'))
        attacker = CodaAttacker(args, model, tokenizer, tokenizer_mlm, codebert_mlm, fasttext_model, generated_substitutions)

        source_codes = []
        with open(args.eval_data_file) as rf:
            for line in rf:
                source_codes.append(line.split(' <CODESPLIT> ')[0].strip().replace("\\n", "\n").replace('\"', '"'))        

    if args.model_type == 'codebert':
        eval_dataset = CodeBertTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_type == 'graphcodebert':
        eval_dataset = GraphCodeBertTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_type == 'unixcoder':
        eval_dataset = UniXCoderTextDataset(tokenizer, args, args.eval_data_file)
    print(len(eval_dataset), len(source_codes), flush=True)
    start_time = time.time()
    total_cnt = 0
    success_attack = 0
    tmp_final_code = " "
    prefix = '../log/attack_log'
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    query_times = 0

    for index, example in enumerate(eval_dataset):
        code = source_codes[index]
        if args.attack_name == 'mhm':
            subs = substs[index]
            orig_prob, orig_label = model.get_results([example], args.eval_batch_size)
            orig_prob = orig_prob[0]
            orig_label = orig_label[0]
            if args.model_name == 'codebert':
                true_label = example[1].item()
            elif args.model_name == 'graphcodebert':
                true_label = example[3].item()
            elif args.model_name == 'unixcoder':
                true_label = example[1].item()
                            
            if true_label != orig_label:
                is_success = -4
            else:
                _res = attacker.mcmc_random(tokenizer, code,
                                _label=true_label, _n_candi=30,
                                _max_iter=50, _prob_threshold=1, subs = subs)
                if _res['succ'] is None:
                    is_success = -1
                if _res['succ'] == True:
                    is_success = 1
                else:
                    is_success = -1
        if args.attack_name == 'alert':
            subs = substs[index]
            code, prog_length, final_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words, invocation_number1, min_gap_prob1 = attacker.greedy_attack(example, code, subs)

            if is_success == -1 and args.use_ga:
                # 如果不成功，则使用gi_attack
                code, prog_length, final_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words, invocation_number2, min_gap_prob2 = attacker.ga_attack(example, code, subs, initial_replace=replaced_words)

            tmp_final_code = final_code
        if args.attack_name == 'coda':
            try:
                is_success, invocation_number, gap_time, final_code, min_gap_prob = attacker.attack(example, code) 
                tmp_final_code = final_code           
            except:
                continue
        if is_success >= -1:
            # 如果原来正确
            total_cnt += 1
        if is_success >= 1:
            success_attack += 1

        end_time = time.time()

        if total_cnt is not 0:
            print("Success rate: {}".format(1.0 * success_attack / total_cnt), flush=True)
        
        print("Query times in this attack: ", model.query - query_times, flush=True)
        print("All Query times: ", model.query, flush=True)

        query_times = model.query
        print("Successful items count: {}".format(success_attack), flush=True)
        print("Total count: {} \nIndex: {} \nTime: {} min".format(total_cnt, index, round((end_time - start_time)/60, 2)), flush=True)
        
        if is_success >= 1: 
            print('num %d SUCCESS!\n'%index, flush=True)
        else:
            print('num %d FAILED!\n'%index, flush=True)

    print("Success rate: {}".format(1.0 * success_attack / total_cnt))
    print("Successful items count: {}".format(success_attack))
    print("Total count: {} \nIndex: {} \nTime: {}".format(total_cnt, index, end_time - start_time))
    
if __name__ == '__main__':
    main()