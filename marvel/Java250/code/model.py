import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
import numpy as np
import sys
sys.path.append("../../../")
from language_parser.run_parser import get_identifiers_from_tokens

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

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # take <s> token (equiv. to [CLS])
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CodeBERT(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeBERT, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        padding_idx = self.encoder.embeddings.padding_idx
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + padding_idx

        input_ids=input_ids.view(-1,self.args.block_size)
        attn_mask=input_ids.ne(1)
        inputs_embeddings=self.encoder.embeddings.word_embeddings(input_ids)  
        outputs = self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_ids)[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
      
    def get_results(self, dataset, batch_size):
        '''
        给定example和tgt model，返回预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")       
            label = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs,label)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels

class GraphCodeBERT(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(GraphCodeBERT, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.query = 0
    
    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None): 

        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)
    
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]

        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]

        logits=self.classifier(outputs)
        prob=F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob

    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs_ids = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
            attn_mask = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            position_idx = batch[2].to("cuda" if torch.cuda.is_available() else "cpu")
            label = batch[3].to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs_ids, attn_mask, position_idx, label)
                logits.append(logit.cpu().numpy())
        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))
        return probs, pred_labels

class UniXCoder(nn.Module):   
    def __init__(self,encoder,config,tokenizer,args):
        super(UniXCoder, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0
    
    def tokenize(self, inputs, mode="<encoder-only>", max_length=512, padding=False):
        """ 
        Convert string to token ids 
                
        Parameters:

        * `inputs`- list of input strings.
        * `max_length`- The maximum total source sequence length after tokenization.
        * `padding`- whether to pad source sequence length to max_length. 
        * `mode`- which mode the sequence will use. i.e. <encoder-only>, <decoder-only>, <encoder-decoder>
        """
        assert mode in ["<encoder-only>", "<decoder-only>", "<encoder-decoder>"]
        assert max_length < 1024
        
        tokenizer = self.tokenizer
        
        tokens_ids = []
        tokens_list = []
        for x in inputs:
            tokens = tokenizer.tokenize(x)
            if mode == "<encoder-only>":
                tokens = tokens[:max_length-4]
                tokens = [tokenizer.cls_token,mode,tokenizer.sep_token] + tokens + [tokenizer.sep_token]                
            elif mode == "<decoder-only>":
                tokens = tokens[-(max_length-3):]
                tokens = [tokenizer.cls_token,mode,tokenizer.sep_token] + tokens
            else:
                tokens = tokens[:max_length-5]
                tokens = [tokenizer.cls_token,mode,tokenizer.sep_token] + tokens + [tokenizer.sep_token]
            tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            if padding:
                tokens_id = tokens_id + [self.config.pad_token_id] * (max_length-len(tokens_id))
            tokens_ids.append(tokens_id)
            tokens_list.append(tokens)
        
        return tokens_ids,tokens_list
        
    def forward(self, input_ids=None,labels=None): 
        input_ids = input_ids.view(-1,self.args.block_size)
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=self.classifier(outputs)
        prob=F.softmax(logits)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
        
    def get_results(self, dataset, batch_size):
        '''
            用当前模型在给定数据集上求预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits=[] 
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")        
            label = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                lm_loss,logit = self.forward(inputs,label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                
            nb_eval_steps += 1
        logits=np.concatenate(logits,0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels

class CodeBERTnoise(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeBERTnoise, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0
        self.config.output_attentions=True

    def forward(self, input_ids=None, labels=None, noise=None):
        padding_idx = self.encoder.embeddings.padding_idx
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + padding_idx

        input_ids=input_ids.view(-1,self.args.block_size)
        attn_mask=input_ids.ne(1)
        inputs_embeddings=self.encoder.embeddings.word_embeddings(input_ids)  

        if noise is not None:
            inputs_embeddings += noise

        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_ids,output_attentions=True)
        
        sequence_outputs = outputs[0]  
        logits = self.classifier(sequence_outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, sequence_outputs, outputs.attentions[0]
        else:
            return prob, logits
    
    def get_outputs(self, input_ids=None, labels=None, noise=None):
        padding_idx = self.encoder.embeddings.padding_idx
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + padding_idx

        input_ids=input_ids.view(-1,self.args.block_size)
        attn_mask=input_ids.ne(1)
        inputs_embeddings=self.encoder.embeddings.word_embeddings(input_ids)  

        if noise is not None:
            inputs_embeddings += noise

        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_ids,output_attentions=True)
        
        sequence_outputs = outputs[0]
        return sequence_outputs
              
    def get_results(self, dataset, batch_size, new_infer=False):
        '''
        给定example和tgt model，返回预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")       
            label = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            if new_infer:
                _,_,_, attentions = self.forward(inputs,label)
                noise = calculate_noise(self, inputs, attentions, self.args)
            with torch.no_grad():
                lm_loss, logit, _, _ = self.forward(inputs,label,noise = noise if new_infer else None)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels

class GraphCodeBERTnoise(nn.Module):
    def __init__(self,  encoder, config, tokenizer, args):
        super(GraphCodeBERTnoise, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.query = 0
        self.config.output_attentions=True

    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None, noise=None):
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
        
        if noise is not None:
            inputs_embeddings += noise

        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx,output_attentions=True)
        
        sequence_outputs = outputs[0]  
        
        logits = self.classifier(sequence_outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, sequence_outputs, outputs.attentions[0]
        else:
            return prob, logits

    def get_outputs(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None, noise=None):
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
        
        if noise is not None:
            inputs_embeddings += noise

        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx,output_attentions=True)
        
        sequence_outputs = outputs[0]
        return sequence_outputs

    def get_results(self, dataset, batch_size, new_infer=False):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs_ids = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
            attn_mask = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            position_idx = batch[2].to("cuda" if torch.cuda.is_available() else "cpu")
            label = batch[3].to("cuda" if torch.cuda.is_available() else "cpu")
            if new_infer:
                _,_,_, attentions = self.forward(inputs_ids, attn_mask, position_idx, label)
                noise = calculate_noise(self, inputs_ids, attentions, self.args)
            with torch.no_grad():
                lm_loss, logit, _, _ = self.forward(inputs_ids, attn_mask, position_idx, label, noise = noise if new_infer else None)
                logits.append(logit.cpu().numpy())
        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))
        return probs, pred_labels

class UniXCodernoise(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(UniXCodernoise, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0
        self.config.output_attentions=True

    def forward(self, input_ids=None, labels=None, noise=None):
        padding_idx = self.encoder.embeddings.padding_idx
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + padding_idx

        input_ids=input_ids.view(-1,self.args.block_size)
        attn_mask=input_ids.ne(1)
        inputs_embeddings=self.encoder.embeddings.word_embeddings(input_ids)  

        if noise is not None:
            inputs_embeddings += noise

        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_ids,output_attentions=True)
        
        sequence_outputs = outputs[0]  
        logits = self.classifier(sequence_outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, sequence_outputs, outputs.attentions[0]
        else:
            return prob, logits
    
    def get_outputs(self, input_ids=None, labels=None, noise=None):
        padding_idx = self.encoder.embeddings.padding_idx
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        position_ids = incremental_indices.long() + padding_idx

        input_ids=input_ids.view(-1,self.args.block_size)
        attn_mask=input_ids.ne(1)
        inputs_embeddings=self.encoder.embeddings.word_embeddings(input_ids)  

        if noise is not None:
            inputs_embeddings += noise

        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_ids,output_attentions=True)
        
        sequence_outputs = outputs[0]
        return sequence_outputs
              
    def get_results(self, dataset, batch_size, new_infer=False):
        '''
        给定example和tgt model，返回预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")       
            label = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            if new_infer:
                _,_,_, attentions = self.forward(inputs,label)
                noise = calculate_noise(self, inputs, attentions, self.args)
            with torch.no_grad():
                lm_loss, logit, _, _ = self.forward(inputs,label,noise = noise if new_infer else None)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels

class RobertaClassificationHead_twoContact(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2*config.hidden_size, 2*config.hidden_size)
        self.dropout = nn.Dropout(2*config.hidden_dropout_prob)
        self.out_proj = nn.Linear(2*config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CodeBERT_twoContact(nn.Module):
    def __init__(self, config, model1, model2, args):
        super(CodeBERT_twoContact, self).__init__()
        self.args = args
        self.config = config 
        self.model1 = model1
        self.model2 = model2
        self.classifier = RobertaClassificationHead_twoContact(self.config)
        self.query = 0
        
    def forward(self, input_ids=None, labels=None, noise=None):
        outputs1 = self.model1.get_outputs(input_ids, labels, noise)[:,0,:]
        outputs2 = self.model2.get_outputs(input_ids, labels, noise=None)[:,0,:]
        outputs = torch.cat((outputs1, outputs2), dim=1)
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob, logits
        
    def get_results(self, dataset, batch_size):
        '''
        给定example和tgt model，返回预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")       
            label = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            _,_,_, attentions = self.model1(inputs,label)
            noise = calculate_noise(self.model1, inputs, attentions, self.args)
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs,label,noise)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels
        
class GraphCodeBERT_twoContact(nn.Module):
    def __init__(self, config, model1, model2, args):
        super(GraphCodeBERT_twoContact, self).__init__()
        self.args = args
        self.config = config 
        self.model1 = model1
        self.model2 = model2
        self.classifier = RobertaClassificationHead_twoContact(self.config)
        self.query = 0
        
    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels=None, noise=None):
        outputs1 = self.model1.get_outputs(inputs_ids, attn_mask, position_idx, labels, noise)[:, 0, :]
        outputs2 = self.model2.get_outputs(inputs_ids, attn_mask, position_idx, labels, noise=None)[:, 0, :]
        outputs = torch.cat((outputs1, outputs2), dim=1)
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob, logits
        
    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)
        self.eval()
        eval_loss = 0.0
        nb_eval_steps =0
        logits = []
        for batch in eval_dataloader:
            inputs_ids = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")
            attn_mask = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            position_idx = batch[2].to("cuda" if torch.cuda.is_available() else "cpu")
            label = batch[3].to("cuda" if torch.cuda.is_available() else "cpu")
            
            _,_,_, attentions = self.model1(inputs_ids,attn_mask,position_idx,label)
            noise = calculate_noise(self.model1, inputs_ids, attentions, self.args)
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs_ids,attn_mask,position_idx,label,noise)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels


class UniXCoder_twoContact(nn.Module):
    def __init__(self, config, model1, model2, args):
        super(UniXCoder_twoContact, self).__init__()
        self.args = args
        self.config = config 
        self.model1 = model1
        self.model2 = model2
        self.classifier = RobertaClassificationHead_twoContact(self.config)
        self.query = 0
        
    def forward(self, input_ids=None, labels=None, noise=None):
        outputs1 = self.model1.get_outputs(input_ids, labels, noise)[:,0,:]
        outputs2 = self.model2.get_outputs(input_ids, labels, noise=None)[:,0,:]
        outputs = torch.cat((outputs1, outputs2), dim=1)
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob, logits
        
    def get_results(self, dataset, batch_size):
        '''
        给定example和tgt model，返回预测的label和probability
        '''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")       
            label = batch[1].to("cuda" if torch.cuda.is_available() else "cpu")
            _,_,_, attentions = self.model1(inputs,label)
            noise = calculate_noise(self.model1, inputs, attentions, self.args)
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs,label,noise)
                # 调用这个模型. 重写了反前向传播模型.
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels   