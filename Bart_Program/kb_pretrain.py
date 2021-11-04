import argparse
import json
import logging
import os
import random
import time
from queue import SimpleQueue

import numpy as np
import torch
from torch import optim
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer)
from transformers.utils.dummy_pt_objects import DataCollatorForSeq2Seq
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.misc import seed_everything

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings

warnings.simplefilter(
    "ignore")  # hide warnings that caused by invalid sparql query


class Node:
    def __init__(self, _type, _info):
        self.type = _type
        self.info = _info
        if self.type == 'entity':
            self.name = self.info['name']
        elif self.type == 'relation':
            self.name = '%s %s' % (self.info['predicate'],
                                   self.info['direction'])
        elif self.type == 'attribute name':
            self.name = self.info['key']
        else:
            self.name = str(self.info)

    def __repr__(self):
        return self.name


def seq_rep(seq):
    res = []
    for s, v, o in seq:
        res.append('(%s, %s, %s)' % (s.name, v.name, o.name))
    return ', '.join(res)


def rand_mask(path):
    node = random.choice(random.choice(path))
    ans = node.name
    node.name = '<mask>'
    res = 'Predict %s %s' % (node.type, seq_rep(path))
    node.name = ans
    return res, ans


def kb_encode_dataset(kb, tokenizer, mode='triple'):
    """
        mode: triple or path
    """
    entities = kb['entities']
    concepts = kb['concepts']
    id2name = dict()
    for key, val in entities.items():
        id2name[key] = val['name']
    for key, val in concepts.items():
        id2name[key] = val['name']

    def get_ent_triple(ent):
        name = ent['name']
        for attr in ent['attributes']:
            attr_name = attr['key']
            val = str(attr['value']['value'])
            yield ('Predict entity (<mask>, %s, %s)' % (attr_name, val), name)
            yield ('Predict attribute name (%s, <mask>, %s)' % (name, val),
                   attr_name)
            yield ('Predict attribute value (%s, %s, <mask>)' %
                   (name, attr_name), val)
            qual_list = []
            for qual_key, qual_items in attr['qualifiers'].items():
                for qual_item in qual_items:
                    val = str(qual_item['value'])
                    qual_list.append(
                        ('Predict attribute name (<mask>, %s, %s)' %
                         (qual_key, val), attr_name))
                    qual_list.append(
                        ('Predict qualifier name (%s, <mask>, %s)' %
                         (attr_name, val), qual_key))
                    qual_list.append(
                        ('Predict qualifier value (%s, %s, <mask>)' %
                         (attr_name, qual_key), val))
            yield from random.sample(qual_list, int(0.1 * len(qual_list)))
        for rel in ent['relations']:
            obj_name = id2name[rel['object']]
            rel_name = '%s %s' % (rel['predicate'], rel['direction'])
            yield ('Predict head entity (<mask>, %s, %s)' %
                   (rel_name, obj_name), name)
            yield ('Predict tail entity (%s, %s, <mask>)' % (name, rel_name),
                   obj_name)
            yield ('Predict relation (%s, <mask>, %s)' % (name, obj_name),
                   rel_name)
            qual_list = []
            for qual_key, qual_items in rel['qualifiers'].items():
                for qual_item in qual_items:
                    val = str(qual_item['value'])
                    qual_list.append(
                        ('Predict attribute name (<mask>, %s, %s)' %
                         (qual_key, val), rel_name))
                    qual_list.append(
                        ('Predict qualifier name (%s, <mask>, %s)' %
                         (rel_name, val), qual_key))
                    qual_list.append(
                        ('Predict qualifier value (%s, %s, <mask>)' %
                         (rel_name, qual_key), val))
            yield from random.sample(qual_list, int(0.1 * len(qual_list)))

    def get_set(all_triples):
        all_triples = random.sample(all_triples, int(0.1 * len(all_triples)))
        inputs, outputs = zip(*all_triples)
        input_ids = tokenizer.batch_encode_plus(list(inputs),
                                                padding=True,
                                                max_length=1024,
                                                truncation=True)
        source_ids = input_ids['input_ids']
        source_mask = input_ids['attention_mask']
        target_ids = tokenizer.batch_encode_plus(list(outputs),
                                                 padding=True,
                                                 max_length=1024,
                                                 truncation=True)
        return source_ids, source_mask, target_ids['input_ids']

    def rand_walk(head_ent, n_sample=10, max_len=3):
        q = SimpleQueue()
        for _ in range(n_sample):
            q.put((Node('entity', head_ent), []))
        res = []
        while not q.empty():
            node, seq = q.get()
            if len(seq) >= max_len or (len(seq) > 0
                                       and np.random.rand() < 0.25):
                res.append(seq)
                continue
            if node.type == 'entity':
                if 'relations' not in ent:
                    res.append(seq)
                    continue
                rels = ent['relations']
                attrs = ent['attributes']
                edges = rels + attrs
                if len(edges) == 0:
                    res.append(seq)
                    continue
                indice = np.random.choice(len(edges), 1)
                for idx in indice:
                    seq_copy = list(seq)
                    if idx < len(rels):
                        rel = Node('relation', edges[idx])
                        obj_id = rel.info['object']
                        if obj_id in entities:
                            obj = entities[obj_id]
                        else:
                            obj = concepts[obj_id]
                        obj = Node('entity', obj)
                        if obj.name != node.name:
                            seq_copy.append((node, rel, obj))
                            q.put((rel, seq_copy))
                            q.put((obj, seq_copy))
                    else:
                        attr = Node('attribute name', edges[idx])
                        val = Node('attribute value',
                                   str(attr.info['value']['value']))
                        seq_copy.append((node, attr, val))
                        q.put((attr, seq_copy))

            elif node.type == 'relation' or node.type == 'attribute name':
                attr = node.info
                if len(attr['qualifiers']) > 0:
                    seq_copy = list(seq)
                    qual_list = []
                    for qual_key, qual_items in attr['qualifiers'].items():
                        for qual_item in qual_items:
                            qual_list.append(
                                (qual_key, str(qual_item['value'])))
                    k, v = random.choice(qual_list)
                    qual_key = Node('qualifier name', k)
                    qual_val = Node('qualifier value', v)
                    seq_copy.append((node, qual_key, qual_val))
                    res.append(seq_copy)
        return res

    all_triples = []
    entity_items = list(entities.values())
    random.shuffle(entity_items)
    split_point = int(0.9 * len(entity_items))
    train_entities = entity_items[:split_point]
    valid_entities = entity_items[split_point:]
    all_triples = []
    if mode == 'triple':
        for ent in train_entities:
            all_triples.extend(get_ent_triple(ent))
    else:
        for ent in train_entities:
            for path in rand_walk(ent, 100):
                if len(path) > 0:
                    all_triples.append(rand_mask(path))
    train_set = get_set(all_triples)
    logging.info('# train set %d' % len(all_triples))
    all_triples = []
    for ent in valid_entities:
        all_triples.extend(get_ent_triple(ent))
    valid_set = get_set(all_triples)
    logging.info('# valid set %d' % len(all_triples))
    return train_set, valid_set


def preprocess(kb_path, tokenizer):
    kb = json.load(open(kb_path))
    dataset = kb_encode_dataset(kb, tokenizer)
    return dataset


def collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    target_ids = torch.stack(batch[2])
    return source_ids, source_mask, target_ids


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids = inputs

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        target_ids = torch.LongTensor(self.target_ids[index])
        return source_ids, source_mask, target_ids

    def __len__(self):
        return len(self.source_ids)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, inputs, batch_size, training=False, ratio=1.0):
        dataset = Dataset(inputs)
        # np.shuffle(dataset)
        # dataset = dataset[:(int)(len(dataset) / 10)]
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate,
        )


def validate(args, model, data, device, tokenizer):
    model.eval()
    with torch.no_grad():
        loss_list = []
        for batch in data:
            pad_token_id = tokenizer.pad_token_id
            source_ids, source_mask, y = batch[0], batch[1], batch[2]
            source_ids.to(device)
            source_mask.to(device)
            y.to(device)
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone()
            labels[y[:, 1:] == pad_token_id] = -100

            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": labels.to(device),
            }
            outputs = model(**inputs)
            loss = outputs[0]
            loss_list.append(loss.item())
    return np.mean(loss_list)


def train_step(args,
               model,
               train_loader,
               device,
               tokenizer,
               optimizer,
               scheduler,
               ratio=0.1):
    step_loss = 0
    stop_step = int(ratio * len(train_loader))
    for step, batch in enumerate(train_loader):
        if step >= stop_step:
            break
        model.train()
        pad_token_id = tokenizer.pad_token_id
        source_ids, source_mask, y = batch[0], batch[1], batch[2]
        source_ids.to(device)
        source_mask.to(device)
        y.to(device)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone()
        labels[y[:, 1:] == pad_token_id] = -100

        inputs = {
            "input_ids": source_ids.to(device),
            "attention_mask": source_mask.to(device),
            "decoder_input_ids": y_ids.to(device),
            "labels": labels.to(device),
        }
        outputs = model(**inputs)
        loss = args.kbp_weight * outputs[0]
        loss.backward()
        step_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
    return step_loss / stop_step


def prepare_data(args, kb_path, tokenizer):
    kb = json.load(open(kb_path))
    dataset = kb_encode_dataset(kb, tokenizer, args.kbp_mode)
    train_inputs, valid_outputs = dataset 

    train_loader = DataLoader(train_inputs,
                              args.batch_size,
                              training=True,
                              ratio=args.sample)
    val_loader = DataLoader(valid_outputs, 64)
    return train_loader, val_loader


def train(args):
    device = torch.device(args.device)
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig,
                                                  BartForConditionalGeneration,
                                                  BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model = model.to(device)

    logging.info("Create train_loader and val_loader.........")
    kb_path = os.path.join('./dataset/', 'kb.json')
    train_loader, val_loader = prepare_data(args, kb_path, tokenizer)
    logging.info(model)
    t_total = len(
        train_loader
    ) // args.gradient_accumulation_steps * args.num_train_epochs  # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bart_param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in bart_param_optimizer
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        args.weight_decay,
        'lr':
        args.learning_rate
    }, {
        'params': [
            p for n, p in bart_param_optimizer
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0,
        'lr':
        args.learning_rate
    }]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = optim.AdamW(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)
    logging.info('Checking...')
    val_loss = validate(args, model, val_loader, device, tokenizer)
    logging.info('[Check] valid loss %.8f' % val_loss)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    prefix = 0
    save_best_num = 3
    acc_ckpt_pairs = [(100, 'NOT_EXIST')]
    for epoch in range(int(args.num_train_epochs)):
        epoch_loss = train_step(args, model, train_loader, device, tokenizer,
                                optimizer, scheduler)
        logging.info('[Train] Epoch %d: loss %.8f' %
                     (epoch, epoch_loss / len(train_loader)))
        val_loss = validate(args, model, val_loader, device, tokenizer)
        logging.info('[Valid] Epoch %d: loss %.8f' % (epoch, val_loss))
        if acc_ckpt_pairs[-1][0] > val_loss:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, args.comment,
                                      'epoch_%d' % epoch)
            acc_ckpt_pairs.append((val_loss, output_dir))
            acc_ckpt_pairs.sort(key=lambda x: x[0], reverse=True)
            if len(acc_ckpt_pairs) > save_best_num:
                _, path_to_del = acc_ckpt_pairs.pop()
                if os.path.exists(path_to_del):
                    logging.info('Delete ckpt: %s' % path_to_del)
                    os.system('rm -rf %s' % path_to_del)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logging.info("Saving model checkpoint to %s", output_dir)
            tokenizer.save_vocabulary(output_dir)
            torch.save(optimizer.state_dict(),
                       os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(),
                       os.path.join(output_dir, "scheduler.pt"))
            logging.info("Saving optimizer and scheduler states to %s",
                         output_dir)
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    # logging.info("===================Test==================")
    # evaluate(args, model, test_loader, device)
    return tr_loss


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--comment', default='default', type=str)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    parser.add_argument('--save_dir',
                        required=True,
                        help='path to save checkpoints and logs')
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt')

    parser.add_argument('--sample', default=1.0, type=float)
    parser.add_argument('--valid', action='store_true')
    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--num_train_epochs', default=25, type=int)
    parser.add_argument('--save_steps', default=448, type=int)
    parser.add_argument('--logging_steps', default=448, type=int)
    parser.add_argument(
        '--warmup_proportion',
        default=0.1,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training."
    )
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")

    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default=1e-4, type=float)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(
        os.path.join(args.save_dir, '%s_%s.log' % (args.comment, time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k + ': ' + str(v))

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()
