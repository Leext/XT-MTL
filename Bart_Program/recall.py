import argparse
import enum
import json
import logging
import os
import pickle
import re
import shutil
import time
from datetime import date

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer)
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.misc import MetricLogger, ProgressBar, seed_everything

from .data import DataLoader
from .model import BartCBR
from .preprocess import get_func_rel_seq, get_program_seq, get_rel_seq

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings

warnings.simplefilter(
    "ignore")  # hide warnings that caused by invalid sparql query

eps = 1e-6


def set_f1(s1: set, s2: set):
    tp = len(s1 & s2)
    fp = len(s2 - s1)
    fn = len(s1 - s2)
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    return f1


def get_simi_matrix(origin_batch):
    rel_seqs = [
        set(get_rel_seq(info['program'])) for info in origin_batch
    ]
    matrix = [[set_f1(s1, s2) for s2 in rel_seqs] for s1 in rel_seqs]
    return matrix


def validate(args, model: BartCBR, data, device, tokenizer):
    model.eval()
    all_loss = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            origin_info = batch[-1]
            source_ids = batch[0].to(device)
            source_mask = batch[1].to(device)
            sent_rep = model.get_sent_rep(source_ids, source_mask)
            text_simi_mat = torch.tensor(get_simi_matrix(origin_info),
                                         device=device)
            loss = model.similarity_loss(sent_rep, text_simi_mat)
            all_loss.append(loss.cpu().numpy())
    return np.mean(all_loss)


def recall(args, model, data, device, tokenizer):
    model.eval()
    k = 10
    sent_reps = []
    origin_info_list = []
    recall_index = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            origin_info = batch[-1]
            source_ids = batch[0].to(device)
            source_mask = batch[1].to(device)
            sent_rep = model.get_sent_rep(source_ids, source_mask)
            sent_reps.append(sent_rep)
            origin_info_list.extend(origin_info)
        sent_rep = torch.cat(sent_reps)
        for sent in tqdm(sent_rep, total=sent_rep.size(0)):
            _, index = torch.topk(torch.matmul(sent, sent_rep.T), k, dim=0)
            recall_index.append(index.cpu().numpy())
    return origin_info_list, recall_index


def predict(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'test.pt')
    train_loader = DataLoader(vocab_json,
                              train_pt,
                              args.batch_size,
                              training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    vocab = train_loader.vocab
    logging.info("Create model.........")
    tokenizer = BartTokenizer.from_pretrained(args.ckpt)
    model = BartCBR(args.ckpt)
    model = model.to(device)
    logging.info(model)
    origin_info_list, recall_index = recall(args, model, train_loader, device,
                                            tokenizer)
    if not os.path.isdir(args.recall_dump):
        os.makedirs(args.recall_dump)
    dump_path = os.path.join(args.recall_dump, 'dump.txt')
    dump_pkl = os.path.join(args.recall_dump, 'dump.pkl')
    with open(dump_pkl, 'wb') as f:
        pickle.dump(recall_index, f)
    with open(dump_path, 'w') as f:
        for idx, rindex in enumerate(recall_index):
            print(idx, file=f)
            src_info = origin_info_list[idx]
            src_rels = set(get_rel_seq(src_info['program']))
            print(src_info['question'], file=f)
            print(get_program_seq(src_info['program']), file=f)
            for i in rindex:
                tgt_info = origin_info_list[i]
                tgt_rels = set(get_rel_seq(tgt_info['program']))
                if src_info['question'] != tgt_info['question']:
                    print('\tQ: %s' % tgt_info['question'], file=f)
                    print('\t%.5f  %s' %
                          (set_f1(src_rels, tgt_rels),
                           get_program_seq(tgt_info['program'])),
                          file=f)
            print(file=f)


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir',
                        required=True,
                        help='path to save checkpoints and logs')
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--recall_dump', required=True)

    # training parameters
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')

    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)
    # parser.add_argument('--top_p', default=)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default=1e-4, type=float)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(
        os.path.join(args.save_dir, '{}.predict.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    seed_everything(666)

    predict(args)


if __name__ == '__main__':
    main()