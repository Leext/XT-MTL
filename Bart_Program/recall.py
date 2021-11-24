import argparse
from ast import parse
import enum
from importlib.machinery import SourceFileLoader
import json
import logging
import os
import pickle
import re
import shutil
import time
from datetime import date
import heapq

import numpy as np
from numpy.lib.utils import source
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm.utils import disp_trim
from utils.misc import MetricLogger, ProgressBar, seed_everything
from multiprocessing import Pool

from .data import DataLoader
from .model import BartCBR
from .program_utils import get_program_seq, rep_fn_map, simi_fn_map

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings

warnings.simplefilter(
    "ignore")  # hide warnings that caused by invalid sparql query


def get_topk(rep, simi_fn, sent_reps, k):
    res = [(simi_fn(rep, rep2), idx) for idx, rep2 in enumerate(sent_reps)]
    res = heapq.nlargest(k, res, key=lambda item: item[0])
    res = np.array([i for _, i in res])
    return res


def _get_topk(args):
    return get_topk(*args)


def get_simi_topk(src_batch, tgt_batch, simi_fn, rep_fn, k=10, parallel=16):
    src_sent_reps = [rep_fn(info['program']) for info in src_batch]
    tgt_sent_reps = [rep_fn(info['program']) for info in tgt_batch]

    if parallel > 1:
        with Pool(parallel) as pool:
            simi_topk = pool.map(_get_topk, [(rep, simi_fn, src_sent_reps, k)
                                             for rep in tgt_sent_reps])
    else:
        simi_topk = [
            get_topk(rep, simi_fn, src_sent_reps, k) for rep in tgt_sent_reps
        ]
    return simi_topk


def get_simi_matrix(origin_batch, rep_fn, simi_fn):
    rel_seqs = [rep_fn(info['program']) for info in origin_batch]
    matrix = [[simi_fn(s1, s2) for s2 in rel_seqs] for s1 in rel_seqs]
    return matrix


def unpack_pair_batch(batch):
    source_ids = torch.cat((batch[0], batch[2]))
    source_mask = torch.cat((batch[1], batch[3]))
    orgin_info = batch[-2] + batch[-1]
    return source_ids, source_mask, orgin_info


def validate(args, model: BartCBR, data, device, tokenizer, simi_matrix_func):
    model.eval()
    all_loss = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            if args.pair:
                source_ids, source_mask, origin_info = unpack_pair_batch(batch)
                source_ids = source_ids.to(device)
                source_mask = source_mask.to(device)
            else:
                origin_info = batch[-1]
                source_ids = batch[0].to(device)
                source_mask = batch[1].to(device)
            sent_rep = model.get_sent_rep(source_ids, source_mask)
            text_simi_mat = torch.tensor(simi_matrix_func(origin_info),
                                         device=device)
            loss = model.similarity_loss(sent_rep, text_simi_mat)
            all_loss.append(loss.cpu().numpy())
    return np.mean(all_loss)


def build_db(model, data, device):
    model.eval()
    sent_reps = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            source_ids = batch[0].to(device)
            source_mask = batch[1].to(device)
            sent_rep = model.get_sent_rep(source_ids, source_mask)
            sent_reps.append(sent_rep)
        sent_rep = torch.cat(sent_reps)
        return sent_rep


def recall(args, model, recall_db, data, device, k=10):
    model.eval()
    recall_index = []
    origin_info_list = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            origin_info_list.extend(batch[-1])
            source_ids = batch[0].to(device)
            source_mask = batch[1].to(device)
            sent_rep = model.get_sent_rep(source_ids, source_mask)
            _, index = torch.topk(torch.matmul(sent_rep, recall_db.T),
                                  k + 1,
                                  dim=1)
            recall_index.extend(index.cpu().numpy())
    return origin_info_list, recall_index


def dump_index(recall_index, origin_info_list, db_origin_info, dump_dir, name,
               rep_fn, simi_fn):
    dump_path = os.path.join(dump_dir, '%s.txt' % name)
    dump_pkl = os.path.join(dump_dir, '%s.pkl' % name)
    with open(dump_path, 'w') as f:
        for idx, rindex in enumerate(recall_index):
            print(idx, file=f)
            if name == 'train':
                rindex = rindex[rindex != idx]
            rindex = rindex[:rindex.size - 1]
            recall_index[idx] = rindex
            src_info = origin_info_list[idx]
            print(src_info['question'], file=f)
            if name != 'test':
                src_rels = rep_fn(src_info['program'])
                print(get_program_seq(src_info['program']), file=f)
            for i in rindex:
                tgt_info = db_origin_info[i]
                tgt_rels = rep_fn(tgt_info['program'])
                print('\tQ: %s' % tgt_info['question'], file=f)
                if name != 'test':
                    print('\t%.5f  %s' %
                          (simi_fn(src_rels, tgt_rels),
                           get_program_seq(tgt_info['program'])),
                          file=f)
            print(file=f)
    with open(dump_pkl, 'wb') as f:
        pickle.dump(recall_index, f)


def predict(args):
    device = torch.device(args.device)

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    test_loader = DataLoader(vocab_json, test_pt, args.batch_size)
    vocab = train_loader.vocab

    logging.info("Create model.........")
    # tokenizer = BartTokenizer.from_pretrained(args.ckpt)
    model = BartCBR(args.ckpt)
    model = model.to(device)
    logging.info(model)

    logging.info('Build db.........')
    recall_db = build_db(model, train_loader, device)
    rep_fn = rep_fn_map.get(args.rep_fn)
    simi_fn = simi_fn_map.get(args.simi_fn)
    if not os.path.isdir(args.recall_dump):
        os.makedirs(args.recall_dump)
    db_origin_info = None
    for name, data_loader in zip(('train', 'valid', 'test'),
                                 (train_loader, val_loader, test_loader)):
        logging.info('Recall for %s set' % name)
        origin_info_list, recall_index = recall(args,
                                                model,
                                                recall_db,
                                                data_loader,
                                                device,
                                                k=args.k)
        if name == 'train':
            db_origin_info = origin_info_list
        print('%s # samples: %d' % (name, len(origin_info_list)))
        print('%s # recall index: %d' % (name, len(recall_index)))
        dump_index(recall_index, origin_info_list, db_origin_info,
                   args.recall_dump, name, rep_fn, simi_fn)


def predict_rule(args):
    logging.info('Loading raw data......')
    train_json = os.path.join(args.input_dir, 'train.json')
    val_json = os.path.join(args.input_dir, 'val.json')
    train_json = json.load(open(train_json))
    val_json = json.load(open(val_json))
    rep_fn = rep_fn_map.get(args.rep_fn)
    simi_fn = simi_fn_map.get(args.simi_fn)
    if not os.path.isdir(args.recall_dump):
        os.makedirs(args.recall_dump)
    for name, data_json in zip(('train', 'valid'), (train_json, val_json)):
        logging.info('Cal top k for %s set' % name)
        recall_index = get_simi_topk(data_json, data_json, simi_fn, rep_fn,
                                     args.k, args.parallel)
        logging.info('%s # samples: %d' % (name, len(data_json)))
        logging.info('%s # recall index: %d' % (name, len(recall_index)))
        dump_index(recall_index, data_json, data_json, args.recall_dump, name,
                   rep_fn, simi_fn)


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
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--rule', action='store_true')
    parser.add_argument('--rep_fn', type=str, required=True)
    parser.add_argument('--simi_fn', type=str, required=True)

    # training parameters
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--parallel', type=int, default=16)

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
        logging.info(k + ': ' + str(v))

    seed_everything(666)

    if args.rule:
        predict_rule(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()