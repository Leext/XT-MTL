import argparse
from ast import parse
import json
import logging
import os
import shutil
import time
from datetime import date
import heapq

import torch
import torch.nn as nn
# from .sparql_engine import get_sparql_answer
import torch.optim as optim
from tqdm import tqdm
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer)
from utils.load_kb import DataForSPARQL
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.misc import MetricLogger, ProgressBar, seed_everything

from .data import DataLoader, PairDataLoader
from .model import BartCBR
from .recall import get_simi_matrix, validate, get_rel_seq, get_func_rel_rep, get_func_rep, unpack_pair_batch, rep_fn_map, simi_fn_map

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings

warnings.simplefilter(
    "ignore")  # hide warnings that caused by invalid sparql query


def train(args):
    device = torch.device(args.device)
    logging.info('training device: %s' % str(device))

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    if args.pair:
        train_recall_index = os.path.join(args.recall_index, 'train.pkl')
        train_loader = PairDataLoader(vocab_json,
                                      train_pt,
                                      train_recall_index,
                                      args.batch_size,
                                      args.k,
                                      training=True)
        valid_recall_index = os.path.join(args.recall_index, 'valid.pkl')
        val_loader = PairDataLoader(vocab_json, val_pt, valid_recall_index,
                                    args.batch_size, args.k)
    else:
        train_loader = DataLoader(vocab_json,
                                  train_pt,
                                  args.batch_size,
                                  training=True)
        val_loader = DataLoader(vocab_json, val_pt, 64)
    # test_loader = DataLoader(vocab_json, test_pt, 64)

    vocab = train_loader.vocab
    # kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))
    # rule_executor = RuleExecutor(vocab, os.path.join(args.input_dir, 'kb.json'))
    logging.info("Create model.........")
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    model = BartCBR(args.model_name_or_path)
    model = model.to(device)
    simi_fn = simi_fn_map.get(args.simi_fn)
    rep_fn = rep_fn_map.get(args.rep_fn)
    simi_matrix_func = lambda x: get_simi_matrix(x, rep_fn, simi_fn)
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
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(
            args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        # Train!
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_loader.dataset))
        logging.info("  Num Epochs = %d", args.num_train_epochs)
        logging.info("  Gradient Accumulation steps = %d",
                     args.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path
                      ) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_loader) //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_loader) // args.gradient_accumulation_steps)
        logging.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logging.info("  Continuing training from epoch %d", epochs_trained)
        logging.info("  Continuing training from global step %d", global_step)
        logging.info("  Will skip the first %d steps in the first epoch",
                     steps_trained_in_current_epoch)
    logging.info('Checking...')
    logging.info("===================Dev==================")
    valid_loss = validate(args, model, val_loader, device, tokenizer,
                          simi_matrix_func)
    tr_loss, logging_loss = 0.0, 0.0
    save_best_num = 3
    loss_ckpt_pairs = [(1e10, '')]
    model.zero_grad()
    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        for step, batch in enumerate(train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            pad_token_id = tokenizer.pad_token_id
            if args.pair:
                source_ids, source_mask, origin_info = unpack_pair_batch(batch)
            else:
                source_ids, source_mask = batch[0], batch[1]
                origin_info = batch[-1]
            source_ids = source_ids.to(device)
            source_mask = source_mask.to(device)
            sent_rep = model.get_sent_rep(source_ids, source_mask)
            text_simi_mat = torch.tensor(simi_matrix_func(origin_info),
                                         device=device)
            loss = model.similarity_loss(sent_rep, text_simi_mat)
            loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        logging.info('\n')
        logging.info("===================Dev==================")
        valid_loss = validate(args, model, val_loader, device, tokenizer,
                              simi_matrix_func)
        logging.info('Epoch %d: valid loss %.5f' % (epoch, valid_loss))
        if loss_ckpt_pairs[-1][0] > valid_loss:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, args.comment,
                                      "epoch_%d" % epoch)
            loss_ckpt_pairs.append((valid_loss, output_dir))
            loss_ckpt_pairs.sort(key=lambda x: x[0])
            if len(loss_ckpt_pairs) > save_best_num:
                _, path_to_del = loss_ckpt_pairs.pop()
                if os.path.exists(path_to_del):
                    logging.info('Delete ckpt: %s' % path_to_del)
                    os.system('rm -rf %s' % path_to_del)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.bart_gen.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logging.info("Saving model checkpoint to %s" % output_dir)
            tokenizer.save_vocabulary(output_dir)
            torch.save(optimizer.state_dict(),
                       os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(),
                       os.path.join(output_dir, "scheduler.pt"))
            logging.info("Saving optimizer and scheduler states to %s",
                         output_dir)
            logging.info("\n")
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    # logging.info("===================Test==================")
    # evaluate(args, model, test_loader, device)
    return global_step, tr_loss / global_step


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
    parser.add_argument('--recall_index')

    # training parameters
    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--rep_fn', required=True)
    parser.add_argument('--simi_fn', required=True)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--num_train_epochs', default=25, type=int)
    parser.add_argument('--device', default='cuda:0')
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
        os.path.join(args.save_dir, '%s_%s.log' % (args.comment, time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()
