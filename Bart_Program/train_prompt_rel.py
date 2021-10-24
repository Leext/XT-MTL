import argparse
import json
import logging
import os
import shutil
import time
from datetime import date

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

from Bart_Program.executor_rule import RuleExecutor
from Bart_Program.predict import validate, validate_prompt_rel

from .data import PromptRelDataLoader
from .program_utils import get_program_seq, get_rel_seq

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings

warnings.simplefilter(
    "ignore")  # hide warnings that caused by invalid sparql query


def train(args):
    device = torch.device(args.device)

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')

    loader_class = PromptRelDataLoader
    train_loader = loader_class(vocab_json,
                                train_pt,
                                args.batch_size,
                                training=True,
                                ratio=args.sample)
    val_loader = loader_class(vocab_json, val_pt, 64)
    # test_loader = DataLoader(vocab_json, test_pt, 64)

    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig,
                                                  BartForConditionalGeneration,
                                                  BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model = model.to(device)
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

    logging.info('Checking...')
    logging.info("===================Dev==================")
    acc, edit_simi, valid_error_list = validate_prompt_rel(
        args, model, val_loader, device, tokenizer)
    if args.valid:
        output_dir = os.path.join(args.output_dir, args.comment)
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'incorrect_pred.txt'), 'w') as f:
            for info, output, pred_ans in valid_error_list:
                print('Q: %s' % info['question'], file=f)
                print('Gold Rel: %s' % get_rel_seq(info['program']), file=f)
                print('Pred Rel: %s' % pred_ans, file=f)
        return
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    prefix = 0
    save_best_num = 3
    acc_ckpt_pairs = [(0, 'NOT_EXIST')]
    for epoch in range(int(args.num_train_epochs)):
        logging.info("===================Train==================")
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            # Skip past any already trained steps if resuming training
            # if steps_trained_in_current_epoch > 0:
            #     steps_trained_in_current_epoch -= 1
            #     continue
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
            loss = outputs[0]
            loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
        logging.info('Epoch %d: loss %.5f' %
                     (epoch, epoch_loss / len(train_loader)))
        logging.info("===================Dev==================")
        acc, edit_simi, incorrect_list = validate_prompt_rel(
            args, model, val_loader, device, tokenizer)
        logging.info('Epoch %d: acc %.5f,\tedit similarity %.5f' %
                     (epoch, acc, edit_simi))
        if acc_ckpt_pairs[-1][0] < acc:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, args.comment,
                                      'epoch_%d' % epoch)
            acc_ckpt_pairs.append((acc, output_dir))
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
            with open(os.path.join(output_dir, 'incorrect_pred.txt'),
                      'w') as f:
                for info, output, pred_ans in incorrect_list:
                    print('Q: %s' % info['question'], file=f)
                    print('Gold Rel: %s' % get_rel_seq(info['program']),
                          file=f)
                    print('Pred Rel: %s' % pred_ans, file=f)
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
        logging.info(k + ': ' + str(v))

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()
