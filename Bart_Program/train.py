import argparse
import json
import logging
import os
import shutil
import time
from datetime import date

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .sparql_engine import get_sparql_answer
import torch.optim as optim
from tqdm import tqdm
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer)
from utils.load_kb import DataForSPARQL
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.misc import MetricLogger, ProgressBar, seed_everything

from Bart_Program.executor_rule import RuleExecutor
from Bart_Program.predict import validate, validate_prompt, validate_prompt_rel
import Bart_Program.kb_pretrain as KBP

from .data import CBRDataLoader, DataLoader, PromptDataLoader
from .program_utils import get_program_seq

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings

warnings.simplefilter(
    "ignore")  # hide warnings that caused by invalid sparql query


def train_step(args, model, train_loader, tokenizer, device, optimizer,
               scheduler):
    epoch_loss = 0.0
    for step, batch in enumerate(train_loader):
        # Skip past any already trained steps if resuming training
        # if steps_trained_in_current_epoch > 0:
        #     steps_trained_in_current_epoch -= 1
        #     continue
        model.train()
        pad_token_id = tokenizer.pad_token_id
        if args.cbr:
            source_ids, source_mask, y = batch[2], batch[3], batch[-3]
        else:
            source_ids, source_mask, y = batch[0], batch[1], batch[-3]
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
        epoch_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
    return epoch_loss, {}


def train_step_prompt(args, model, train_loader, tokenizer, device, optimizer,
                      scheduler):
    def part_step(source_ids, source_mask, y):
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
        return outputs[:2]

    epoch_loss = 0.0
    tasks = args.tasks.split(',')
    task2tgt = dict()
    for t in tasks:
        key, tgt, w = t.split(':')
        task2tgt[key] = (tgt, float(w))
    all_task_loss = {task: 0 for task in task2tgt.keys()}
    if args.kld:
        all_task_loss['kld'] = 0
    for step, batch in enumerate(train_loader):
        model.train()
        pad_token_id = tokenizer.pad_token_id
        task_loss = []
        q_logits, q_cbr_logits = None, None
        for task, (tgt, w) in task2tgt.items():
            part_loss, logits = part_step(batch['%s_ids' % task],
                                          batch['%s_mask' % task],
                                          batch['%s_ids' % tgt])
            task_loss.append(w * part_loss)
            if task == 'q':
                q_logits = logits
            if task == 'q_cbr':
                q_cbr_logits = logits
            all_task_loss[task] += part_loss.item()
        if args.kld and q_logits is not None and q_cbr_logits is not None:
            kld_loss = args.kld_weight * F.kl_div(F.log_softmax(q_cbr_logits),
                                                  F.log_softmax(q_logits),
                                                  reduce='batchmean',
                                                  log_target=True)
            task_loss.append(kld_loss)
            all_task_loss['kld'] += kld_loss.item()
        loss = torch.sum(torch.stack(task_loss))
        loss.backward()
        epoch_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
    return epoch_loss, all_task_loss


def save_checkpoint(args,
                    model,
                    results,
                    tokenizer,
                    optimizer,
                    scheduler,
                    acc_ckpt_pairs,
                    cur_acc,
                    epoch,
                    save_best_num=3):
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, args.comment,
                              'epoch_%d' % epoch)
    acc_ckpt_pairs.append((cur_acc, epoch, output_dir))
    acc_ckpt_pairs.sort(key=lambda x: x[0], reverse=True)
    if len(acc_ckpt_pairs) > save_best_num:
        _, _, path_to_del = acc_ckpt_pairs.pop()
        if os.path.exists(path_to_del):
            logging.info('Delete ckpt: %s' % path_to_del)
            os.system('rm -rf %s' % path_to_del)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (model.module if hasattr(model, "module") else model
                     )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logging.info("Saving model checkpoint to %s", output_dir)
    tokenizer.save_vocabulary(output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir,
                                                    "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir,
                                                    "scheduler.pt"))
    logging.info("Saving optimizer and scheduler states to %s", output_dir)
    dump_error_cases(results, output_dir)


def dump_error_cases(results, output_dir):
    for task, result in results.items():
        with open(os.path.join(output_dir, '%s_incorrect_pred.txt' % task),
                  'w') as f:
            print('Total %d cases, acc %.8f' % (len(result[1]), result[0]),
                  file=f)
            if task == 'q' or task == 'q_cbr':
                for info, output, pred_ans in result[1]:
                    print('Q: %s' % info['question'], file=f)
                    print('Gold: %s' % get_program_seq(info['program']),
                          file=f)
                    print('Gold Ans: %s' % info['answer'], file=f)
                    print('Pred: %s' % output, file=f)
                    print('Pred Ans: %s' % pred_ans, file=f)
            else:
                for info, pred_ans, true_ans in result[1]:
                    print('Q: %s' % info['question'], file=f)
                    print('Gold: %s' % str(true_ans), file=f)
                    print('Pred: %s' % str(pred_ans), file=f)


def train(args):
    device = torch.device(args.device)

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')

    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig,
                                                  BartForConditionalGeneration,
                                                  BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model = model.to(device)
    logging.info(model)

    if args.kbp:
        kb_train, kb_val = KBP.prepare_data(
            args, os.path.join(args.input_dir, 'kb.json'), tokenizer)

    if args.type == 'cbr':
        loader_class = CBRDataLoader
        train_step_fn = train_step
        valid_step_fn = validate
    elif args.type == 'default':
        loader_class = DataLoader
        train_step_fn = train_step
        valid_step_fn = validate
    elif args.type == 'prompt':
        loader_class = PromptDataLoader
        train_step_fn = train_step_prompt
        valid_step_fn = validate_prompt
    train_loader = loader_class(vocab_json,
                                train_pt,
                                args.batch_size,
                                training=True,
                                ratio=args.sample)
    val_loader = loader_class(vocab_json, val_pt, 64)
    # test_loader = DataLoader(vocab_json, test_pt, 64)

    vocab = train_loader.vocab
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))
    rule_executor = RuleExecutor(vocab, os.path.join(args.input_dir,
                                                     'kb.json'))

    t_total = len(
        train_loader
    ) // args.gradient_accumulation_steps * args.num_train_epochs  # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bart_param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [{
        'params': [p for n, p in bart_param_optimizer
                        if not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay,
        'lr': args.learning_rate},
        {'params': [p for n, p in bart_param_optimizer
                        if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,
        'lr': args.learning_rate
    }] # yapf: disable
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
    tasks = args.tasks.split(',')
    task2tgt = dict()
    for t in tasks:
        key, tgt, w = t.split(':')
        task2tgt[key] = (tgt, float(w))
    logging.info('task2tgt %s' % str(task2tgt))
    logging.info('Checking...')
    results = valid_step_fn(args, kb, model, val_loader, device, tokenizer,
                            rule_executor)

    output_dir = os.path.join(args.output_dir, args.comment)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    dump_error_cases(results, output_dir)
    if args.valid:
        return
    model.zero_grad()
    acc_ckpt_pairs = [(0, -1, 'NOT_EXIST')]
    for epoch in range(int(args.num_train_epochs)):
        if args.kbp and (epoch % args.kbp_period) == 0:
            kb_train_loss = KBP.train_step(args, model, kb_train, device,
                                           tokenizer, optimizer, scheduler,
                                           args.kbp_sample)
            logging.info('[Train] Epoch %d: kb pretrain loss %.8f' %
                         (epoch, kb_train_loss))

        epoch_loss, loss_results = train_step_fn(args, model, train_loader,
                                                 tokenizer, device, optimizer,
                                                 scheduler)
        loss_stat = ['final loss %.8f' % (epoch_loss / len(train_loader))] + [
            '%s loss %.8f' % (task, val / len(train_loader))
            for task, val in loss_results.items()
        ]
        loss_stat = ',\t'.join(loss_stat)
        logging.info('[Train] Epoch %d: %s' % (epoch, loss_stat))
        if epoch >= args.start_valid_epoch:
            results = valid_step_fn(args, kb, model, val_loader, device,
                                    tokenizer, rule_executor)
            acc = results[args.main_task][0]
            stat = [
                '%s acc %.8f' % (task, result[0])
                for task, result in results.items()
            ]
            stat = ',\t'.join(stat)
            logging.info('[Valid] Epoch %d: %s' % (epoch, stat))
            if acc_ckpt_pairs[-1][0] < acc:
                save_checkpoint(args, model, results, tokenizer, optimizer,
                                scheduler, acc_ckpt_pairs, acc, epoch)
            elif epoch - acc_ckpt_pairs[0][1] >= 10:
                logging.info('Early stop, best at epoch %d, acc %.8f' %
                             (acc_ckpt_pairs[0][1], acc_ckpt_pairs[0][0]))
                break
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    # logging.info("===================Test==================")
    # evaluate(args, model, test_loader, device)


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

    parser.add_argument('--cbr', action='store_true')
    parser.add_argument('--type', default='default', type=str)
    parser.add_argument('--tasks', default='q:program:1', type=str)
    parser.add_argument('--main_task', default='q', type=str)
    parser.add_argument('--revise', action='store_true')
    parser.add_argument('--sample', default=1.0, type=float)
    parser.add_argument('--valid', action='store_true')
    # training parameters
    parser.add_argument('--kbp', action='store_true')
    parser.add_argument('--kbp_sample', default=0.1, type=float)
    parser.add_argument('--kbp_period', default=2, type=int)
    parser.add_argument('--kbp_mode', default='triple')
    parser.add_argument('--kld', action='store_true')
    parser.add_argument('--kld_weight', default=0.1, type=float)
    parser.add_argument('--start_valid_epoch', default=0, type=int)
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
