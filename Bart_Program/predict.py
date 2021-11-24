import argparse
from collections import defaultdict
import json
import logging
import os
import re
import shutil
import time
from datetime import date

import torch
import torch.nn as nn
import torch.optim as optim
# from tqdm import tqdm
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer)
from utils.load_kb import DataForSPARQL
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.misc import MetricLogger, ProgressBar, seed_everything

from Bart_Program.executor_rule import RuleExecutor
from Bart_Program.program_utils import edit_similarity, get_program_seq, get_rel_rep, get_rel_seq, program2seq, seq2program, get_func_seq

from .data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings

warnings.simplefilter(
    "ignore")  # hide warnings that caused by invalid sparql query


def tqdm(data, total=0, desc=''):
    # logging.info('start %s' % desc)
    yield from data
    # logging.info('%s done, total %d' % (desc, total))


def post_process(text):
    pattern = re.compile(r'".*?"')
    nes = []
    for item in pattern.finditer(text):
        nes.append((item.group(), item.span()))
    pos = [0]
    for name, span in nes:
        pos += [span[0], span[1]]
    pos.append(len(text))
    assert len(pos) % 2 == 0
    assert len(pos) / 2 == len(nes) + 1
    chunks = [text[pos[i]:pos[i + 1]] for i in range(0, len(pos), 2)]
    for i in range(len(chunks)):
        chunks[i] = chunks[i].replace('?', ' ?').replace('.', ' .')
    bingo = ''
    for i in range(len(chunks) - 1):
        bingo += chunks[i] + nes[i][0]
    bingo += chunks[-1]
    return bingo


def vis(args, kb, model, data, device, tokenizer):
    while True:
        # text = 'Who is the father of Tony?'
        # text = 'Donald Trump married Tony, where is the place?'
        text = input('Input your question:')
        with torch.no_grad():
            input_ids = tokenizer.batch_encode_plus([text],
                                                    max_length=512,
                                                    pad_to_max_length=True,
                                                    return_tensors="pt",
                                                    truncation=True)
            source_ids = input_ids['input_ids'].to(device)
            outputs = model.generate(
                input_ids=source_ids,
                max_length=500,
            )
            outputs = [
                tokenizer.decode(output_id,
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
                for output_id in outputs
            ]
            outputs = [post_process(output) for output in outputs]
            print(outputs[0])


def predict(args, kb, model, data, device, tokenizer, executor):
    model.eval()
    count, correct = 0, 0
    pattern = re.compile(r'(.*?)\((.*?)\)')
    with torch.no_grad():
        all_outputs = []
        for batch in tqdm(data, total=len(data)):
            batch = batch[:3]
            # source_ids, source_mask, choices = [x.to(device) for x in batch]
            if args.type == 'cbr':
                source_ids = batch[2].to(device)
            else:
                source_ids = batch[0].to(device)
            outputs = model.generate(
                input_ids=source_ids,
                max_length=500,
            )

            all_outputs.extend(outputs.cpu().numpy())
            # break  ???

        outputs = [
            tokenizer.decode(output_id,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True)
            for output_id in all_outputs
        ]
        # questions = [tokenizer.decode(source_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for source_id in all_answers]
        with open(os.path.join(args.save_dir, 'predict.txt'), 'w') as f:
            for output in tqdm(outputs):
                func_list, inputs_list = seq2program(output)
                ans = executor.forward(func_list,
                                       inputs_list,
                                       ignore_error=True)
                if ans == None:
                    ans = 'no'
                f.write(ans + '\n')


def validate(args, kb, model, data, device, tokenizer, executor):
    model.eval()
    count, correct = 0, 0
    if args.filter_rels != '':
        filter_rels = eval(args.filter_rels)
    else:
        filter_rels = set()

    with torch.no_grad():
        all_outputs = []
        all_answers = []
        all_info = []
        for batch in tqdm(data, total=len(data), desc='valid generate'):
            if args.type == 'cbr':
                source_ids = batch[2].to(device)
            else:
                source_ids = batch[0].to(device)
            answer = batch[-2]
            outputs = model.generate(
                input_ids=source_ids,
                max_length=500,
            )

            all_outputs.extend(outputs.cpu().numpy())
            all_answers.extend(answer.cpu().numpy())
            all_info.extend(batch[-1])
            # break

        outputs = [
            tokenizer.decode(output_id,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True)
            for output_id in all_outputs
        ]
        # questions = [tokenizer.decode(source_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for source_id in all_answers]
        # total = []
        incorrect_list = []
        for output, info in tqdm(zip(outputs, all_info), total=len(all_info)):
            if args.filter_rels != '':
                rels = get_rel_rep(info['program'])
                if len(rels & filter_rels) == 0:
                    continue
            func_list, inputs_list = seq2program(output)
            if args.revise:
                func_list, inputs_list = executor.revise_program(
                    info['question'], func_list, inputs_list)
                output = program2seq(func_list, inputs_list)
            ans = executor.forward(func_list, inputs_list, ignore_error=True)
            if ans == None:
                ans = 'no'
            if ans == info['answer']:
                correct += 1
            else:
                incorrect_list.append((info, output, ans))
            count += 1
        acc = correct / count
        return {'q': (acc, incorrect_list)}


def validate_prompt(args, kb, model, data, device, tokenizer, executor):
    def stat_results_qa(pred_ans_list, all_info):
        count, correct = 0, 0
        rel_count, rel_correct = 0, 0
        incorrect_list = []
        if args.filter_rels != '':
            filter_rels = eval(args.filter_rels)
        else:
            filter_rels = set()
        for pred_ans, info in tqdm(zip(pred_ans_list, all_info),
                                   total=len(all_info),
                                   desc='stat results'):
            has_rel = args.filter_rels != '' and \
                    len(get_rel_rep(info['program']) & filter_rels) != 0
            func_list, inputs_list = seq2program(pred_ans)
            if args.revise:
                func_list, inputs_list = executor.revise_program(
                    info['question'], func_list, inputs_list)
                pred_ans = program2seq(func_list, inputs_list)
            ans = executor.forward(func_list, inputs_list, ignore_error=True)
            if ans == None:
                ans = 'no'
            if ans == info['answer']:
                correct += not has_rel
                rel_correct += has_rel
            else:
                incorrect_list.append((info, pred_ans, ans))
            count += not has_rel
            rel_count += has_rel
        acc = correct / count
        if args.filter_rels != '':
            logging.info('[Valid] Unseen Rel acc %.5f' %
                         (rel_correct / rel_count))
            logging.info('[Valid] Seen Rel acc %.5f' % (correct / count))
        return acc, incorrect_list

    def stat_results_rel(pred_ans_list, all_info, seq_fn):
        count, correct = 0, 0
        incorrect_list = []
        for pred_ans, info in tqdm(zip(pred_ans_list, all_info),
                                   total=len(all_info),
                                   desc='stat results'):
            true_rels = seq_fn(info['program'])
            pred_rels = [
                s.strip() for s in pred_ans.split(';') if s.strip() != ''
            ]
            if pred_rels == true_rels:
                correct += 1
            else:
                incorrect_list.append((info, pred_rels, true_rels))
            count += 1
        acc = correct / count
        return acc, incorrect_list

    model.eval()
    tasks = args.tasks.split(',')
    task2tgt = dict()
    for t in tasks:
        key, tgt, w = t.split(':')
        task2tgt[key] = (tgt, float(w))
    with torch.no_grad():
        all_answers = []
        all_info = []
        all_task_outputs = defaultdict(list)
        for batch in tqdm(data, total=len(data), desc='valid generate'):
            answer = batch['answer']
            for task in task2tgt.keys():
                task_key = '%s_ids' % task
                if task_key not in batch:
                    continue
                inputs = batch[task_key].to(device)
                outputs = model.generate(input_ids=inputs, max_length=500)
                all_task_outputs[task].extend([
                    tokenizer.decode(output,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
                    for output in outputs.cpu().numpy()
                ])

            all_answers.extend(answer.cpu().numpy())
            all_info.extend(batch['origin_info'])

        results = dict()
        for task in all_task_outputs.keys():
            if 'rel' in task:
                results[task] = stat_results_rel(all_task_outputs[task],
                                                 all_info, get_rel_seq)
            elif 'func' in task:
                results[task] = stat_results_rel(all_task_outputs[task],
                                                 all_info, get_func_seq)

            elif task.startswith('q'):
                results[task] = stat_results_qa(all_task_outputs[task],
                                                all_info)

        return results


def predict_prompt(args, model, data, device, tokenizer, executor):
    model.eval()
    with torch.no_grad():
        all_answers = []
        all_outputs = []
        all_info = []
        for batch in tqdm(data, total=len(data), desc='predict generate'):
            inputs = batch['%s_ids' % args.main_task].to(device)
            all_info.extend(batch['origin_info'])
            outputs = model.generate(input_ids=inputs, max_length=500)
            all_outputs.extend([
                tokenizer.decode(output,
                                 skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
                for output in outputs.cpu().numpy()
            ])
        for seq, info in zip(all_outputs, all_info):
            func_list, inputs_list = seq2program(seq)
            if args.revise:
                func_list, inputs_list = executor.revise_program(
                    info['question'], func_list, inputs_list)
                seq = program2seq(func_list, inputs_list)
            ans = executor.forward(func_list, inputs_list, ignore_error=True)
            if ans == None:
                ans = 'no'
            all_answers.append(ans)
        return all_answers


def validate_prompt_rel(args, model, data, device, tokenizer):
    model.eval()
    count, correct = 0, 0
    with torch.no_grad():
        all_outputs = []
        all_info = []
        for batch in tqdm(data, total=len(data)):
            source_ids = batch[0].to(device)
            outputs = model.generate(
                input_ids=source_ids,
                max_length=500,
            )

            all_outputs.extend(outputs.cpu().numpy())
            all_info.extend(batch[-1])
            # break

        outputs = [
            tokenizer.decode(output_id,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True)
            for output_id in all_outputs
        ]
        incorrect_list = []
        cum_simi = 0
        for output, info in tqdm(zip(outputs, all_info)):
            true_rels = get_rel_seq(info['program'])
            pred_rels = [
                s.strip() for s in output.split(';') if s.strip() != ''
            ]
            if true_rels == pred_rels:
                correct += 1
            else:
                incorrect_list.append((info, output, pred_rels))
            cum_simi += edit_similarity(true_rels, pred_rels)
            count += 1
        acc = correct / count
        edit_simi = cum_simi / count
        logging.info('acc: {}'.format(acc))
        logging.info('edit similarity: {}'.format(edit_simi))

        return acc, edit_simi, incorrect_list


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    kb = DataForSPARQL(os.path.join('./dataset/', 'kb.json'))
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig,
                                                  BartForConditionalGeneration,
                                                  BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.ckpt)
    model = model_class.from_pretrained(args.ckpt)
    model = model.to(device)
    logging.info(model)
    rule_executor = RuleExecutor(vocab, os.path.join('./dataset/', 'kb.json'))
    # validate(args, kb, model, val_loader, device, tokenizer, rule_executor)
    predict(args, kb, model, val_loader, device, tokenizer, rule_executor)


    # vis(args, kb, model, val_loader, device, tokenizer)
def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir',
                        required=True,
                        help='path to save checkpoints and logs')
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt', required=True)

    # training parameters
    parser.add_argument('--batch_size', default=256, type=int)
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

    train(args)


if __name__ == '__main__':
    main()
