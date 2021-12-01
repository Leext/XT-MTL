"""
We need the last function to help extract the final answer of SPARQL, used in check_sparql
"""

import argparse
import random
import json
import os
import pickle
import re
from itertools import chain

import numpy as np
from tqdm import tqdm
from transformers import *

from Bart_Program.program_utils import get_func_seq, get_rel_seq, get_program_seq
from Bart_Program.program_permute import permute_program_seq


def encode_dataset(dataset, vocab, tokenizer, name, boost=False):
    questions = []
    programs = []
    choices = []
    answers = []
    for item in tqdm(dataset):
        count = 1
        if name != 'test':
            program = item['program']
            program = get_program_seq(program)
            if boost and name == 'train':
                p_list = permute_program_seq(program)
            else:
                p_list = [program]
            count = len(p_list)
            for p in p_list:
                programs.append(p)
                answers.append(vocab['answer_token_to_idx'].get(item['answer']))
        question = item['question']
        questions.extend([question] * count)
        _ = [vocab['answer_token_to_idx'][w] for w in item['choices']]
        choices.extend([_] * count)

    input_ids = tokenizer.batch_encode_plus(questions,
                                            padding=True,
                                            max_length=1024,
                                            truncation=True)
    source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    if name != 'test':
        target_ids = tokenizer.batch_encode_plus(programs,
                                                 max_length=1024,
                                                 padding=True,
                                                 truncation=True)
        target_ids = np.array(target_ids['input_ids'], dtype=np.int32)
    else:
        target_ids = np.array([], dtype=np.int32)
    choices = np.array(choices, dtype=np.int32)
    answers = np.array(answers, dtype=np.int32)
    return source_ids, source_mask, target_ids, choices, answers


def cbr_encode_dataset(dataset,
                       recall_db,
                       recall_index,
                       vocab,
                       tokenizer,
                       test=False,
                       k_cases=10):
    assert len(dataset) == len(recall_index)
    questions = []
    programs = []
    programs_seqs = [get_program_seq(item['program']) for item in recall_db]
    questions_plus_cases = []
    answers = []
    for item, index_list in tqdm(zip(dataset, recall_index),
                                 total=len(dataset)):
        questions.append(item['question'])
        sents = [item['question']]
        for i in index_list[:k_cases]:
            sents.append(recall_db[i]['question'])
            sents.append(programs_seqs[i])
        question = ' </s> '.join(sents)
        questions_plus_cases.append(question)
        if not test:
            programs.append(get_program_seq(item['program']))
            answers.append(vocab['answer_token_to_idx'].get(item['answer']))

    input_ids = tokenizer.batch_encode_plus(questions,
                                            padding=True,
                                            max_length=1024,
                                            truncation=True)
    source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    input_ids = tokenizer.batch_encode_plus(questions_plus_cases,
                                            padding=True,
                                            max_length=1024,
                                            truncation=True)
    source_ids_cbr = np.array(input_ids['input_ids'], dtype=np.int32)
    source_mask_cbr = np.array(input_ids['attention_mask'], dtype=np.int32)
    if not test:
        target_ids = tokenizer.batch_encode_plus(programs,
                                                 padding=True,
                                                 max_length=1024,
                                                 truncation=True)
        target_ids = np.array(target_ids['input_ids'], dtype=np.int32)
    else:
        target_ids = np.array([], dtype=np.int32)
    answers = np.array(answers, dtype=np.int32)
    return source_ids, source_mask, source_ids_cbr, source_mask_cbr, target_ids, answers


def prompt_rel_encode_dataset(dataset, vocab, tokenizer, test=False):
    questions = []
    targets = []
    for item in tqdm(dataset, total=len(dataset)):
        questions.append('question ' + item['question'] +
                         ' Relations in the question are ')
        if not test:
            rels = get_rel_seq(item['program'])
            target = '; '.join(rels)
            targets.append(target)

    input_ids = tokenizer.batch_encode_plus(questions,
                                            padding=True,
                                            max_length=1024,
                                            truncation=True)
    source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    if not test:
        target_ids = tokenizer.batch_encode_plus(targets,
                                                 padding=True,
                                                 max_length=1024,
                                                 truncation=True)
        target_ids = np.array(target_ids['input_ids'], dtype=np.int32)
    else:
        target_ids = np.array([], dtype=np.int32)
    return source_ids, source_mask, target_ids


def prompt_encode_dataset(dataset, vocab, tokenizer, test=False):
    questions = []
    programs = []
    choices = []
    answers = []
    for item in tqdm(dataset):
        question = 'question ' + item['question'] \
                    + ' Program for the question is '
        questions.append(question)
        _ = [vocab['answer_token_to_idx'][w] for w in item['choices']]
        choices.append(_)
        if not test:
            program = item['program']
            program = get_program_seq(program)
            programs.append(program)
            answers.append(vocab['answer_token_to_idx'].get(item['answer']))

    input_ids = tokenizer.batch_encode_plus(questions,
                                            max_length=1024,
                                            padding=True,
                                            truncation=True)
    source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    if not test:
        target_ids = tokenizer.batch_encode_plus(programs,
                                                 max_length=1024,
                                                 padding=True,
                                                 truncation=True)
        target_ids = np.array(target_ids['input_ids'], dtype=np.int32)
    else:
        target_ids = np.array([], dtype=np.int32)
    choices = np.array(choices, dtype=np.int32)
    answers = np.array(answers, dtype=np.int32)
    return source_ids, source_mask, target_ids, choices, answers


def prompt_cbr_encode_dataset(dataset,
                              recall_db,
                              recall_index,
                              vocab,
                              tokenizer,
                              name,
                              k_cases=10,
                              boost=False):
    assert len(dataset) == len(recall_index)
    questions = []
    questions_plus_cases = []
    questions_rels = []
    questions_funcs = []
    programs = []
    rels = []
    funcs = []
    programs_seqs = [get_program_seq(item['program']) for item in recall_db]
    answers = []
    for item, index_list in tqdm(zip(dataset, recall_index),
                                 total=len(dataset)):
        count = 1
        if name != 'test':
            seq = get_program_seq(item['program'])
            cur_rels = get_rel_seq(item['program'])
            cur_funcs = get_func_seq(item['program'])
            if boost and name == 'train':
                p_list = permute_program_seq(seq)
            else:
                p_list = [seq]
            count = len(p_list)
            for p in p_list:
                programs.append(p)
                random.shuffle(cur_rels)
                rels.append('; '.join(cur_rels))
                random.shuffle(cur_funcs)
                funcs.append('; '.join(cur_funcs))
                answers.append(vocab['answer_token_to_idx'].get(item['answer']))
        question = 'Question 0 ' + item['question']
        questions.extend([question + ' Program for Question 0 is'] * count)
        questions_rels.extend([question + ' Relations in the Question 0 are '] *
                              count)
        questions_funcs.extend(
            [question + ' Functions in the Question 0 are '] * count)
        sents = [question]
        for idx, i in enumerate(index_list[:k_cases]):
            question = 'Question %d %s Program for Question %d is %s' % (
                idx + 1, recall_db[i]['question'], idx + 1, programs_seqs[i])
            sents.append(question)
        sents.append('Program for Question 0 is')
        question = ' '.join(sents)
        questions_plus_cases.extend([question] * count)

    input_ids = tokenizer.batch_encode_plus(questions,
                                            padding=True,
                                            max_length=1024,
                                            truncation=True)
    q_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    q_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    input_ids = tokenizer.batch_encode_plus(questions_plus_cases,
                                            padding=True,
                                            max_length=1024,
                                            truncation=True)
    q_cbr_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    q_cbr_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    input_ids = tokenizer.batch_encode_plus(questions_rels,
                                            padding=True,
                                            max_length=1024,
                                            truncation=True)
    q_rel_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    q_rel_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    input_ids = tokenizer.batch_encode_plus(questions_funcs,
                                            padding=True,
                                            max_length=1024,
                                            truncation=True)
    q_func_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    q_func_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    if name != 'test':
        target_ids = tokenizer.batch_encode_plus(programs,
                                                 padding=True,
                                                 max_length=1024,
                                                 truncation=True)
        program_ids = np.array(target_ids['input_ids'], dtype=np.int32)
        target_ids = tokenizer.batch_encode_plus(rels,
                                                 padding=True,
                                                 max_length=1024,
                                                 truncation=True)
        rel_ids = np.array(target_ids['input_ids'], dtype=np.int32)
        target_ids = tokenizer.batch_encode_plus(funcs,
                                                 padding=True,
                                                 max_length=1024,
                                                 truncation=True)
        func_ids = np.array(target_ids['input_ids'], dtype=np.int32)
    else:
        target_ids = np.array([], dtype=np.int32)
        program_ids = target_ids
        rel_ids = target_ids
        func_ids = target_ids
    answers = np.array(answers, dtype=np.int32)
    res = {
        "q_ids": q_ids,
        "q_mask": q_mask,
        "q_rel_ids": q_rel_ids,
        "q_rel_mask": q_rel_mask,
        "q_func_ids": q_func_ids,
        "q_func_mask": q_func_mask,
        "q_cbr_ids": q_cbr_ids,
        "q_cbr_mask": q_cbr_mask,
        "program_ids": program_ids,
        "rel_ids": rel_ids,
        "func_ids": func_ids,
        "answer": answers
    }
    print('Shapes: ')
    for key, val in res.items():
        print(key, val.shape)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--type', type=str, default='default')
    parser.add_argument('--cbr_k', type=int, default=3)
    parser.add_argument('--recall_index_dir')
    parser.add_argument('--boost', action='store_true')
    parser.add_argument('--model_name_or_path', required=True)
    args = parser.parse_args()

    print('Build kb vocabulary')
    vocab = {'answer_token_to_idx': {}}
    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'val.json')))
    test_set = json.load(open(os.path.join(args.input_dir, 'test.json')))
    for question in chain(train_set, val_set, test_set):
        for a in question['choices']:
            if not a in vocab['answer_token_to_idx']:
                vocab['answer_token_to_idx'][a] = len(
                    vocab['answer_token_to_idx'])

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    fn = os.path.join(args.output_dir, 'vocab.json')
    print('Dump vocab to {}'.format(fn))
    with open(fn, 'w') as f:
        json.dump(vocab, f, indent=2)
    for k in vocab:
        print('{}:{}'.format(k, len(vocab[k])))
    tokenizer = BartTokenizerFast.from_pretrained(args.model_name_or_path)
    if 'cbr' in args.type:
        print('Load recall index : ', args.recall_index_dir)
        train_recall = pickle.load(
            open(os.path.join(args.recall_index_dir, 'train.pkl'), 'rb'))
        valid_recall = pickle.load(
            open(os.path.join(args.recall_index_dir, 'valid.pkl'), 'rb'))
        test_recall = pickle.load(
            open(os.path.join(args.recall_index_dir, 'test.pkl'), 'rb'))
    else:
        train_recall, valid_recall, test_recall = None, None, None
    for name, dataset, recall_index in zip(
        ('train', 'val', 'test'), (train_set, val_set, test_set),
        (train_recall, valid_recall, test_recall)):
        print('Encode {} set'.format(name))
        if args.type == 'cbr':
            outputs = cbr_encode_dataset(dataset,
                                         train_set,
                                         recall_index,
                                         vocab,
                                         tokenizer,
                                         name == 'test',
                                         k_cases=args.cbr_k)
        elif args.type == 'prompt_rel':
            outputs = prompt_rel_encode_dataset(dataset, vocab, tokenizer,
                                                name == 'test')
        elif args.type == 'prompt':
            outputs = prompt_encode_dataset(dataset, vocab, tokenizer,
                                            name == 'test')
        elif args.type == 'prompt_cbr':
            outputs = prompt_cbr_encode_dataset(dataset,
                                                train_set,
                                                recall_index,
                                                vocab,
                                                tokenizer,
                                                name,
                                                k_cases=args.cbr_k,
                                                boost=args.boost)
            outputs['origin_info'] = dataset
            with open(os.path.join(args.output_dir, '{}.pt'.format(name)),
                      'wb') as f:
                pickle.dump(outputs, f)
            continue
        elif args.type == 'default':
            outputs = encode_dataset(dataset,
                                     vocab,
                                     tokenizer,
                                     name,
                                     boost=args.boost)

        print('Shapes: ')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)),
                  'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)
            pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
