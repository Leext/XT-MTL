"""
We need the last function to help extract the final answer of SPARQL, used in check_sparql
"""

import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm
import re

from utils.misc import init_vocab
from transformers import *


def get_program_seq(program):
    seq = []
    for item in program:
        func = item['function']
        inputs = item['inputs']
        seq.append(func + '(' + '<c>'.join(inputs) + ')')
    seq = '<b>'.join(seq)
    # print(program)
    # print(seq)
    return seq


pattern = re.compile(r'(.*?)\((.*?)\)$')


def seq2program(seq: str):
    chunks = seq.split('<b>')
    func_list = []
    inputs_list = []
    for chunk in chunks:
        # print(chunk)
        res = pattern.findall(chunk)
        # print(res)
        if len(res) == 0:
            continue
        res = res[0]
        func, inputs = res[0], res[1]
        if inputs == '':
            inputs = []
        else:
            inputs = inputs.split('<c>')

        func_list.append(func)
        inputs_list.append(inputs)
    return func_list, inputs_list


def encode_dataset(dataset, vocab, tokenizer, test=False):
    questions = []
    programs = []
    for item in tqdm(dataset):
        question = item['question']
        questions.append(question)
        if not test:
            program = item['program']
            program = get_program_seq(program)
            programs.append(program)
    sequences = questions + programs
    encoded_inputs = tokenizer(sequences, padding=True)
    print(encoded_inputs.keys())
    print(encoded_inputs['input_ids'][0])
    print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    print(tokenizer.decode(encoded_inputs['input_ids'][-1]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    assert max_seq_length == len(encoded_inputs['input_ids'][-1])
    print(max_seq_length)
    questions = []
    programs = []
    choices = []
    answers = []
    for item in tqdm(dataset):
        question = item['question']
        questions.append(question)
        _ = [vocab['answer_token_to_idx'][w] for w in item['choices']]
        choices.append(_)
        if not test:
            program = item['program']
            program = get_program_seq(program)
            programs.append(program)
            answers.append(vocab['answer_token_to_idx'].get(item['answer']))

    input_ids = tokenizer.batch_encode_plus(questions,
                                            max_length=max_seq_length,
                                            pad_to_max_length=True,
                                            truncation=True)
    source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)
    if not test:
        target_ids = tokenizer.batch_encode_plus(programs,
                                                 max_length=max_seq_length,
                                                 pad_to_max_length=True,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--cbr', action='store_true')
    parser.add_argument('--cbr_k', type=int, default=3)
    parser.add_argument('--recall_index_dir')
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
    if args.cbr:
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
        if not args.cbr:
            outputs = encode_dataset(dataset, vocab, tokenizer, name == 'test')
        else:
            outputs = cbr_encode_dataset(dataset,
                                         train_set,
                                         recall_index,
                                         vocab,
                                         tokenizer,
                                         name == 'test',
                                         k_cases=args.cbr_k)
        print('Shapes: ')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)),
                  'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)
            pickle.dump(dataset, f)


if __name__ == '__main__':
    main()