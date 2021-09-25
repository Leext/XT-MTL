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


def get_func_seq(program):
    seq = []
    for item in program:
        func = item['function']
        seq.append(func)
    return seq


funcs_with_rel = {
    'FilterStr', 'FilterNum', 'FilterYear', 'FilterDate', 'QFilterStr',
    'QFilterNum', 'QFilterYear', 'QFilterDate', 'Relate', 'SelectBetween',
    'SelectAmong', 'QueryAttr', 'QueryAttrUnderCondition',
    'QueryAttrQualifier', 'QueryRelationQualifier'
}


def get_func_rel_seq(program):
    seq = []
    for item in program:
        func = item['function']
        inputs = item['inputs']
        seq.append(func)
        if func in funcs_with_rel:
            seq.append(inputs[0])
    return seq


def get_rel_seq(program):
    seq = []
    for item in program:
        func = item['function']
        inputs = item['inputs']
        if func in funcs_with_rel:
            seq.append(inputs[0])
    return seq


pattern = re.compile(r'(.*?)\((.*?)\)')


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
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
        os.mkdir(args.output_dir)
    fn = os.path.join(args.output_dir, 'vocab.json')
    print('Dump vocab to {}'.format(fn))
    with open(fn, 'w') as f:
        json.dump(vocab, f, indent=2)
    for k in vocab:
        print('{}:{}'.format(k, len(vocab[k])))
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    for name, dataset in zip(('train', 'val', 'test'),
                             (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(dataset, vocab, tokenizer, name == 'test')
        assert len(outputs) == 5
        print(
            'shape of input_ids of questions, attention_mask of questions, input_ids of sparqls, choices and answers:'
        )
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)),
                  'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)
            pickle.dump(dataset, f)


if __name__ == '__main__':
    main()