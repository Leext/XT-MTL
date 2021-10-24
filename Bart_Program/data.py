import json
import logging
import pickle
from threading import settrace

import numpy as np
import torch
from torch._C import _debug_set_autodiff_subgraph_inlining
from utils.misc import invert_dict


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab


def sample(batch_data, ratio: float):
    if ratio >= 1.0:
        return batch_data
    datasize = len(batch_data[0])
    sample_size = int(ratio * datasize)
    indice = np.random.choice(datasize, sample_size, replace=False)
    sample_data = []
    for item in batch_data:
        if type(item) is list:
            data = [item[i] for i in indice]
        else:
            data = item[indice] if len(item) == datasize else item
        sample_data.append(data)
    return sample_data

def sample_dict(batch_data:dict, ratio:float):
    if ratio >= 1.0:
        return batch_data
    datasize = len(batch_data['origin_info'])
    sample_size = int(ratio * datasize)
    indice = np.random.choice(datasize, sample_size, replace=False)
    sample_data = dict()
    for key, item in batch_data.items():
        if type(item) is list:
            data = [item[i] for i in indice]
        else:
            data = item[indice] if len(item) == datasize else item
        sample_data[key] = data
    return sample_data


def collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    choices = torch.stack(batch[2])
    if batch[4][0] is None:
        target_ids, answer = None, None
    else:
        target_ids = torch.stack(batch[3])
        answer = torch.cat(batch[4])
    origin_info = batch[5]
    return source_ids, source_mask, choices, target_ids, answer, origin_info


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids, self.choices, self.answers, self.origin_info = inputs
        self.is_test = len(self.answers) == 0

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        choices = torch.LongTensor(self.choices[index])
        if self.is_test:
            target_ids = None
            answer = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
            answer = torch.LongTensor([self.answers[index]])
        origin_info = self.origin_info[index]
        return source_ids, source_mask, choices, target_ids, answer, origin_info

    def __len__(self):
        return len(self.source_ids)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 vocab_json,
                 question_pt,
                 batch_size,
                 training=False,
                 ratio=1.0):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))

        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(5):
                inputs.append(pickle.load(f))
            inputs.append(pickle.load(f))  # origin questions
        if training:
            logging.info('sample on %s' % question_pt)
            logging.info('sample before %d' % (len(inputs[0])))
            inputs = sample(inputs, ratio)
            logging.info('sample after %d' % (len(inputs[0])))
        dataset = Dataset(inputs)
        # np.shuffle(dataset)
        # dataset = dataset[:(int)(len(dataset) / 10)]
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate,
        )
        self.vocab = vocab


def collate_pair(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    pos_ids = torch.cat(batch[2])
    pos_mask = torch.cat(batch[3])
    origin_info = list(batch[4])
    pos_origin_info = []
    for info in batch[5]:
        pos_origin_info.extend(info)
    return source_ids, source_mask, pos_ids, pos_mask, origin_info, pos_origin_info


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, recall_index, k):
        self.source_ids, self.source_mask, self.target_ids, self.choices, self.answers, self.origin_info = inputs
        self.k = k
        self.recall_index = recall_index
        self.is_test = len(self.answers) == 0

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        choice = np.random.choice(self.recall_index[index],
                                  self.k,
                                  replace=False)
        pos_ids = torch.LongTensor(self.source_ids[choice])
        pos_mask = torch.LongTensor(self.source_mask[choice])
        origin_info = self.origin_info[index]
        pos_origin_info = [self.origin_info[i] for i in choice]
        return source_ids, source_mask, pos_ids, pos_mask, origin_info, pos_origin_info

    def __len__(self):
        return len(self.source_ids)


class PairDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 vocab_json,
                 question_pt,
                 recall_index,
                 batch_size,
                 k,
                 training=False):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))

        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(5):
                inputs.append(pickle.load(f))
            inputs.append(pickle.load(f))  # origin questions
        with open(recall_index, 'rb') as f:
            recall_index = pickle.load(f)
        dataset = PairDataset(inputs, recall_index, k)
        # np.shuffle(dataset)
        # dataset = dataset[:(int)(len(dataset) / 10)]
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate_pair,
        )
        self.vocab = vocab


def collate_cbr(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    source_ids_cbr = torch.stack(batch[2])
    source_mask_cbr = torch.stack(batch[3])
    if batch[-2][0] is None:
        target_ids, answer = None, None
    else:
        target_ids = torch.stack(batch[4])
        answer = torch.cat(batch[5])
    origin_info = batch[-1]
    return source_ids, source_mask, source_ids_cbr, source_mask_cbr, target_ids, answer, origin_info


class CBRDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.source_ids_cbr, self.source_mask_cbr, self.target_ids, self.answers, self.origin_info = inputs
        self.is_test = len(self.answers) == 0

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        source_ids_cbr = torch.LongTensor(self.source_ids_cbr[index])
        source_mask_cbr = torch.LongTensor(self.source_mask_cbr[index])
        if self.is_test:
            target_ids = None
            answer = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
            answer = torch.LongTensor([self.answers[index]])
        origin_info = self.origin_info[index]
        return source_ids, source_mask, source_ids_cbr, source_mask_cbr, target_ids, answer, origin_info

    def __len__(self):
        return len(self.source_ids)


class CBRDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 vocab_json,
                 question_pt,
                 batch_size,
                 training=False,
                 ratio=1.0):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))

        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(6):
                inputs.append(pickle.load(f))
            inputs.append(pickle.load(f))  # origin questions
        if training:
            logging.info('sample on %s' % question_pt)
            logging.info('sample before %d' % (len(inputs[0])))
            inputs = sample(inputs, ratio)
            logging.info('sample after %d' % (len(inputs[0])))
        dataset = CBRDataset(inputs)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate_cbr,
        )
        self.vocab = vocab


def prompt_rel_collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    if batch[2][0] is None:
        target_ids = None
    else:
        target_ids = torch.stack(batch[2])
    origin_info = batch[-1]
    return source_ids, source_mask, target_ids, origin_info


class PromptRelDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids, self.origin_info = inputs
        self.is_test = len(self.target_ids) == 0

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        if self.is_test:
            target_ids = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])

        origin_info = self.origin_info[index]
        return source_ids, source_mask, target_ids, origin_info

    def __len__(self):
        return len(self.source_ids)


class PromptRelDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 vocab_json,
                 question_pt,
                 batch_size,
                 training=False,
                 ratio=1.0):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))

        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(3):
                inputs.append(pickle.load(f))
            inputs.append(pickle.load(f))  # origin questions
        if training:
            print('sample on %s' % question_pt)
            print('sample before %d' % (len(inputs[0])))
            inputs = sample(inputs, ratio)
            print('sample after %d' % (len(inputs[0])))
        dataset = PromptRelDataset(inputs)
        # np.shuffle(dataset)
        # dataset = dataset[:(int)(len(dataset) / 10)]
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=prompt_rel_collate,
        )
        self.vocab = vocab


def collate_prompt(batch):
    batch = list(zip(*batch))
    q_ids = torch.stack(batch[0])
    q_mask = torch.stack(batch[1])
    q_rel_ids = torch.stack(batch[2])
    q_rel_mask = torch.stack(batch[3])
    q_func_ids = torch.stack(batch[4])
    q_func_mask = torch.stack(batch[5])
    q_cbr_ids = torch.stack(batch[6])
    q_cbr_mask = torch.stack(batch[7])
    if batch[-2][0] is None:
        program_ids = None
        rel_ids = None
        func_ids = None
        answer = None
    else:
        program_ids = torch.stack(batch[8])
        rel_ids = torch.stack(batch[9])
        func_ids = torch.stack(batch[10])
        answer = torch.cat(batch[5])
    origin_info = batch[-1]
    return {
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
        "answer": answer,
        "origin_info": origin_info
    }


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        """
            inputs: {
                "q_ids/mask":,
                "q_rel_ids/mask":,
                "q_func_ids/mask",
                "q_cbr_ids/mask",
                "rel_ids":,
                "func_ids":,
                "program_ids":,
                "answer":,
                "origin_info":,
            }
        """
        for key, value in inputs.items():
            setattr(self, key, value)
        self.is_test = len(self.answer) == 0

    def __getitem__(self, index):
        q_ids = torch.LongTensor(self.q_ids[index])
        q_mask = torch.LongTensor(self.q_mask[index])
        q_rel_ids = torch.LongTensor(self.q_rel_ids[index])
        q_rel_mask = torch.LongTensor(self.q_rel_mask[index])
        q_func_ids = torch.LongTensor(self.q_func_ids[index])
        q_func_mask = torch.LongTensor(self.q_func_mask[index])
        q_cbr_ids = torch.LongTensor(self.q_cbr_ids[index])
        q_cbr_mask = torch.LongTensor(self.q_cbr_mask[index])
        if self.is_test:
            program_ids = None
            rel_ids = None
            func_ids = None
            answer = None
        else:
            program_ids = torch.LongTensor(self.program_ids[index])
            rel_ids = torch.LongTensor(self.rel_ids[index])
            func_ids = torch.LongTensor(self.func_ids[index])
            answer = torch.LongTensor([self.answer[index]])
        origin_info = self.origin_info[index]
        return (q_ids, q_mask, q_rel_ids, q_rel_mask, q_func_ids, q_func_mask,
                q_cbr_ids, q_cbr_mask, program_ids, rel_ids, func_ids, answer,
                origin_info)

    def __len__(self):
        return len(self.origin_info)


class PromptDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 vocab_json,
                 question_pt,
                 batch_size,
                 training=False,
                 ratio=1.0):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))

        with open(question_pt, 'rb') as f:
            inputs = pickle.load(f)
        if training:
            logging.info('sample on %s' % question_pt)
            logging.info('sample before %d' % (len(inputs['origin_info'])))
            inputs = sample_dict(inputs, ratio)
            logging.info('sample after %d' % (len(inputs['origin_info'])))
        dataset = PromptDataset(inputs)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate_prompt,
        )
        self.vocab = vocab