import json
import pickle

import numpy as np
import torch
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
    def __init__(self, vocab_json, question_pt, batch_size, training=False, ratio=1.0):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))

        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(5):
                inputs.append(pickle.load(f))
            inputs.append(pickle.load(f))  # origin questions
        if training:
            inputs = sample(inputs, ratio)
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
    def __init__(self, vocab_json, question_pt, batch_size, training=False, ratio=1.0):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))

        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(6):
                inputs.append(pickle.load(f))
            inputs.append(pickle.load(f))  # origin questions
        dataset = CBRDataset(inputs)
        if training:
            inputs = sample(inputs, ratio)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate_cbr,
        )
        self.vocab = vocab
