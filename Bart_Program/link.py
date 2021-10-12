import argparse
import json
import os
import pickle
import re

from .preprocess import seq2program, get_program_seq
from Levenshtein import distance
import traceback

def get_all_names(kb):
    pass


kb_json = 'dataset_fix/kb.json'


def edit_similarity(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0
    return 1 - distance(s1, s2) / max(len(s1), len(s2))


def search(query, cands, return_orgin=False):
    if query in cands:
        return query
    max_d, max_cand = 0, None
    for cand in cands:
        d = edit_similarity(query, cand)
        if d > max_d:
            max_d = d
            max_cand = cand
    if max_cand is None and return_orgin:
        return query
    return max_cand


def collect_nums(text):
    words = re.split('[ ?()]', text)
    nums = []
    for w in words:
        w = w.replace(',', '')
        try:
            if '.' in w:
                ww = w.split('.')
                n = str(int(ww[0])) + str(int(ww[1]))
            else:
                n = str(int(w))
            nums.append(n)
        except:
            pass
    return nums


class Linker2:
    def __init__(self, kb_json: str):
        kb = json.load(open(kb_json))
        concepts = kb['concepts']
        entities = kb['entities']
        for con_id, con_info in concepts.items():
            con_info['name'] = ' '.join(con_info['name'].split())
        for ent_id, ent_info in entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())
        self.concept_names = {con['name'] for con in concepts.values()}
        self.entity_names = {ent['name'] for ent in entities.values()}
        attr_names = []
        for ent in entities.values():
            for attr in ent['attributes']:
                if attr['value']['type'] != 'quantity' and type(attr['value']['value']) is str:
                    attr_names.append(attr['value']['value'])
                for item in attr['qualifiers'].values():
                    for sub_item in item:
                        if sub_item['type'] != 'quantity' and type(sub_item['value']) is str:
                            attr_names.append(sub_item['value'])
            for rel in ent['relations']:
                for qual in rel['qualifiers'].values():
                    for item in qual:
                        if item['type'] == 'string':
                            attr_names.append(item['value'])
        self.attr_names = set(attr_names)

    def revise_program(self, qtext, program_seq):
        try:
            program = seq2program(program_seq)
            nums = collect_nums(qtext)
            for func, inputs in zip(*program):
                if func == 'Find':
                    inputs[0] = search(inputs[0], self.entity_names, True)
                elif func == 'FilterConcept':
                    inputs[0] = search(inputs[0], self.concept_names, True)
                elif func in {'FilterStr', 'QFilterStr', 'QueryAttrQualifier'}:
                    inputs[1] = search(inputs[1], self.attr_names, True)
                elif func == 'VerifyStr':
                    inputs[0] = search(inputs[0], self.attr_names, True)
                elif func in {'FilterNum', 'QFilterNum'}:
                    inputs[1] = search(inputs[1], nums, True)
            program = [{
                'function': func,
                'inputs': inputs
            } for func, inputs in zip(*program)]
            program_seq_revised = get_program_seq(program)
            return program_seq_revised
        except:
            traceback.print_exc()
        return program_seq


if __name__ == '__main__':
    pass
