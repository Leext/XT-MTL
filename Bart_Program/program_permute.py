from queue import SimpleQueue
from copy import deepcopy
from typing import List
from itertools import permutations, chain
from utils.value_class import ValueClass, comp
from Bart_Program.program_utils import get_program_seq, seq2program
'''
    program等价变换
        逆着DAG图做遍历
            1. 对两个函数之间的Filter类函数做排列
            2. 交换双参数函数的参数(And, Or, SelectBetween)
'''


def build_dep(func_list):
    dependency = []
    branch_stack = []
    for i, p in enumerate(func_list):
        if p in {'FindAll', 'Find'}:
            dep = []
            branch_stack.append(i - 1)
        elif p in {
                'And', 'Or', 'SelectBetween', 'QueryRelation',
                'QueryRelationQualifier'
        }:
            dep = [branch_stack[-1], i - 1]
            branch_stack.pop()
        else:
            dep = [i - 1]
        dependency.append(dep)
    return dependency


flex_funcs = {
    'FilterConcept', 'FilterStr', 'FilterNum', 'FilterYear', 'FilterDate',
    'QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate'
}


def get_program_permutation(func_list, inputs_list, dep_list) -> List[str]:
    res = []

    def dfs_swap(idx):
        for i in range(idx, len(func_list)):
            func = func_list[i]
            if func in {'And', 'Or', 'SelectBetween'}:
                dfs_swap(i + 1)
                dep_list[i][0], dep_list[i][1] = dep_list[i][1], dep_list[i][0]
                dfs_swap(i + 1)
                dep_list[i][0], dep_list[i][1] = dep_list[i][1], dep_list[i][0]
                break
        else:
            res.append(dfs_gen())

    def dfs_gen():
        seq = []

        def dfs(idx):
            for i in dep_list[idx]:
                dfs(i)
            seq.append(idx)

        dfs(len(func_list) - 1)
        return seq

    def gen_seq(indice):
        seq = [
            func_list[i] + '(' + '<c>'.join(inputs_list[i]) + ')'
            for i in indice
        ]
        seq = '<b>'.join(seq)
        return seq

    def permute_flex_funcs(indice: List[int]) -> List[List[int]]:
        # split
        seq_list = []
        is_flex = []
        cur = []
        flex = False
        for i in indice:
            func = func_list[i]
            if func in flex_funcs:
                if flex:
                    cur.append(i)
                else:
                    seq_list.append(cur)
                    cur = [i]
                    is_flex.append(flex)
                flex = True
            else:
                if flex:
                    seq_list.append(cur)
                    cur = [i]
                    is_flex.append(flex)
                else:
                    cur.append(i)
                flex = False
        seq_list.append(cur)
        is_flex.append(flex)
        res = []
        cur = []

        def dfs_gen(i):
            if i == len(seq_list):
                res.append(cur.copy())
                return
            if is_flex[i]:
                for s in permutations(seq_list[i]):
                    cur.append(s)
                    dfs_gen(i + 1)
                    cur.pop()
            else:
                cur.append(seq_list[i])
                dfs_gen(i + 1)
                cur.pop()

        dfs_gen(0)
        res = [list(chain(*seq)) for seq in res]
        return res

    dfs_swap(0)
    all_res = []
    for seq in res:
        all_res += permute_flex_funcs(seq)
    res = [gen_seq(seq) for seq in all_res]
    return res


def permute_program_seq(program: str):
    func_list, inputs_list = seq2program(program)
    dep = build_dep(func_list)
    perm = get_program_permutation(func_list, inputs_list, dep)
    return perm
