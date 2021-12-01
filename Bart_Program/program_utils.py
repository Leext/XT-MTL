import re


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


def program2seq(funcs, inputs_list):
    seq = [
        func + '(' + '<c>'.join(inputs) + ')'
        for func, inputs in zip(funcs, inputs_list)
    ]
    seq = '<b>'.join(seq)
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


funcs_with_rel = {
    'FilterStr', 'FilterNum', 'FilterYear', 'FilterDate', 'QFilterStr',
    'QFilterNum', 'QFilterYear', 'QFilterDate', 'Relate', 'SelectBetween',
    'SelectAmong', 'QueryAttr', 'QueryAttrUnderCondition', 'QueryAttrQualifier',
    'QueryRelationQualifier'
}

funcs_with_2rels = {
    'QueryRelationQualifier', 'QueryAttrUnderCondition', 'Relate'
}
funcs_with_3rels = {'QueryAttrQualifier'}

##################### Program Rep Functions ################################


def get_func_rels(item):
    func = item['function']
    inputs = item['inputs']
    if func in funcs_with_rel:
        yield inputs[0]
    if func in funcs_with_2rels:
        yield inputs[1]
    if func in funcs_with_3rels:
        yield inputs[2]


def get_rel_seq(program):
    seq = []
    for item in program:
        seq.extend(get_func_rels(item))
    return seq


def get_rel_rep(program):
    seq = set()
    for item in program:
        seq.update(get_func_rels(item))
    return seq


def get_func_rel_seq(program):
    seq = []
    for item in program:
        seq.append(item['function'])
        seq.extend(get_func_rels(item))
    return seq


def get_func_rel_rep(program):
    seq = set()
    counter = dict()
    for item in program:
        func = item['function']
        if func in counter:
            counter[func] += 1
            func = '%s_%d' % (func, counter[func])
        else:
            counter[func] = 1
        seq.add(func)
        for rel in get_func_rels(item):
            if rel in counter:
                counter[rel] += 1
                rel = '%s_%d' % (rel, counter[rel])
            else:
                counter[rel] = 1
            seq.add(rel)
    return seq


def get_func_seq(program):
    return [item['function'] for item in program]


def get_func_rep(program):
    seq = set()
    counter = dict()
    for item in program:
        func = item['function']
        if func in counter:
            counter[func] += 1
            func = '%s_%d' % (func, counter[func])
        else:
            counter[func] = 1
        seq.add(func)
    return seq


##################### Program Similarity Functions ################################
eps = 1e-6


def set_f1(s1: set, s2: set):
    tp = len(s1 & s2)
    fp = len(s2 - s1)
    fn = len(s1 - s2)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1


def levenshtein3(s, t):
    ''' From Wikipedia article; Iterative with two matrix rows. '''
    if len(s) == 0:
        return len(t)
    elif len(t) == 0:
        return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(t)]


def edit_similarity(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0
    return 1 - levenshtein3(s1, s2) / max(len(s1), len(s2))


rep_fn_map = {
    'func_rel_seq': get_func_rel_seq,
    'rel_seq': get_rel_seq,
    'func_seq': get_func_seq,
    'func_rel': get_func_rel_rep,
    'rel': get_rel_rep,
    'func': get_func_rep
}
simi_fn_map = {'edit': edit_similarity, 'f1': set_f1}

def get_program_labels(functions):
    cur_labels = []
    for f in functions:
        if f in {'Relate'} or f.startswith('Filter'):
            cur_labels.append('multihop')
            break
    for f in functions:
        if f in {'QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate', 'QueryAttrUnderCondition', 'QueryAttrQualifier', 'QueryRelationQualifier'}:
            cur_labels.append('qualifier')
            break
    for f in functions:
        if f in {'SelectBetween','SelectAmong'}:
            cur_labels.append('comparison')
            break
    for f in functions:
        if f in {'And', 'Or'}:
            cur_labels.append('logical')
            break
    for f in functions:
        if f in {'Count'}:
            cur_labels.append('count')
            break
    for f in functions:
        if f in {'VerifyStr','VerifyNum','VerifyYear','VerifyDate'}:
            cur_labels.append('verify')
            break
    return cur_labels