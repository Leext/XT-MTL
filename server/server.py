import sys
sys.path.append('/home/LAB/lixt/KBQA/')
import json
import argparse
import os
from typing import List, Tuple

from flask import Flask, jsonify, request, render_template
from tqdm import tqdm
import traceback
import random
import torch
import transformers
from transformers import BartForConditionalGeneration, BartTokenizerFast
from Bart_Program.model import BartCBR
from Bart_Program.executor_rule import RuleExecutor
from Bart_Program.program_utils import seq2program, get_program_seq, program2seq, get_program_labels

from MyQA.train.Server import Server as BoostServer

class ModelManager:
    def __init__(self,
                 qa_model_path,
                 recall_model_path,
                 base_model_path='') -> None:
        print('Loading QA model %s' % qa_model_path)
        self.qa_model = BartForConditionalGeneration.from_pretrained(
            qa_model_path)
        self.qa_model.eval()
        print('Loading Recall model %s' % recall_model_path)
        self.recall_model = BartCBR(recall_model_path)
        self.recall_model.eval()
        if os.path.isdir(base_model_path):
            print('Loading base model %s' % base_model_path)
            self.base_model = BartForConditionalGeneration.from_pretrained(
                base_model_path)
            self.base_model.eval()
        else:
            self.base_model = None

    def generate(self, input_ids, use_base=False):
        with torch.no_grad():
            if use_base and self.base_model is not None:
                model = self.base_model
                print('Use base')
            else:
                model = self.qa_model
                print('Use exp')
            outputs = model.generate(input_ids=input_ids, max_length=500)

        return outputs

    def sent_rep(self, input_ids):
        with torch.no_grad():
            return self.recall_model.get_sent_rep(input_ids['input_ids'],
                                                  input_ids['attention_mask'])


class KBManager:
    def __init__(self, kb_path) -> None:
        self.executor = RuleExecutor(None, kb_path)

    def exec_program(self, program: str):
        func_list, inputs_list = seq2program(program)
        print('Before revise:', program2seq(func_list, inputs_list))
        # func_list, inputs_list = self.executor.revise_program(
        #     program, func_list, inputs_list)
        answer = self.executor.forward(func_list,
                                       inputs_list,
                                       ignore_error=True)
        if answer is None:
            answer = '知识图谱中未找到相关信息'
        print('After revise :', program2seq(func_list, inputs_list))
        return answer, (func_list, inputs_list)


class IndexManager:
    def __init__(self, questions, vectors) -> None:
        self.questions = questions
        self.vectors = vectors

    def get_topk(self, vector, k: int):
        _, index = torch.topk(torch.matmul(vector, self.vectors.T), k, dim=1)
        # print(index.shape)
        q = [self.questions[i] for i in index[0]]
        return q
    
    def insert(self, q, v):
        self.questions.append(q)
        self.vectors = torch.cat([self.vectors, v])


class QADataManager:
    def __init__(self, raw_json_path) -> None:
        with open(raw_json_path, 'r') as f:
            data = json.load(f)
        self.q2program = {
            item['question']: get_program_seq(item['program'])
            for item in data
        }

    def find(self, text):
        return self.q2program.get(text)

    def random_example_list(self, examples=None):
        """
        [{
            "q": "",
            "labels": [""]
        }]
        """
        if examples is None:
            examples = random.sample(self.q2program.keys(), 6)
        labels = [
            get_program_labels(seq2program(self.q2program[q])[0])
            for q in examples
        ]
        res = [{'question': q, 'labels': l} for q, l in zip(examples, labels)]
        return res


graph_tmpl = """digraph G {
node [shape=none margin=0 fontname="Courier New"]
edge []
rankdir=LR; 
%s
%s
}"""
node_tmpl = """{name}[label=<
    <table border="0" cellborder="1" cellspacing="0" cellpadding="6" bgcolor="LightGreen">
    <tr>
        <td><B>{func_name}</B></td>
    </tr>
    {attr}
    </table>
>];"""


def gen_graph(func_list, inputs_list):
    nodes = []
    edges = []
    try:
        for idx, (func, inputs) in enumerate(zip(func_list, inputs_list)):
            if inputs:
                for i in range(len(inputs)):
                    if inputs[i] == '<':
                        inputs[i] = '&lt;'
                    elif inputs[i] == '>':
                        inputs[i] = '&gt;'
                func_attr = '<tr><td>%s</td></tr>' % ('<BR/>'.join(inputs))
            else:
                func_attr = ''
            node = node_tmpl.format(name='node%d' % idx,
                                    func_name=func,
                                    attr=func_attr)
            nodes.append(node)
        branch_stack = []
        for idx, func in enumerate(func_list):
            if func in {'FindAll', 'Find'}:
                branch_stack.append(idx - 1)
            elif func in {
                    'And', 'Or', 'SelectBetween', 'QueryRelation',
                    'QueryRelationQualifier'
            }:
                edges.append('%s -> %s;' %
                             ('node%d' % branch_stack[-1], 'node%d' % idx))
                edges.append('%s -> %s;' % ('node%d' % (idx - 1), 'node%d' %
                                            (idx)))
                branch_stack.pop()
            else:
                edges.append('%s -> %s;' % ('node%d' % (idx - 1), 'node%d' %
                                            (idx)))
    except Exception as e:
        traceback.print_exc()
    graph = graph_tmpl % ('\n'.join(nodes), '\n'.join(edges))
    return graph


class ServerBackend:
    def __init__(self, qa_model_path: str, recall_model_path: str,
                 index_path: str, kb_path: str, raw_data_path: str,
                 base_model_path: str) -> None:
        self.model_mgr = ModelManager(qa_model_path, recall_model_path,
                                      base_model_path)
        self.tokenizer = BartTokenizerFast.from_pretrained(qa_model_path)
        self.kb_mgr = KBManager(kb_path)
        self.data_mgr = QADataManager(raw_data_path)
        questions, rep = self._build_index(index_path)
        self.index_mgr = IndexManager(questions, rep)
        print('Loading boost')
        self.boost_server = self

    def _build_index(self, index_path):
        print('Loading Index')
        questions, rep = torch.load(index_path)
        return questions, rep

    def insert_index(self, question: str, program: str):
        if question in self.data_mgr.q2program:
            return
        self.data_mgr.q2program[question] = program
        input_ids = self.tokenizer.batch_encode_plus([question],
                                                    padding=True,
                                                    max_length=1024,
                                                    truncation=True,
                                                    return_tensors='pt')
        rep = self.model_mgr.recall_model.get_sent_rep(input_ids['input_ids'],
                                        input_ids['attention_mask'])
        self.index_mgr.insert(question, rep)
        print('new size %d, %d' % (len(self.data_mgr.q2program), len(self.index_mgr.vectors)))


    def answer(self, question: str, use_base=False):
        # p = self.data_mgr.find(question)
        p = self.gen_program(question, use_base)
        ans, (func_list, inputs_list) = self.kb_mgr.exec_program(p)
        return {
            'answer': ans,
            'program': program2seq(func_list, inputs_list),
            'graph': gen_graph(func_list, inputs_list)
        }

    def exec_program(self, program: str):
        # p = self.data_mgr.find(question)
        ans, (func_list, inputs_list) = self.kb_mgr.exec_program(program)
        return {
            'answer': ans,
            'program': program2seq(func_list, inputs_list),
            'graph': gen_graph(func_list, inputs_list)
        }

    def _prepare_qa_input(self,
                          question: str,
                          cases: List[Tuple[str, str]],
                          use_base=False):
        if not use_base:
            question = 'Question 0 %s Program for Question 0 is' % question
            sents = [question]
            for idx, (q, p) in enumerate(cases):
                question = 'Question %d %s Program for Question %d is %s' % (
                    idx + 1, q, idx + 1, p)
                sents.append(question)
            sents.append('Program for Question 0 is')
            input_str = ' '.join(sents)
        else:
            input_str = question
        input_ids = self.tokenizer.batch_encode_plus([input_str],
                                                     padding=True,
                                                     max_length=1024,
                                                     truncation=True,
                                                     return_tensors='pt')
        return input_ids['input_ids']

    def _prepare_recall_input(self, question: str):
        if type(question) is str:
            question = [question]
        input_ids = self.tokenizer.batch_encode_plus(question,
                                                     padding=True,
                                                     max_length=1024,
                                                     truncation=True,
                                                     return_tensors='pt')
        return input_ids

    def _decode(self, output_ids):
        res = [
            self.tokenizer.decode(output,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)
            for output in output_ids
        ]
        return res

    def gen_program(self, question: str, use_base=False):
        if not use_base:
            input_ids = self._prepare_recall_input(question)
            q_rep = self.model_mgr.sent_rep(input_ids)
            q_cases = self.index_mgr.get_topk(q_rep, 5)
            cases = [(q, self.data_mgr.q2program.get(q)) for q in q_cases]
        else:
            cases = []
        input_ids = self._prepare_qa_input(question, cases, use_base)
        output_ids = self.model_mgr.generate(input_ids, use_base)
        program = self._decode(output_ids)[0]
        return program


def build_index(args):
    device = torch.device(args.device)
    with open('./dataset/train.json', 'r') as f:
        data = json.load(f)
    questions = [item['question'] for item in data]
    recall_model = BartCBR(args.recall_model_path)
    recall_model.eval()
    tokenizer = BartTokenizerFast.from_pretrained(args.qa_model_path)
    recall_model.to(device)
    with torch.no_grad():
        batch_size = 64
        reps = []
        for i in tqdm(range(0, len(questions), batch_size)):
            input_ids = tokenizer.batch_encode_plus(questions[i:i +
                                                              batch_size],
                                                    padding=True,
                                                    max_length=1024,
                                                    truncation=True,
                                                    return_tensors='pt')
            input_ids.to(device)
            rep = recall_model.get_sent_rep(input_ids['input_ids'],
                                            input_ids['attention_mask'])
            reps.append(rep.cpu())
        rep = torch.cat(reps)
        torch.save((questions, rep), args.index_path)


def run_flask(args):
    kb_path = './dataset/kb.json'
    raw_data_path = './dataset/train.json'
    server = ServerBackend(args.qa_model_path, args.recall_model_path,
                           args.index_path, kb_path, raw_data_path,
                           args.base_model_path)
    app = Flask(__name__,
                static_folder="./dist/assets",
                template_folder="./dist")

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/program_dag')
    def get_dag():
        seq = request.args['program']
        res = gen_graph(*seq2program(seq))
        return res

    @app.route('/examples')
    def get_examples():
        model_name = request.args['model']
        if model_name == '数据增强方法':
            jsonify(server.boost_server.random_example_list())
        else:
            return jsonify(server.data_mgr.random_example_list())

    @app.route('/init_examples')
    def get_init_examples():
        model_name = request.args['model']
        init_q = [
            "Was Shelley Duvalls, a cast member of The Shining, playing Wendy Torrance, born before 1984?",
            "Which film has the shorter runtime, The Avengers (in the spy film genre) or Quadrophenia (released in Denmark)?",
            "How many London areas are in the United Kingdom or correspond to local dialing code 020?",
            "How many miniseries that were published before 1984 mainly depicted World War II?",
            "Does Palo Alto, birthplace of Hewlett-Packard, or Charleston, capital of West Virginia, have less elevation above sea level?",
            "Among the feature films that is distributed at United Kingdom,which one has the smallest duration ?"
        ]
        if model_name == '数据增强方法':
            jsonify(server.boost_server.random_example_list())
        return jsonify(server.data_mgr.random_example_list(init_q))

    @app.route('/qa')
    def qa():
        q = request.args['question']
        model_name = request.args['model']
        method_name = request.args['method']
        use_base = method_name == '基线'
        print('Question:', q)
        if model_name == '数据增强方法':
            res = server.boost_server.answer(q)
        else:
            res = server.answer(q, use_base)
        return res

    @app.route('/exec')
    def exec_program():
        p = request.args['program']
        res = server.exec_program(p)
        return res

    @app.route('/save')
    def save_case():
        p = request.args['program']
        q = request.args['question']
        server.insert_index(q, p) 
        return "ok"

    app.run('0.0.0.0', args.port, debug=True)


# best_qa_model = './Bart_Program/saves/mtl_rel_cbr_infill+2+_kbp2_0.5_w0.1_real_smp_1.0_revise/epoch_52'
best_qa_model = './Bart_Program/saves/mtl_qcbr_rel_0.1_kbp_0.5_real_smp_1.0_revise/epoch_25'
base_qa_model = './Bart_Program/saves/vanilla_0.1/epoch_25'
best_recall_model = './Bart_Program/saves/recall_rule_rel/epoch_3'
index_path = './server/index/index.pt'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=9997, type=int)
    parser.add_argument('--qa_model_path', default=best_qa_model)
    parser.add_argument('--base_model_path', default=base_qa_model)
    parser.add_argument('--recall_model_path', default=best_recall_model)
    parser.add_argument('--index_path', default=index_path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--build_index', action='store_true')
    args = parser.parse_args()

    if args.build_index:
        build_index(args)
    run_flask(args)
