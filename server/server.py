import json

from flask import Flask, jsonify, request, render_template
import traceback
import random
import torch
import transformers
from transformers import BartForConditionalGeneration, BartTokenizerFast
from Bart_Program.model import BartCBR
from Bart_Program.executor_rule import RuleExecutor
from Bart_Program.program_utils import seq2program, get_program_seq, program2seq


class ModelManager:
    def __init__(self, qa_model_path, recall_model_path) -> None:
        return
        self.qa_model = BartForConditionalGeneration.from_pretrained(
            qa_model_path)
        self.recall_model = BartCBR(recall_model_path)


class KBManager:
    def __init__(self, kb_path) -> None:
        self.executor = RuleExecutor(None, kb_path)

    def exec_program(self, program: str):
        func_list, inputs_list = seq2program(program)
        func_list, inputs_list = self.executor.revise_program(
            program, func_list, inputs_list)
        answer = self.executor.forward(func_list,
                                       inputs_list,
                                       ignore_error=True)
        return answer, (func_list, inputs_list)


class IndexManager:
    pass


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

    def random_example_list(self):
        examples = random.sample(self.q2program.keys(), 10)
        return examples


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
    def __init__(
        self,
        qa_model_path: str,
        recall_model_path: str,
        cbr_data_path: str,
        kb_path: str,
        raw_data_path: str,
    ) -> None:
        self.model_mgr = ModelManager(qa_model_path, recall_model_path)
        self.kb_mgr = KBManager(kb_path)
        self.data_manager = QADataManager(raw_data_path)
        self.index_mgr = IndexManager()

    def _build_index(self):
        pass

    def insert_index(self, question: str, program: str):
        pass

    def answer(self, question: str):
        p = self.data_manager.find(question)
        ans, (func_list, inputs_list) = self.kb_mgr.exec_program(p)
        return {
            'answer': ans,
            'program': program2seq(func_list, inputs_list),
            'graph': gen_graph(func_list, inputs_list)
        }


def run_flask():
    kb_path = './dataset/kb.json'
    raw_data_path = './dataset/train.json'
    server = ServerBackend("", "", "", kb_path, raw_data_path)
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
        return jsonify(server.data_manager.random_example_list())

    @app.route('/qa')
    def qa():
        q = request.args['question']
        res = server.answer(q)
        return res

    app.run('0.0.0.0', 9997, debug=True)


if __name__ == '__main__':
    run_flask()