import os, json, pickle

from .program_utils import get_program_seq

def load_pred_dump(fname):
	questions, preds, golds = [], [], []
	with open(fname, 'r') as f:
		for line in f:
			if line[0] == 'Q':
				questions.append(line.strip()[3:])
			elif line[0] == 'G':
				golds.append(line.strip()[6:])
			elif line[0] == 'P':
				preds.append(line.strip()[6:])
	return questions, golds, preds

def load_recall_index(fname):
	with open(fname, 'rb') as f:
		return pickle.load(f)

def load_json(fname):
	with open(fname, 'r') as f:
		return json.load(f)

def main():
	train_json = load_json('dataset/train.json')
	valid_json = load_json('dataset/val.json')
	valid_recall_index = load_recall_index('Bart_Program/recall_dump/valid.pkl')
	model1_pred_dump = load_pred_dump('Bart_Program/saves/vanilla/epoch_24/incorrect_pred.txt')
	valid_question_index = {item['question']: idx for idx, item in enumerate(valid_json)}
	for q, g, p in zip(*model1_pred_dump):
		idx = valid_question_index[q]
		print('Q: %s' % q)
		print('Gold: %s' % g)
		print('Pred: %s' % p)
		recall_index = valid_recall_index[idx]
		for i in recall_index:
			item = train_json[i]
			print('\tQ: %s' % item['question'])
			print('\tGold: %s ' % get_program_seq(item['program']))
		print()
		input()


if __name__ == '__main__':
	main()