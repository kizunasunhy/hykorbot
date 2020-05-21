#To make training data
from itertools import permutations
import json
from collections import OrderedDict
import random

data = open('ChatbotData.csv', 'r', encoding='utf-8')

output = open('train_data_chatbot.txt', 'w', encoding='utf-8')

lines = data.readlines()

data.close()

total_data = OrderedDict()
train_list = []
valid_list = []

cnt = 0
total_len = len(lines)

random.shuffle(lines)

def random_sampling(num_samples, except_sent):
    samples = []
    while(len(samples) < num_samples):
        random_idx = random.randint(0, len(lines)-1)
        random_line = lines[random_idx].replace('\n', '')
        sample = random_line.split(',')[1]
        if sample != except_sent:
            samples.append(sample)
    return samples
    

for i, line in enumerate(lines):
    if i == 0:
        continue
    line = line.replace('\n', '') 
    
    utterances = OrderedDict()
    train = OrderedDict()

    sent1 = line.split(',')[0]
    sent2 = line.split(',')[1]

    candidates = random_sampling(15, sent2)
    candidates.append(sent2)

    utterances['candidates'] = candidates
    utterances['history'] = [sent1]

    train['utterances'] = [utterances]

    if i < len(lines) * 0.95:
        train_list.append(train)
    else:
        valid_list.append(train)

total_data['train'] = train_list

total_data['valid'] = valid_list

output.write(json.dumps(total_data, ensure_ascii=False, indent="\t") )

output.close()
