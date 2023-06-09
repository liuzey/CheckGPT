import os
import sys
import numpy as np
import torch
import json
import h5py
import time
import argparse
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('domain', type=str)
parser.add_argument('brief', type=str)
parser.add_argument('task', type=str)
parser.add_argument('--pairs', type=int, default=0)
args = parser.parse_args()

domain = args.domain
brief = args.brief
task = args.task

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large').to(device)
roberta.eval()

with open("../GPABenchmark/{}_TASK{}/gpt.json".format(brief, task, brief), 'r') as f1:
    data1 = json.load(f1)
f1.close()

with open("../GPABenchmark/{}_TASK{}/hum.json".format(brief, task, brief), 'r') as f2:
    data2 = json.load(f2)
f2.close()

print(len(list(data1.keys())))
print(len(list(data2.keys())))
too_long = 0
total_length = args.number if args.number else len(list(data1.keys())) + len(list(data2.keys()))

start = time.time()
data = h5py.File('./embeddings/{}{}_{}.h5'.format(brief, task, total_length), 'w')
data.create_dataset('data', (total_length, 512, 1024), dtype='f4')
data.create_dataset('label', (total_length, 1), dtype='i')


def fetch_representation(text):
    tokens = roberta.encode(text)
    last_layer_features = None

    if len(tokens) <= 512:
        last_layer_features = roberta.extract_features(tokens)
    return last_layer_features


for i, (gpt, hum) in enumerate(zip(data1.values(), data2.values())):
    gpt_features, hum_features = fetch_representation(gpt), fetch_representation(hum)

    if gpt_features is None or hum_features is None:
        too_long += 1
        continue

    gpt_features_ = F.pad(gpt_features.last_hidden_state, (0, 0, 0, 512 - gpt_features.size(1)))
    hum_features_ = F.pad(hum_features.last_hidden_state, (0, 0, 0, 512 - hum_features.size(1)))

    data["data"][2 * i] = gpt_features_.clone().detach().cpu()
    data["label"][2 * i] = torch.zeros(1)

    data["data"][2 * i + 1] = hum_features_.clone().detach().cpu()
    data["label"][2 * i + 1] = torch.ones(1)

    if i % 200 == 0:
        print("{}{} at {}th pair. Time used: {}s. Outliers: {}".format(domain, task, i, time.time()-start, too_long))

    i += 1
data.close()






