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
parser.add_argument('--number', type=int, default=0)
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


for gpt, hum in zip(data1.values(), data2.values()):
    gpt_features, hum_features = fetch_representation(gpt), fetch_representation(hum)

    if gpt_features is None or hum_features is None:
        too_long += 1
        continue

        length = int(last_layer_features.shape[1])

        padding = torch.zeros(512, 1024, dtype=torch.float64)
        padding[:length] = last_layer_features

        data["data"][i] = padding.clone().detach().cpu()
        data["label"][i] = torch.zeros(1)

    if i % 500 == 0:
        print("{}{} at {} at data1. Time used: {}s. Outliers: {}".format(domain, task, i, time.time()-start, too_long))

    i += 1


for item in data2.values():
    tokens = roberta.encode(item)
    # print(tokens)
    if len(tokens) > 512:
        too_long += 1
        continue

    if i >= skip:
        last_layer_features = roberta.extract_features(tokens)
        # print(last_layer_features.shape)
        length = int(last_layer_features.shape[1])

        padding = torch.zeros(512, 1024, dtype=torch.float64)
        padding[:length] = last_layer_features

        data["data"][i] = padding.clone().detach().cpu()
        data["label"][i] = torch.ones(1)

    if i % 500 == 0:
        print("{}{} at {} at data2. Time used: {}s. Outliers: {}".format(domain, task, i, time.time() - start, too_long))

    i += 1

data.close()






