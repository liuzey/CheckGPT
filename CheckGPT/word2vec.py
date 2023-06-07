import os
import sys
import numpy as np
import torch
import json
import h5py
import time
import argparse


parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('domain', type=str)
parser.add_argument('brief', type=str)
parser.add_argument('task', type=str)
args = parser.parse_args()

domain = args.domain
brief = args.brief
task = args.task
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large').cuda()
roberta.eval()

with open("./{}{}/{}_gpt.json".format(brief, task, brief), 'r') as f1:
    data1 = json.load(f1)

with open("./{}{}/{}_ground.json".format(brief, task, brief), 'r') as f2:
    data2 = json.load(f2)

print(len(list(data1.keys())))
print(len(list(data2.keys())))

total_length = len(list(data1.keys())) + len(list(data2.keys()))

start = time.time()
data = h5py.File('./embeddings/{}{}/{}{}.h5'.format(brief, task, domain, task), 'w')
data.create_dataset('data', (total_length, 512, 1024), dtype='f2')
data.create_dataset('label', (total_length, 1), dtype='i')

i = 0
too_long = 0
skip = 0

for item in data1.values():
    tokens = roberta.encode(item)

    if len(tokens) > 512:
        too_long += 1
        continue

    if i >= skip:
        last_layer_features = roberta.extract_features(tokens)
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






