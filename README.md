# CheckGPT

## Description

The official repository of paper: "Check Me If You Can: Detecting ChatGPT-Generated Academic Writing using CheckGPT".

A newer version is available [here](https://github.com/liuzey/CheckGPT-v2). The archived artifact is available on [Zenodo](https://10.5281/zenodo.11046882), including the GPABench2 datasets and experiment scripts.

## Table of Contents

- [Data](#data)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Data
Data structure: { {KEY}:{TEXT} }. You may use your own data following this structure.

## Features
To turn text into features, use [*features.py*](CheckGPT/features.py).

## Installation
Run
```bash
pip install -r requirements.txt
```

## Usage
For training, testing and transfer learning, use [*dnn.py*](CheckGPT/dnn.py) using this format:
```bash
python dnn.py {SUBJECT} {TASK} {EXP_ID} 
```

**Examples:**
1. To train a model from scratch on CS and task 1:
```bash
python dnn.py CS 1 0001 
```

2. To test a model on HSS and task 3:
```bash
python dnn.py HSS 3 0001 --test 1
```

3. To evaluate any text, run and follow instructions:
```bash
python test.py
```



