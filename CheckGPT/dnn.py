import os
import copy
import h5py
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from configparser import ConfigParser

from model import AttenLSTM

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('domain', type=str)
parser.add_argument('task', type=int)
parser.add_argument('expid', type=str)
parser.add_argument('--early', type=float, default=99.8)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--dataamount', type=int, default=1000)
parser.add_argument('--trans', type=int, default=0)
parser.add_argument('--splitr', type=float, default=0.8)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--mdomain', type=str, default="CS")
parser.add_argument('--mtask', type=int, default=1)
parser.add_argument('--printall', type=int, default=1)
parser.add_argument('--cont', type=int, default=0)
parser.add_argument('--adam', type=int, default=1)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batchsize', type=int, default=320)
parser.add_argument('--meta', type=int, default=0)
args = parser.parse_args()

LOG_INTERVAL = 50
ID = args.expid

if not os.path.exists("./exp/{}".format(ID)):
    os.mkdir("./exp/{}".format(ID))

SEED = args.seed
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED = bool(args.pretrain)
TRANSFER = bool(args.trans)
SGD_OR_ADAM = "adam" if args.adam else "sgd"
LEARNING_RATE = 2e-4 if SGD_OR_ADAM == "adam" else 1e-3
N_EPOCH = 200 if (TRANSFER or SGD_OR_ADAM == "sgd") else 100
CONTINUE = args.cont
BATCH_SIZE = args.batchsize
EARLYSTOP = args.early
SAVE = bool(args.save)
TEST = bool(args.test)
PRINT_ALL = bool(args.printall)
TEST_SIZE = 512 if TEST else BATCH_SIZE
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True

domain, task = args.domain, str(args.task)
m_domain, m_task = args.mdomain, str(args.mtask)

if TRANSFER:
    assert m_domain is not None


class MyDataset(data.Dataset):
    def __init__(self, archive):
        self.archive = archive
        self.length = len(h5py.File(self.archive, 'r')["data"])
        self.dataset, self.labels = None, None

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.archive, 'r')["data"]
            self.labels = h5py.File(self.archive, 'r')["label"]
        x, y = torch.from_numpy(self.dataset[index]).float(), torch.from_numpy(self.labels[index]).long()
        return x, y

    def __len__(self):
        return self.length


class TransNet(nn.Module):
    def __init__(self, host, mode=0):
        super(TransNet, self).__init__()
        self.host = host
        self.mode = mode
        self.dropout = nn.Dropout(p=0.05)
        self.layers = copy.deepcopy(self.host.fc)

    def forward(self, x):
        x = self.layers(self.dropout(x))
        return x


def save_checkpoint(model, path, optimizer, scheduler, epoch, acc):
    info_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": scheduler.state_dict(),
        "last_epoch": epoch,
        "best_acc": acc
    }

    torch.save(info_dict, path)
    return


def load_checkpoint(model, path, optimizer, scheduler):
    cp = torch.load(path)
    model.load_state_dict(cp["model_state_dict"], strict=True)
    optimizer.load_state_dict(cp["optimizer_state_dict"])
    scheduler.load_state_dict(cp["lr_scheduler_state_dict"])
    last_epoch = cp["last_epoch"]
    best_acc = cp["best_acc"]
    return last_epoch, best_acc


def load_data(domain, task, size_train=96, size_test=96):
    if not TRANSFER:
        dir_ = './embeddings/{}_TASK{}_{}.{}'
        full_dataset = MyDataset(dir_.format(domain, task, args.dataamount, "h5"))
        torch.random.manual_seed(args.seed)
        train_size = int(args.splitr * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    else:
        torch.random.manual_seed(SEED)
        full_data = MyDataset('./embeddings/s{}{}_t{}{}.h5'.format(m_domain, args.mtask,
                                                                                     domain, task))
        train_size = args.dataamount
        test_size = len(full_data) - train_size

        train_data, _ = torch.utils.data.random_split(full_data, [train_size, test_size])
        test_data = MyDataset('./embeddings/s{}{}_t{}{}.h5'.format(m_domain, args.mtask,
                                                                                     domain, task))

    train_loader = DataLoader(dataset=train_data,
                              num_workers=4,
                              batch_size=size_train,
                              drop_last=True,
                              shuffle=True,
                              pin_memory=True,
                              prefetch_factor=2)
    test_loader = DataLoader(dataset=test_data,
                              num_workers=4,
                              batch_size=size_test,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=True,
                              prefetch_factor=2)
    return train_loader, test_loader


def test(model, dataloader, epoch, print_freq=1):
    model.eval()
    n_correct, correct_0, correct_1, sum_0, sum_1 = 0, 0, 0, 0, 0
    start = time.time()

    with torch.no_grad():
        for i, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE).squeeze(1)
            class_output = model(t_img)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == t_label).sum().item()
            correct_0 += ((pred[1] == t_label) * (t_label == 0)).sum().item()
            correct_1 += ((pred[1] == t_label) * (t_label == 1)).sum().item()
            sum_0 += (t_label == 0).sum().item()
            sum_1 += (t_label == 1).sum().item()
            if i % (LOG_INTERVAL*print_freq) == 0 and PRINT_ALL:
                print('Batch: [{}/{}], Time used: {:.4f}s'.format(i, len(dataloader), time.time() - start))

    accu = float(n_correct) / len(dataloader.dataset) * 100
    accu_0 = float(correct_0) / sum_0 * 100
    accu_1 = float(correct_1) / sum_1 * 100
    if PRINT_ALL:
        print('{}{}， Epoch:{}, Test accuracy: {:.4f}%, Acc_GPT: {:.4f}%, Acc_Human: {:.4f}%'.format(args.task,
                                                                                                    args.domain, epoch,
                                                        accu, accu_0, accu_1))
    return accu, accu_0, accu_1


def train(model, optimizer, scheduler, dataloader, test_loader):
    loss_class = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if CONTINUE:
        last_epoch, best_acc = load_checkpoint(model, "./exp/{}_Task{}.pth".format(domain, task),
                                               optimizer, scheduler)
        best_acc = test(model, test_loader, -1)
        print("Checkpoint Loaded.")
    else:
        last_epoch, best_acc = 1, 0

    len_dataloader = len(dataloader)
    for epoch in range(last_epoch, N_EPOCH + 1):
        start = time.time()
        model.train()
        data_iter = iter(dataloader)
        n_correct = 0

        i = 1
        while i < len_dataloader + 1:
            data_source = next(data_iter)
            optimizer.zero_grad()

            img, label = data_source[0].to(DEVICE), data_source[1].to(DEVICE).squeeze(1)

            class_output = model(img)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == label).sum().item()
            err = loss_class(class_output, label)
            scaler.scale(err).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % LOG_INTERVAL == 0 and PRINT_ALL:
                print(
                    'Epoch: [{}/{}], Batch: [{}/{}], Err: {:.4f}, Time used: {:.4f}s'.format(
                        epoch, N_EPOCH, i, len_dataloader, err.item(), time.time() - start,
                        ))
            i += 1
            # torch.cuda.empty_cache()

        accu = float(n_correct) / (len(dataloader.dataset)) * 100
        if PRINT_ALL:
            print('{}_TASK{}， Epoch:{}, Train accuracy: {:.4f}%'.format(domain, task, epoch, accu))
        if not TRANSFER and SAVE:
            save_checkpoint(model, "./exp/{}/Checkpoint_{}_Task{}.pth".format(ID, domain, task),
                            optimizer, scheduler, epoch, best_acc)
        acc = test(model, test_loader, epoch)

        scheduler.step()

        if acc > best_acc:
            old_acc, best_acc = best_acc, acc
            if TRANSFER:
                name = "./exp/{}/Best_s_{}{}_t_{}{}.pth".format(ID, m_domain, m_task, domain, task)
            else:
                name = "./exp/{}/Best_{}_Task{}.pth".format(ID, domain, task)
            if SAVE:
                torch.save(model.state_dict(), name)
            if not TRANSFER:
                print("Best model saved.")

        if EARLYSTOP > 0 and best_acc > EARLYSTOP:
            break

    return best_acc


if __name__ == '__main__':
    torch.random.manual_seed(SEED)
    train_loader, test_loader = load_data(domain, task, size_train=BATCH_SIZE, size_test=TEST_SIZE)
    rnn = AttenLSTM(input_size=1024, hidden_size=256, batch_first=True, dropout=args.dropout, bidirectional=True, num_layers=2, device=DEVICE_NAME).to(DEVICE)

    config = ConfigParser()

    config.read('./exp/{}/config.ini'.format(ID))
    config.add_section('main')
    for key, value in vars(args).items():
        config.set('main', key, str(value))

    with open('./exp/{}/config.ini'.format(ID), 'w') as f:
        config.write(f)

    if PRETRAINED or TEST:
        rnn.load_state_dict(torch.load("../Pretrained/{}_Task{}.pth".format(domain, task)), strict=True)
        if args.meta:
            rnn.load_state_dict(torch.load("../Pretrained/Unified_Task123.pth"), strict=True)

    if TRANSFER:
        rnn.load_state_dict(torch.load("../Pretrained/{}_Task{}.pth".format(m_domain, m_task)), strict=True)
        model = TransNet(rnn)
        del rnn
    else:
        model = rnn

    if not TEST:
        for param in model.parameters():
            param.requires_grad = True

    if SGD_OR_ADAM == "adam":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCH)

    if not TEST:
        best_acc = train(model, optimizer, scheduler, train_loader, test_loader)
    else:
        best_acc, _, _ = test(model, test_loader, 0)

    if TRANSFER:
        print("Transfer Learning, S: {}, T: {}, Acc: {:.4f}%".format(m_domain+m_task, domain+task, best_acc))
