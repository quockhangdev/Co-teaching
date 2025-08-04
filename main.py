# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.cxr import CXR
from model import CNN
import argparse
import numpy as np
import datetime
from loss import loss_coteaching
from torch.amp import autocast, GradScaler
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--result_dir", type=str, default="results/")
parser.add_argument("--noise_rate", type=float, default=0.2)
parser.add_argument("--forget_rate", type=float, default=None)
parser.add_argument("--noise_type", type=str, default="symmetric")
parser.add_argument("--num_gradual", type=int, default=10)
parser.add_argument("--exponent", type=float, default=1)
parser.add_argument("--top_bn", action="store_true")
parser.add_argument("--dataset", type=str, default="cxr")
parser.add_argument("--n_epoch", type=int, default=200)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--print_freq", type=int, default=50)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--num_iter_per_epoch", type=int, default=400)
parser.add_argument("--epoch_decay_start", type=int, default=80)
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Device & CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Dataset setup
batch_size = args.batch_size
input_channel, num_classes = 3, 12  # default for CXR

if args.dataset == "cxr":
    train_dataset = CXR(
        root="/home/khmt/Documents/MyKhanh/GNN/data/img_xray",
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
    )
    test_dataset = CXR(
        root="/home/khmt/Documents/MyKhanh/GNN/data/img_xray",
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
    )

# Forget rate setup
forget_rate = args.noise_rate if args.forget_rate is None else args.forget_rate
noise_or_not = train_dataset.noise_or_not

# Learning rate schedule
alpha_plan = [args.lr] * args.n_epoch
beta1_plan = [0.9] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = (
        float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * args.lr
    )
    beta1_plan[i] = 0.1


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha_plan[epoch]
        param_group["betas"] = (beta1_plan[epoch], 0.999)


rate_schedule = np.ones(args.n_epoch) * forget_rate
rate_schedule[: args.num_gradual] = np.linspace(
    0, forget_rate**args.exponent, args.num_gradual
)

save_dir = os.path.join(args.result_dir, args.dataset, "coteaching")
os.makedirs(save_dir, exist_ok=True)
model_str = f"{args.dataset}_coteaching_{args.noise_type}_{args.noise_rate}"
txtfile = os.path.join(save_dir, f"{model_str}.txt")

if os.path.exists(txtfile):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    os.rename(txtfile, txtfile + f".bak-{timestamp}")


def accuracy(logit, target, topk=(1,)):
    _, pred = logit.topk(max(topk), 1, True, True)
    correct = pred.t().eq(target.view(1, -1).expand_as(pred))
    res = [
        correct[:k].reshape(-1).float().sum(0).mul_(100.0 / target.size(0))
        for k in topk
    ]
    return res


def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, scaler):
    model1.train()
    model2.train()

    train_total, train_correct = 0, 0
    train_total2, train_correct2 = 0, 0
    pure_ratio_1_list, pure_ratio_2_list = [], []

    for i, (images, labels, indexes) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        if i > args.num_iter_per_epoch:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        ind = indexes.cpu().numpy().transpose()

        with autocast(device_type="cuda"):
            logits1 = model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))[0]
            train_total += 1
            train_correct += prec1.item()

            logits2 = model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))[0]
            train_total2 += 1
            train_correct2 += prec2.item()

            loss_1, loss_2, pr1, pr2 = loss_coteaching(
                logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not
            )

        pure_ratio_1_list.append(100 * pr1)
        pure_ratio_2_list.append(100 * pr2)

        optimizer1.zero_grad()
        scaler.scale(loss_1).backward()
        scaler.step(optimizer1)

        optimizer2.zero_grad()
        scaler.scale(loss_2).backward()
        scaler.step(optimizer2)

        scaler.update()

        if (i + 1) % args.print_freq == 0:
            print(
                f"Epoch [{epoch+1}/{args.n_epoch}], Iter [{i+1}] Loss1: {loss_1.item():.4f}, Loss2: {loss_2.item():.4f}, Acc1: {prec1.item():.2f}%, Acc2: {prec2.item():.2f}%"
            )

    return (
        train_correct / train_total,
        train_correct2 / train_total2,
        pure_ratio_1_list,
        pure_ratio_2_list,
    )


def evaluate(test_loader, model1, model2):
    model1.eval()
    model2.eval()
    correct1 = correct2 = total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, total=len(test_loader)):
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            logits1 = model1(images)
            pred1 = logits1.argmax(1)
            correct1 += (pred1 == labels).sum().item()

            logits2 = model2(images)
            pred2 = logits2.argmax(1)
            correct2 += (pred2 == labels).sum().item()

            total += labels.size(0)
    return 100.0 * correct1 / total, 100.0 * correct2 / total


def main():
    print("loading dataset...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    print("building model...")
    cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes).to(device)
    cnn2 = CNN(input_channel=input_channel, n_outputs=num_classes).to(device)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=args.lr)
    scaler = GradScaler()

    with open(txtfile, "a") as f:
        f.write(
            "epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n"
        )

    print("Evaluating untrained models...")
    torch.cuda.empty_cache()
    test_acc1, test_acc2 = evaluate(test_loader, cnn1, cnn2)
    print(f"Epoch 0 - Test Accuracy: Model1 {test_acc1:.2f}%, Model2 {test_acc2:.2f}%")

    with open(txtfile, "a") as f:
        f.write(f"0: 0 0 {test_acc1:.2f} {test_acc2:.2f} 0 0\n")

    for epoch in range(1, args.n_epoch):
        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pr1_list, pr2_list = train(
            train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2, scaler
        )

        torch.cuda.empty_cache()
        test_acc1, test_acc2 = evaluate(test_loader, cnn1, cnn2)
        mean_pr1 = np.mean(pr1_list)
        mean_pr2 = np.mean(pr2_list)

        print(
            f"Epoch {epoch} - Train Acc1: {train_acc1:.2f}%, Acc2: {train_acc2:.2f}%, Test Acc1: {test_acc1:.2f}%, Acc2: {test_acc2:.2f}%"
        )

        with open(txtfile, "a") as f:
            f.write(
                f"{epoch}: {train_acc1:.4f} {train_acc2:.4f} {test_acc1:.2f} {test_acc2:.2f} {mean_pr1:.2f} {mean_pr2:.2f}\n"
            )


if __name__ == "__main__":
    main()
