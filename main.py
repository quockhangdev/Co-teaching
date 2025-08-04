# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
parser.add_argument("--batch_size", type=int, default=32)
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
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--num_iter_per_epoch", type=int, default=500)
parser.add_argument("--epoch_decay_start", type=int, default=80)
parser.add_argument(
    "--max_norm", type=float, default=1.0, help="Max norm for gradient clipping"
)
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

# Gradual forget-rate schedule
rate_schedule = np.ones(args.n_epoch) * forget_rate
rate_schedule[: args.num_gradual] = np.linspace(
    0, forget_rate**args.exponent, args.num_gradual
)

# Prepare output
dirs = os.path.join(args.result_dir, args.dataset, "coteaching")
os.makedirs(dirs, exist_ok=True)
model_str = f"{args.dataset}_coteaching_{args.noise_type}_{args.noise_rate}"
txtfile = os.path.join(dirs, f"{model_str}.txt")
if os.path.exists(txtfile):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    os.rename(txtfile, txtfile + f".bak-{timestamp}")


def accuracy(logit, target, topk=(1,)):
    if target.ndim == 2 and target.size(1) == 1:
        target = target.squeeze(1)
    elif target.ndim > 1:
        target = target.argmax(dim=1)
    _, pred = logit.topk(max(topk), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return res


def train(loader, epoch, m1, opt1, m2, opt2, scaler):
    m1.train()
    m2.train()
    t1_total = t1_correct = t2_total = t2_correct = 0
    pr1_list, pr2_list = [], []

    for i, (imgs, labs, idxs) in tqdm(enumerate(loader), total=len(loader)):
        if i > args.num_iter_per_epoch:
            break
        imgs, labs = imgs.to(device, non_blocking=True), labs.to(
            device, non_blocking=True
        )
        inds = idxs.cpu().numpy().transpose()

        with autocast(device_type="cuda"):
            out1 = m1(imgs)
            p1 = accuracy(out1, labs, topk=(1,))[0]
            t1_total += 1
            t1_correct += p1.item()

            out2 = m2(imgs)
            p2 = accuracy(out2, labs, topk=(1,))[0]
            t2_total += 1
            t2_correct += p2.item()

            loss1, loss2, e1, e2 = loss_coteaching(
                out1, out2, labs, rate_schedule[epoch], inds, noise_or_not
            )

        pr1_list.append(100 * e1)
        pr2_list.append(100 * e2)

        # Model 1 update
        opt1.zero_grad()
        scaler.scale(loss1).backward()
        scaler.unscale_(opt1)
        torch.nn.utils.clip_grad_norm_(m1.parameters(), args.max_norm)
        scaler.step(opt1)

        # Model 2 update
        opt2.zero_grad()
        scaler.scale(loss2).backward()
        scaler.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(m2.parameters(), args.max_norm)
        scaler.step(opt2)

        scaler.update()

        if (i + 1) % args.print_freq == 0:
            print(
                f"Epoch [{epoch+1}/{args.n_epoch}], Iter [{i+1}] "
                f"L1:{loss1.item():.4f}, L2:{loss2.item():.4f}, "
                f"Acc1:{p1.item():.2f}%, Acc2:{p2.item():.2f}%"
            )

    return t1_correct / t1_total, t2_correct / t2_total, pr1_list, pr2_list


def evaluate(loader, m1, m2):
    m1.eval()
    m2.eval()
    c1 = c2 = total = 0
    with torch.no_grad():
        for imgs, labs, _ in tqdm(loader, total=len(loader)):
            imgs, labs = imgs.to(device, non_blocking=True), labs.to(
                device, non_blocking=True
            )
            pred1 = m1(imgs).argmax(1)
            c1 += (pred1 == labs).sum().item()
            pred2 = m2(imgs).argmax(1)
            c2 += (pred2 == labs).sum().item()
            total += labs.size(0)
    return 100 * c1 / total, 100 * c2 / total


def main():
    print("Loading data...")
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

    print("Building models...")
    cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes).to(device)
    cnn2 = CNN(input_channel=input_channel, n_outputs=num_classes).to(device)

    optimizer1 = AdamW(cnn1.parameters(), lr=args.lr)
    optimizer2 = AdamW(cnn2.parameters(), lr=args.lr)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.n_epoch, eta_min=0)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.n_epoch, eta_min=0)

    scaler = GradScaler()

    with open(txtfile, "a") as f:
        f.write(
            "epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n"
        )

    print("Evaluating untrained models...")
    torch.cuda.empty_cache()
    ta1, ta2 = evaluate(test_loader, cnn1, cnn2)
    print(f"Epoch 0 - Test Acc: M1 {ta1:.2f}%, M2 {ta2:.2f}%")
    with open(txtfile, "a") as f:
        f.write(f"0: 0 0 {ta1:.2f} {ta2:.2f} 0 0\n")

    for epoch in range(1, args.n_epoch):
        scheduler1.step()
        scheduler2.step()
        tr1, tr2, pr1, pr2 = train(
            train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2, scaler
        )
        torch.cuda.empty_cache()
        te1, te2 = evaluate(test_loader, cnn1, cnn2)
        print(
            f"Epoch {epoch} - Train M1:{tr1:.2f}%, M2:{tr2:.2f}% | "
            f"Test M1:{te1:.2f}%, M2:{te2:.2f}%"
        )
        with open(txtfile, "a") as f:
            f.write(
                f"{epoch}: {tr1:.4f} {tr2:.4f} {te1:.2f} {te2:.2f} "
                f"{np.mean(pr1):.2f} {np.mean(pr2):.2f}\n"
            )


if __name__ == "__main__":
    main()
