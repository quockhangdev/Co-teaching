# -*- coding:utf-8 -*-
import os
import argparse
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cxr import CXR
from model import CNN
from loss import loss_coteaching


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(logits, target, topk=(1,)):
    """Compute top-k accuracy from logits and true labels."""
    probs = torch.softmax(logits, dim=1)
    if target.ndim == 2 and target.size(1) == 1:
        target = target.squeeze(1)
    elif target.ndim > 1:
        target = target.argmax(dim=1)
    _, pred = probs.topk(max(topk), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return res


def train(
    epoch, loader, m1, m2, opt1, opt2, scaler, schedule, noise_mask, args, device
):
    """Train both models for one epoch."""
    m1.train()
    m2.train()
    meters = {
        k: AverageMeter() for k in ["loss1", "acc1", "loss2", "acc2", "pr1", "pr2"]
    }

    for i, (imgs, labs, idxs) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}")):
        if i >= args.num_iter_per_epoch:
            break
        imgs = imgs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)
        inds = idxs.cpu().numpy()

        with autocast():
            out1 = m1(imgs)
            out2 = m2(imgs)
            p1 = accuracy(out1, labs)[0]
            p2 = accuracy(out2, labs)[0]
            loss1, loss2, e1, e2 = loss_coteaching(
                out1,
                out2,
                labs,
                schedule[epoch],
                inds,
                noise_mask,
                label_smoothing=args.label_smoothing,
            )

        batch = imgs.size(0)
        meters["loss1"].update(loss1.item(), batch)
        meters["acc1"].update(p1.item(), batch)
        meters["loss2"].update(loss2.item(), batch)
        meters["acc2"].update(p2.item(), batch)
        meters["pr1"].update(e1 * 100, batch)
        meters["pr2"].update(e2 * 100, batch)

        opt1.zero_grad()
        scaler.scale(loss1).backward()
        scaler.unscale_(opt1)
        torch.nn.utils.clip_grad_norm_(m1.parameters(), args.max_norm)
        scaler.step(opt1)

        opt2.zero_grad()
        scaler.scale(loss2).backward()
        scaler.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(m2.parameters(), args.max_norm)
        scaler.step(opt2)

        scaler.update()

        if (i + 1) % args.print_freq == 0:
            print(
                f"Iter {i+1}/{len(loader)} "
                f"L1:{meters['loss1'].val:.4f}({meters['loss1'].avg:.4f}) "
                f"Acc1:{meters['acc1'].val:.2f}%({meters['acc1'].avg:.2f}%)"
            )

    return (
        meters["acc1"].avg,
        meters["acc2"].avg,
        meters["pr1"].avg,
        meters["pr2"].avg,
    )


def evaluate(loader, m1, m2, args, device):
    """Evaluate both models, returning accuracies and losses."""
    m1.eval()
    m2.eval()
    meters = {k: AverageMeter() for k in ["loss1", "acc1", "loss2", "acc2"]}

    with torch.no_grad():
        for imgs, labs, _ in tqdm(loader, desc="Evaluate"):
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            out1 = m1(imgs)
            out2 = m2(imgs)
            l1 = F.cross_entropy(out1, labs, label_smoothing=args.label_smoothing)
            l2 = F.cross_entropy(out2, labs, label_smoothing=args.label_smoothing)
            a1 = accuracy(out1, labs)[0]
            a2 = accuracy(out2, labs)[0]
            b = imgs.size(0)
            meters["loss1"].update(l1.item(), b)
            meters["acc1"].update(a1.item(), b)
            meters["loss2"].update(l2.item(), b)
            meters["acc2"].update(a2.item(), b)

    return (
        meters["acc1"].avg,
        meters["acc2"].avg,
        meters["loss1"].avg,
        meters["loss2"].avg,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--result_dir", type=str, default="results/")
    parser.add_argument("--noise_rate", type=float, default=0.2)
    parser.add_argument("--forget_rate", type=float, default=None)
    parser.add_argument("--noise_type", type=str, default="symmetric")
    parser.add_argument("--num_gradual", type=int, default=10)
    parser.add_argument("--exponent", type=float, default=1)
    parser.add_argument("--n_epoch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_iter_per_epoch", type=int, default=500)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--T_0", type=int, default=10)
    parser.add_argument("--T_mult", type=int, default=2)
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing factor"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Only CXR dataset
    train_dataset = CXR(
        root="/home/khmt/Documents/MyKhanh/GNN/data/img_xray",
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
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
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # Setup schedule and logging
    forget = args.noise_rate if args.forget_rate is None else args.forget_rate
    noise_mask = train_dataset.noise_or_not
    schedule = np.ones(args.n_epoch) * forget
    schedule[: args.num_gradual] = np.linspace(
        0, forget**args.exponent, args.num_gradual
    )

    save_dir = os.path.join(args.result_dir, "cxr", "coteaching")
    os.makedirs(save_dir, exist_ok=True)
    txtfile = os.path.join(
        save_dir, f"cxr_coteaching_{args.noise_type}_{args.noise_rate}.txt"
    )
    if os.path.exists(txtfile):
        bak = txtfile + ".bak-" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        os.rename(txtfile, bak)

    model1 = CNN(input_channel=3, n_outputs=12).to(device)
    model2 = CNN(input_channel=3, n_outputs=12).to(device)
    opt1 = AdamW(model1.parameters(), lr=args.lr)
    opt2 = AdamW(model2.parameters(), lr=args.lr)
    sched1 = CosineAnnealingWarmRestarts(
        opt1, T_0=args.T_0, T_mult=args.T_mult, eta_min=0
    )
    sched2 = CosineAnnealingWarmRestarts(
        opt2, T_0=args.T_0, T_mult=args.T_mult, eta_min=0
    )
    scaler = GradScaler()

    # Write log header
    with open(txtfile, "w") as f:
        f.write(
            "epoch train_acc1 train_acc2 test_acc1 test_acc2 test_loss1 test_loss2 "
            "pure_ratio1 pure_ratio2\n"
        )

    # Initial evaluation
    test_acc1, test_acc2, test_loss1, test_loss2 = evaluate(
        test_loader, model1, model2, args, device
    )
    print(
        f"Epoch 0 - Test Acc: M1 {test_acc1:.2f}%, M2 {test_acc2:.2f}% | "
        f"Test Loss: L1 {test_loss1:.4f}, L2 {test_loss2:.4f}"
    )
    with open(txtfile, "a") as f:
        f.write(
            f"0 0.0000 0.0000 {test_acc1:.2f} {test_acc2:.2f} "
            f"{test_loss1:.4f} {test_loss2:.4f} 0.00 0.00\n"
        )

    # Training loop
    for epoch in range(1, args.n_epoch):
        sched1.step(epoch)
        sched2.step(epoch)
        tr_acc1, tr_acc2, pr1, pr2 = train(
            epoch,
            train_loader,
            model1,
            model2,
            opt1,
            opt2,
            scaler,
            schedule,
            noise_mask,
            args,
            device,
        )
        te_acc1, te_acc2, te_loss1, te_loss2 = evaluate(
            test_loader, model1, model2, args, device
        )

        print(
            f"Epoch {epoch} - Train M1:{tr_acc1:.2f}%, M2:{tr_acc2:.2f}% | "
            f"Test M1:{te_acc1:.2f}%, M2:{te_acc2:.2f}% | "
            f"Test Loss: L1 {te_loss1:.4f}, L2 {te_loss2:.4f}"
        )
        with open(txtfile, "a") as f:
            f.write(
                f"{epoch} {tr_acc1:.4f} {tr_acc2:.4f} {te_acc1:.2f} "
                f"{te_acc2:.2f} {te_loss1:.4f} {te_loss2:.4f} {pr1:.2f} {pr2:.2f}\n"
            )


if __name__ == "__main__":
    main()
