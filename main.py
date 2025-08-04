# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(logit, target, topk=(1,)):
    """Compute top-k accuracy, handling different target shapes."""
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


def train(
    epoch,
    loader,
    model1,
    model2,
    optimizer1,
    optimizer2,
    scaler,
    rate_schedule,
    noise_or_not,
    args,
    device,
):
    """Train both models for one epoch, returning averaged metrics."""
    model1.train()
    model2.train()
    loss1_m = AverageMeter()
    acc1_m = AverageMeter()
    loss2_m = AverageMeter()
    acc2_m = AverageMeter()
    pr1_m = AverageMeter()
    pr2_m = AverageMeter()

    for i, (imgs, labs, idxs) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}")):
        if i >= args.num_iter_per_epoch:
            break
        imgs = imgs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)
        inds = idxs.cpu().numpy().transpose()

        with autocast(device_type="cuda"):
            out1 = model1(imgs)
            out2 = model2(imgs)
            p1 = accuracy(out1, labs, topk=(1,))[0]
            p2 = accuracy(out2, labs, topk=(1,))[0]
            loss1, loss2, e1, e2 = loss_coteaching(
                out1, out2, labs, rate_schedule[epoch], inds, noise_or_not
            )

        loss1_m.update(loss1.item(), imgs.size(0))
        acc1_m.update(p1.item(), imgs.size(0))
        loss2_m.update(loss2.item(), imgs.size(0))
        acc2_m.update(p2.item(), imgs.size(0))
        pr1_m.update(e1 * 100, imgs.size(0))
        pr2_m.update(e2 * 100, imgs.size(0))

        optimizer1.zero_grad()
        scaler.scale(loss1).backward()
        scaler.unscale_(optimizer1)
        torch.nn.utils.clip_grad_norm_(model1.parameters(), args.max_norm)
        scaler.step(optimizer1)

        optimizer2.zero_grad()
        scaler.scale(loss2).backward()
        scaler.unscale_(optimizer2)
        torch.nn.utils.clip_grad_norm_(model2.parameters(), args.max_norm)
        scaler.step(optimizer2)

        scaler.update()

        if (i + 1) % args.print_freq == 0:
            print(
                f"Iter [{i+1}/{len(loader)}] "
                f"L1 {loss1_m.val:.4f} ({loss1_m.avg:.4f}) "
                f"L2 {loss2_m.val:.4f} ({loss2_m.avg:.4f}) "
                f"Acc1 {acc1_m.val:.2f}% ({acc1_m.avg:.2f}%) "
                f"Acc2 {acc2_m.val:.2f}% ({acc2_m.avg:.2f}%)"
            )

    return acc1_m.avg, acc2_m.avg, pr1_m.avg, pr2_m.avg


def evaluate(loader, model1, model2, loss_fn, device):
    """Evaluate both models over the test set, returning accuracies and losses."""
    model1.eval()
    model2.eval()
    acc1_m = AverageMeter()
    acc2_m = AverageMeter()
    loss1_m = AverageMeter()
    loss2_m = AverageMeter()

    with torch.no_grad():
        for imgs, labs, _ in tqdm(loader, desc="Evaluate"):
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            out1 = model1(imgs)
            out2 = model2(imgs)
            l1 = loss_fn(out1, labs)
            l2 = loss_fn(out2, labs)
            a1 = accuracy(out1, labs, topk=(1,))[0]
            a2 = accuracy(out2, labs, topk=(1,))[0]
            loss1_m.update(l1.item(), imgs.size(0))
            loss2_m.update(l2.item(), imgs.size(0))
            acc1_m.update(a1.item(), imgs.size(0))
            acc2_m.update(a2.item(), imgs.size(0))

    return acc1_m.avg, acc2_m.avg, loss1_m.avg, loss2_m.avg


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
    parser.add_argument("--top_bn", action="store_true")
    parser.add_argument("--dataset", type=str, default="cxr")
    parser.add_argument("--n_epoch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_iter_per_epoch", type=int, default=500)
    parser.add_argument("--epoch_decay_start", type=int, default=80)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--T_0", type=int, default=10)
    parser.add_argument("--T_mult", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Data
    batch_size = args.batch_size
    if args.dataset == "cxr":
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

    # Setup
    forget_rate = args.noise_rate if args.forget_rate is None else args.forget_rate
    noise_or_not = train_dataset.noise_or_not
    rate_schedule = np.ones(args.n_epoch) * forget_rate
    rate_schedule[: args.num_gradual] = np.linspace(
        0, forget_rate**args.exponent, args.num_gradual
    )

    save_dir = os.path.join(args.result_dir, args.dataset, "coteaching")
    os.makedirs(save_dir, exist_ok=True)
    txtfile = os.path.join(
        save_dir, f"{args.dataset}_coteaching_{args.noise_type}_{args.noise_rate}.txt"
    )
    if os.path.exists(txtfile):
        bak = txtfile + ".bak-" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        os.rename(txtfile, bak)

    model1 = CNN(input_channel=3, n_outputs=12).to(device)
    model2 = CNN(input_channel=3, n_outputs=12).to(device)
    optimizer1 = AdamW(model1.parameters(), lr=args.lr)
    optimizer2 = AdamW(model2.parameters(), lr=args.lr)
    scheduler1 = CosineAnnealingWarmRestarts(
        optimizer1, T_0=args.T_0, T_mult=args.T_mult, eta_min=0
    )
    scheduler2 = CosineAnnealingWarmRestarts(
        optimizer2, T_0=args.T_0, T_mult=args.T_mult, eta_min=0
    )
    scaler = GradScaler()

    # Log header
    with open(txtfile, "w") as f:
        f.write(
            "epoch train_acc1 train_acc2 test_acc1 test_acc2 test_loss1 test_loss2 pure_ratio1 pure_ratio2\n"
        )

    # Initial eval
    test_acc1, test_acc2, test_loss1, test_loss2 = evaluate(
        test_loader, model1, model2, F.cross_entropy, device
    )
    print(
        f"Epoch 0 - Test Acc: M1 {test_acc1:.2f}%, M2 {test_acc2:.2f}% | "
        f"Test Loss: L1 {test_loss1:.4f}, L2 {test_loss2:.4f}"
    )
    with open(txtfile, "a") as f:
        f.write(
            f"0 {0:.4f} {0:.4f} {test_acc1:.2f} {test_acc2:.2f} "
            f"{test_loss1:.4f} {test_loss2:.4f} {0:.2f} {0:.2f}\n"
        )

    # Training loop
    for epoch in range(1, args.n_epoch):
        scheduler1.step(epoch)
        scheduler2.step(epoch)

        train_acc1, train_acc2, pure1, pure2 = train(
            epoch,
            train_loader,
            model1,
            model2,
            optimizer1,
            optimizer2,
            scaler,
            rate_schedule,
            noise_or_not,
            args,
            device,
        )
        test_acc1, test_acc2, test_loss1, test_loss2 = evaluate(
            test_loader, model1, model2, F.cross_entropy, device
        )

        print(
            f"Epoch {epoch} - "
            f"Train Acc: M1 {train_acc1:.2f}%, M2 {train_acc2:.2f}% | "
            f"Test Acc: M1 {test_acc1:.2f}%, M2 {test_acc2:.2f}% | "
            f"Test Loss: L1 {test_loss1:.4f}, L2 {test_loss2:.4f}"
        )
        with open(txtfile, "a") as f:
            f.write(
                f"{epoch} {train_acc1:.4f} {train_acc2:.4f} {test_acc1:.2f} "
                f"{test_acc2:.2f} {test_loss1:.4f} {test_loss2:.4f} "
                f"{pure1:.2f} {pure2:.2f}\n"
            )


if __name__ == "__main__":
    main()
