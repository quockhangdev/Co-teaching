import os
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
from .utils import noisify

class CXR(data.Dataset):
    """
    Dataset `CXR` với cấu trúc thư mục:
        root/
            train/
                class0/
                class1/
                ...
            val/
                class0/
                class1/
                ...
    Hỗ trợ thêm noise trên nhãn giống MNIST.noisify và cung cấp số class.

    Args:
        root (str): Thư mục gốc chứa `train/` và `val/`.
        train (bool): Nếu True load từ `train/`, ngược lại load từ `val/`.
        transform (callable): Transform ảnh.
        target_transform (callable): Transform nhãn.
        noise_type (str): Kiểu noise (`'sym'`, `'asym'`, hoặc `'clean'`).
        noise_rate (float): Tỉ lệ noise.
        random_state (int): Seed cho noisify.
    Attributes:
        classes (List[str]): Tên các class.
        class_to_idx (Dict[str,int]): Map tên class -> idx.
        samples (List[(str,int)]): Danh sách (đường dẫn, nhãn).
        labels (List[int]): Nhãn gốc.
        noisy_labels (List[int]): Nhãn sau khi thêm noise.
        actual_noise_rate (float): Tỉ lệ noise thực tế.
        noise_or_not (np.ndarray): Mask mẫu nào giữ nhãn gốc.
        nb_classes (int): Số class trong dataset.
    """
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        noise_type="clean",
        noise_rate=0.0,
        random_state=0,
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state
        self.dataset = os.path.basename(self.root) or "custom"

        # Chọn folder `train` hoặc `val`
        split = "train" if self.train else "val"
        self.split_dir = os.path.join(self.root, split)
        if not os.path.isdir(self.split_dir):
            raise RuntimeError(f"Folder not found: {self.split_dir}")

        # Scan classes và build list samples
        self.classes = sorted(d.name for d in os.scandir(self.split_dir) if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            folder = os.path.join(self.split_dir, cls)
            for fn in os.listdir(folder):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                    self.samples.append((os.path.join(folder, fn), self.class_to_idx[cls]))

        # Số class
        self.nb_classes = len(self.classes)

        # Gốc labels
        self.labels = [lbl for _, lbl in self.samples]

        # Nếu train và cần noisify
        if self.train and self.noise_type != "clean":
            train_labels = np.array(self.labels).reshape(-1, 1)
            noisy, actual_rate = noisify(
                dataset=self.dataset,
                train_labels=train_labels,
                noise_type=self.noise_type,
                noise_rate=self.noise_rate,
                random_state=self.random_state,
                nb_classes=self.nb_classes,
            )
            self.noisy_labels = [n[0] for n in noisy]
            self.actual_noise_rate = actual_rate
            self.noise_or_not = np.array(self.noisy_labels) == np.array(self.labels)
        else:
            self.noisy_labels = list(self.labels)
            self.actual_noise_rate = 0.0
            self.noise_or_not = np.ones(len(self.labels), dtype=bool)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        # Chọn nhãn (noisy nếu áp noise)
        if self.train and self.noise_type != "clean":
            target = self.noisy_labels[index]
        else:
            target = self.labels[index]

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        s = f"Dataset {self.__class__.__name__}\n"
        s += f"    Split: {'train' if self.train else 'val'} ({len(self)} samples)\n"
        s += f"    Classes ({self.nb_classes}): {self.classes}\n"
        s += f"    Noise: {self.noise_type} @ {self.noise_rate} (actual {self.actual_noise_rate:.3f})\n"
        return s

# Example usage:
# from torchvision import transforms
# train_dataset = CXR(root=args.data_path, train=True, transform=train_transform,
#                     noise_type=args.noise_type, noise_rate=args.noise_rate)
# val_dataset   = CXR(root=args.data_path, train=False, transform=val_transform)
# num_classes   = train_dataset.nb_classes