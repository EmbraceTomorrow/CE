

import torch
from typing import Tuple
from torchvision import transforms
import random

class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer_data = {}

    def add_data_average(self, train_loader, task):
        cur_task_data = []
        nums_per_class = self.buffer_size // (task + 1)
        for i in range(task):
            random.shuffle(self.buffer_data[i])
            self.buffer_data[i] = self.buffer_data[i][:nums_per_class]
        for data in train_loader:
            _, labels, not_aug_inputs = data
            for input, label in zip(not_aug_inputs, labels):
                cur_task_data.append((input.to(self.device), label.to(self.device)))
        random.shuffle(cur_task_data)
        self.buffer_data[task] = cur_task_data[:nums_per_class]

    def get_data_average(self, task, size, transform=None):
        cur_task_data = []
        for i in range(task):
            cur_task_data.extend(self.buffer_data[i])
        random.shuffle(cur_task_data)
        images, labels = map(list, zip(*cur_task_data[:size]))
        # images = [transform(ee) for ee in images]
        return torch.stack(images).to(self.device), torch.tensor(labels).to(self.device)

    def get_all_data_average(self, task, transform: transforms=None) -> Tuple:
        cur_task_data = []
        for i in range(task):
            cur_task_data.extend(self.buffer_data[i])
        images, labels = zip(*cur_task_data)
        images = [transform(ee.cpu()) for ee in images]
        return torch.stack(images).to(self.device), torch.tensor(labels).to(self.device)
