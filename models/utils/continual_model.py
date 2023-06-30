
import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def reset_classifier(self):
        self.net.classifier = torch.nn.Linear(
                self.net.classifier.in_features, self.net.num_classes).to(self.device)
        self.opt = SGD(self.net.parameters(), lr=self.args.lr,
                           weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)


    def load_cp(self, cp_path, new_classes=None, moco=False, ignore_classifier=False) -> None:
        """
        Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

        :param cp_path: path to checkpoint
        :param new_classes: ignore and rebuild classifier with size `new_classes`
        :param moco: if True, allow load checkpoint for Moco pretraining
        """
        try:
            s = torch.load(cp_path, map_location=self.device)
        except:
            print("Warning!!! The pretrain model can not be loaded! Please check the pretrain model path is valid!")
            return
        if 'state_dict' in s:  # loading moco checkpoint
            if not moco:
                raise Exception(
                    'ERROR: Trying to load a Moco checkpoint without setting moco=True')
            s = {k.replace('encoder_q.', ''): i for k,
                 i in s['state_dict'].items() if 'encoder_q' in k}

        if not ignore_classifier:
            if new_classes is not None:
                self.net.classifier = torch.nn.Linear(
                    self.net.classifier.in_features, self.num_aux_classes).to(self.device)
                for k in list(s):
                    if 'classifier' in k:
                        s.pop(k)
            else:
                cl_weights = [s[k] for k in list(s.keys()) if 'classifier' in k]
                if len(cl_weights) > 0:
                    cl_size = cl_weights[-1].shape[0]
                    self.net.classifier = torch.nn.Linear(
                        self.net.classifier.in_features, cl_size).to(self.device)
        else:
            for k in list(s):
                if 'classifier' in k:
                    s.pop(k)
                    
        for k in list(s):
            if 'net' in k:
                s[k[4:]] = s.pop(k)
        for k in list(s):
            if 'wrappee.' in k:
                s[k.replace('wrappee.', '')] = s.pop(k)
        for k in list(s):
            if '_features' in k:
                s.pop(k)

        try:
            self.net.load_state_dict(s)
        except:
            _, unm = self.net.load_state_dict(s, strict=False)

            if new_classes is not None or ignore_classifier:
                assert all(['classifier' in k for k in unm]
                           ), f"Some of the keys not loaded where not classifier keys: {unm}"
            else:
                assert unm is None, f"Missing keys: {unm}"

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr,
                       weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
        self.device = get_device()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x, **kwargs)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
