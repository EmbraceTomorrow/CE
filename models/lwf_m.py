

import torch
from datasets import get_dataset
from torch.optim import SGD
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from torchvision import transforms

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Learning without Forgetting.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_aux_dataset_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--softmax_temp', type=float, required=True,
                        help='Temperature of the softmax function.')
    parser.add_argument('--wd_reg', type=float, required=True,
                        help='Coefficient of the weight decay regularizer.')
    parser.add_argument('--lambda_ftc', type=float, required=True,
                        help='finetune classifier.')
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class LwfM(ContinualModel):
    NAME = 'lwfm'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(LwfM, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.buf_transform = transforms.Compose(self.transform.transforms)
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.dataset = get_dataset(args)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        nc = get_dataset(args).N_TASKS * self.cpt
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)

        self.num_classes = nc

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.load_cp(self.args.load_cp, moco=True, ignore_classifier=True)
            self.reset_classifier()

        self.net.eval()
        if self.current_task > 0:
            # warm-up
            opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
            for epoch in range(self.args.n_epochs):
                for i, data in enumerate(dataset.train_loader):
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                    opt.zero_grad()
                    with torch.no_grad():
                        feats = self.net(inputs, returnt = 'features')
                    mask = self.eye[(self.current_task + 1) * self.cpt - 1] ^ self.eye[self.current_task * self.cpt - 1]
                    outputs = self.net.classifier(feats)[:, mask]
                    loss = self.loss(outputs, labels - self.current_task * self.cpt)
                    loss.backward()
                    opt.step()

            self.old_net = self.dataset.get_backbone()
            self.old_net.load_state_dict(self.net.state_dict())
            self.old_net = self.old_net.to(self.device)
            for p in self.old_net.parameters():
                p.requires_grad = False

        self.net.train()

        self.current_task += 1

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        labels = labels.long()
        self.opt.zero_grad()
        outputs = self.net(inputs)

        mask = self.eye[self.current_task * self.cpt - 1]
        loss = self.loss(outputs[:, mask], labels)

        if len(self.buffer.buffer_data) > 0:
            buf_inputs, buf_labels = self.buffer.get_data_average(self.current_task-1,
                self.args.minibatch_size, transform=None)
            buf_inputs = torch.stack([self.buf_transform(ee.cpu()) for ee in buf_inputs]).to(self.device)
            buf_old_net_outputs = self.old_net(buf_inputs)
            buf_outputs = self.net(buf_inputs)
            mask = self.eye[(self.current_task - 1) * self.cpt - 1]
            loss += self.args.alpha * modified_kl_div(smooth(self.soft(buf_old_net_outputs[:, mask]).to(self.device), self.args.softmax_temp, 1),
                                                      smooth(self.soft(buf_outputs[:, mask]), self.args.softmax_temp, 1))

        loss += self.args.wd_reg * torch.sum(self.net.get_params() ** 2)
        loss.backward()
        self.opt.step()

        return loss.item()

    def add_buffer(self, train_loader):
        self.buffer.add_data_average(train_loader, self.current_task-1)