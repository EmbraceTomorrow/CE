import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser, \
    add_aux_dataset_args
from utils.buffer import Buffer

from torch.optim import SGD
from torchvision import transforms
from datasets import get_dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datasets.buffer_dataset import BufferDataset
import tqdm


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Classification Expander')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--lambda_ftc', type=float, required=True,
                        help='Finetune classifier weight.')
    parser.add_argument('--ftc_epoch', type=int, required=True,
                        help='Finetune classifier epoch.')
    parser.add_argument('--ftc_lr', type=float, required=True,
                        help='Finetune classifier learning rate.')
    return parser


class CE(ContinualModel):
    NAME = 'ce'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(CE, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.buf_transform = transforms.Compose(self.transform.transforms)
        self.ds = get_dataset(args)
        self.cpt = self.ds.N_CLASSES_PER_TASK
        self.n_tasks = self.ds.N_TASKS
        self.num_classes = self.n_tasks * self.cpt
        self.tbwriter = SummaryWriter('logs')
        self.task = 0
        self.prenet = None

    def end_task(self, dataset):
        self.train()
        self.task += 1

    def begin_task(self, dataset):

        if self.task == 0:
            self.load_cp(self.args.load_cp, moco=True, ignore_classifier=True)
            self.opt = SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd,
                           momentum=self.args.optim_mom)
            self.net.train()
        else:
            self.prenet = self.ds.get_backbone()
            self.prenet.load_state_dict(self.net.state_dict())
            self.prenet = self.prenet.to(self.device)
            for p in self.prenet.parameters():
                p.requires_grad = False
            self.opt = SGD(self.net.parameters(), lr=self.args.lr,
                           weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
            self.prenet.train()
            self.net.train()

    def observe(self, inputs, labels, not_aug_inputs, epoch=0):

        labels = labels.long()
        B = len(inputs)

        if len(self.buffer.buffer_data) > 0:
            buf_inputs, buf_labels = self.buffer.get_data_average(self.task,
                                                                  self.args.minibatch_size, transform=None)
            buf_labels = buf_labels.long()
            buf_inputs = torch.stack([self.buf_transform(ee.cpu()) for ee in buf_inputs]).to(self.device)
            inputs = torch.cat([inputs, buf_inputs]).to(self.device)

        self.opt.zero_grad()

        loss_alpha = torch.tensor(0., requires_grad=True)
        loss_beta = torch.tensor(0., requires_grad=True)

        all_outputs, _ = self.net(inputs, returnt='full')
        outputs = all_outputs[:B]
        buf_outputs = all_outputs[B:]

        if self.prenet:
            prenet_all_outputs, _ = self.prenet(inputs, returnt='full')
            prenet_buf_outputs = prenet_all_outputs[B:]

        loss_cross_entropy = self.loss(outputs[:, self.task * self.cpt:(self.task + 1) * self.cpt], labels % self.cpt)
        if len(self.buffer.buffer_data) > 0:
            loss_alpha = F.mse_loss(buf_outputs[:, :(self.task + 1) * self.cpt],
                                    prenet_buf_outputs[:, :(self.task + 1) * self.cpt])
            loss_beta = self.loss(buf_outputs[:, :(self.task + 1) * self.cpt], buf_labels)

        loss = loss_cross_entropy + self.args.alpha * loss_alpha + self.args.beta * loss_beta

        self.tbwriter.add_scalar('train/cross_entropy_loss', loss_cross_entropy.item())
        self.tbwriter.add_scalar('train/loss_alpha', loss_alpha.item())
        self.tbwriter.add_scalar('train/loss_beta', loss_beta.item())

        loss.backward()
        self.opt.step()

        return loss.item()

    def add_buffer(self, train_loader):
        self.buffer.add_data_average(train_loader, self.task)

    def finetune_classifier(self):
        self.opt = torch.optim.SGD(self.net.classifier.parameters(
        ), lr=self.args.ftc_lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
        self.net.train()
        if len(self.buffer.buffer_data) > 0:
            buf_inputs, buf_labels = self.buffer.get_all_data_average(self.task, self.buf_transform)
            dataset = BufferDataset(buf_inputs, buf_labels)
            dataloader = DataLoader(dataset=dataset, batch_size=self.args.minibatch_size, shuffle=True, drop_last=False)
            for _ in tqdm.tqdm(range(self.args.ftc_epoch)):
                for i, data in enumerate(dataloader):
                    inputs, labels = data
                    labels = labels.long()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    buf_output = self.net(inputs)
                    self.opt.zero_grad()
                    loss = self.args.lambda_ftc * self.loss(buf_output[:, :self.task * self.cpt], labels)
                    self.tbwriter.add_scalar('train/loss_ftc', loss.item())
                    loss.backward()
                    self.opt.step()
