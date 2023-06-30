

import importlib
import os
import sys
conf_path = os.getcwd()
sys.path.append(conf_path)
import socket
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_dataset
from models import get_model
from utils.training import train
import torch
import setproctitle

import uuid
import datetime


def parse_args():
    parser = ArgumentParser(description='Base parse', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()

    return args


def main(args=None):

    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    
    # job number 
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    train(model, dataset, args)

if __name__ == '__main__':
    main()
