import argparse
import os
import sys
import logging
from src.config import GWNConfig as cf

def get_logger(log_dir, name, log_filename, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print('Log directory:', log_dir)
    
    return logger

def get_config():
    parser = get_public_config()
    args = parser.parse_args()
    addition = get_model_args(args.model)
    for key, value in addition.items():
        setattr(args, key, value)
    log_dir = './experiments/{}/{}/'.format(args.model, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    args.logger = logger
    args.log_dir = log_dir
    return args

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=cf.device)
    parser.add_argument('--dataset', type=str, default=cf.dataset)
    parser.add_argument('--years', type=str, default=cf.years)
    parser.add_argument('--model', type=str, default=cf.model)
    parser.add_argument('--seed', type=int, default=cf.seed)
    parser.add_argument('--batch_size', type=int, default=cf.batch_size)
    parser.add_argument('--seq_length', type=int, default=cf.seq_length)
    parser.add_argument('--horizon', type=int, default=cf.horizon)
    parser.add_argument('--input_dim', type=int, default=cf.input_dim)
    parser.add_argument('--output_dim', type=int, default=cf.output_dim)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--lr_step', type=int, default=20)
    parser.add_argument('--lr_decay', type=float, default=1e-8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=int, default=5)

    parser.add_argument('--nor_adj', action='store_true')
    parser.add_argument('--use_mixed_proj', action='store_true')
    parser.add_argument('--save', type=str, default="test.pt")
    #parser.add_argument('--adj_type', type=str, default='doubletransition')

    parser.add_argument('--ssm', action='store_true')
    parser.add_argument('--state_size', type=int, default=64)
    parser.add_argument('--in_steps', type=int, default=12)
    parser.add_argument('--out_steps', type=int, default=12)
    parser.add_argument('--steps_per_day', type=int, default=288)
    parser.add_argument('--input_embedding_dim', type=int, default=12)
    parser.add_argument('--tod_embedding_dim', type=int, default=12)
    parser.add_argument('--dow_embedding_dim', type=int, default=12)
    parser.add_argument('--spatial_embedding_dim', type=int, default=0)
    parser.add_argument('--adaptive_embedding_dim', type=int, default=40)
    parser.add_argument('--feed_forward_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=12)
    return parser

def get_model_args(model_name):
    model_args = {}
    if model_name == 'xxx':
        pass
    return model_args
