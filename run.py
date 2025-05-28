#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
# from tensorboard_logger import Logger as TbLogger
from tqdm import tqdm

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
#from nets.attention_model_minsum import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
import warnings
import copy
# from problems.vrp import CVRP
from problems import HCVRP

def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    # tb_logger = None
    # if not opts.no_tensorboard:
    #     tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_v{}_c{}".format(opts.problem,opts.veh_num,opts.graph_size), opts.run_name))

    # save model to outputs dir
    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)
    # problem = HCVRP(opts.graph_size,opts.veh_num,opts.obj)

    # Load data from load_path
    # if u have run the model before, u can continue from resume path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        opts.obj,
        problem,
        n_heads=opts.n_heads,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    # multi-gpu
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})


    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
    )

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    big_batch_size = 100
    small_batch_size = 24

    big_nums = 2500
    small_nums = 128
    """创建包含多种规模的验证集"""
    configs = [
    
        # {'size': 300, 'veh_num': 3, 'batch_size': small_batch_size, 'nums': small_nums},
        # {'size': 500, 'veh_num': 3, 'batch_size': small_batch_size, 'nums': small_nums},
        {'size': 1000, 'veh_num': 3, 'batch_size': small_batch_size, 'nums': small_nums},
        
        # {'size': 300, 'veh_num': 5, 'batch_size': small_batch_size, 'nums': small_nums},
        # {'size': 500, 'veh_num': 5, 'batch_size': small_batch_size, 'nums': small_nums},
        # {'size': 1000, 'veh_num': 5, 'batch_size': small_batch_size, 'nums': small_nums},

        # {'size': 300, 'veh_num': 7, 'batch_size': small_batch_size, 'nums': small_nums},
        # {'size': 500, 'veh_num': 7, 'batch_size': small_batch_size, 'nums': small_nums},
        # {'size': 1000, 'veh_num': 7, 'batch_size': small_batch_size, 'nums': small_nums}
    ]
    # 为每种规模创建验证数据，然后合并
    val_datasets = []
    for config in tqdm(configs):
        dataset = HCVRP.make_dataset(
            size=config['size'],
            veh_num=config['veh_num'],
            num_samples=config['nums']
        )
        val_datasets.append(dataset)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        # baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        loss = validate(model, val_datasets, opts)
        print('Validation overall avg_cost: {}'.format(loss))
    else:
        best_avg_loss = 1000
        early_times = 5
        best_model = None
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            avg_loss = train_epoch(
                model,
                optimizer,
                lr_scheduler,
                epoch,
                val_datasets,
                problem,
                None,
                opts
            )
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                early_times = 5
                print('Saving best model and state...')
                best_model = copy.deepcopy(model)
            else:
                early_times = early_times -1
            if early_times == -1:
                break
    print('Saving best model and state...')
    torch.save(
        {
            'model': get_inner_model(best_model).state_dict(),
            'optimizer': optimizer.state_dict(),
            # rng_state is the state of random generator
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
        },
        # save state of runned model in outputs
        os.path.join(opts.save_dir, 'best_model.pt')
    )


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(get_options())
