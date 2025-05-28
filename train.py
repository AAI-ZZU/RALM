import os
import time
import torch
import math


from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    # multi batch
    avg_cost = 0
    count = 0
    for data in tqdm(dataset):
        count += 1
        cost_data = rollout1(model, data, opts)
        avg_cost += cost_data.mean()
    print('Validation overall avg_cost: {}'.format(avg_cost/count))
    return avg_cost

def rollout1(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        # do not need backpropogation
        with torch.no_grad():
            cost, _, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    # tqdm is a function to show the progress bar
    return torch.cat([
        eval_model_bat(bat)
        for bat
        in DataLoader(dataset, batch_size=64)
    ], 0)

def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        # do not need backpropogation
        with torch.no_grad():
            cost, _, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    # tqdm is a function to show the progress bar
    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)



def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    # if epoch == 1:
    #     avg_reward = validate(model, val_dataset, opts)
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch)

    # if not opts.no_tensorboard:  # need tensorboard
    #     tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    print("***********Training data is being prepared***********")

    # big_batch_size = 128
    # small_batch_size = 64

    # big_nums = 51200
    # small_nums = 25600
    # """创建包含多种规模的验证集"""
    # configs = [
    
    #     {'size': 40, 'veh_num': 3, 'batch_size': big_batch_size, 'nums': big_nums},
    #     {'size': 80, 'veh_num': 3, 'batch_size': big_batch_size, 'nums': big_nums},
    #     {'size': 120, 'veh_num': 3, 'batch_size': big_batch_size, 'nums': big_nums},
    #     {'size': 180, 'veh_num': 3, 'batch_size': small_batch_size, 'nums': small_nums},

    #     {'size': 40, 'veh_num': 5, 'batch_size': big_batch_size, 'nums': big_nums},
    #     {'size': 80, 'veh_num': 5, 'batch_size':big_batch_size, 'nums': big_nums},
    #     {'size': 120, 'veh_num': 5, 'batch_size':small_batch_size, 'nums': small_nums},

    #     {'size': 40, 'veh_num': 7, 'batch_size': big_batch_size, 'nums': big_nums},
    #     {'size': 80, 'veh_num': 7, 'batch_size': big_batch_size, 'nums': big_nums},
    #     {'size': 140, 'veh_num': 7, 'batch_size': small_batch_size, 'nums': small_nums}
        
    # ]


    big_batch_size = 64
    small_batch_size = 48

    big_nums = 12800
    small_nums =6400
    """创建包含多种规模的验证集"""
    configs = [
        {'size': 180, 'veh_num': 3, 'batch_size': small_batch_size, 'nums': small_nums},
        {'size': 240, 'veh_num': 3, 'batch_size': small_batch_size, 'nums': small_nums},

        {'size': 180, 'veh_num': 5, 'batch_size':small_batch_size, 'nums': small_nums},
        {'size': 250, 'veh_num': 5, 'batch_size':small_batch_size, 'nums': small_nums},
        
        {'size': 140, 'veh_num': 7, 'batch_size': small_batch_size, 'nums': small_nums},
        {'size': 210, 'veh_num': 7, 'batch_size': small_batch_size, 'nums': small_nums}
        
    ]

    # 为每种规模创建验证数据，然后合并
    # 为每种规模创建数据加载器
    data_loaders = []
    for config in tqdm(configs):
        dataset = problem.make_dataset(
            size=config['size'],
            veh_num=config['veh_num'],
            num_samples=config['nums']
        )
        data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=0)
        data_loaders.append(data_loader)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    # 在训练开始前，打开日志文件
    log_file_path = os.path.join(opts.save_dir, 'training_log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 确定数据加载器的总轮数
    total_iterations = len(data_loaders[0])
    
    batch_id=0
    all_loss=0
    all_cost =0
    all_d_loss =0
    all_grad_norms = 0
    # 迭代每一轮
    for iteration in tqdm(range(total_iterations)):
        # 从每个数据加载器获取一个batch
        for dl in data_loaders:
            batch = next(iter(dl))
            loss, cost, d_loss = train_batch(model,optimizer,epoch,batch_id,step,batch,tb_logger,opts,log_file_path,iteration)
            batch_id += 1
            all_loss += loss.item()
            all_cost += cost.item()
            all_d_loss += d_loss.item()
            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()
            optimizer.step()
            grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
            all_grad_norms += grad_norms[0][0].item()

        if step % 5 == 0:
            log_message = "epoch:{}, grad_norms:{}, avg_cost:{}, avg_loss:{}, avg_d_loss:{}\n".format(epoch, all_grad_norms/batch_id, all_cost/batch_id, all_loss/batch_id, all_d_loss/batch_id)
            # 同时打印到控制台和写入文件
            print(log_message, end='')
            with open(log_file_path, 'a') as log_file:
                log_file.write(log_message + "\n")
            batch_id = 0
            all_loss = 0
            all_cost = 0
            all_d_loss = 0
            all_grad_norms = 0
        step += 1
    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # save results every checkpoint_epoches, saving memory
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                # rng_state is the state of random generator
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            # save state of runned model in outputs
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)
    return avg_reward


def apply_rotation(coordinates, angle):
    """
    对坐标应用旋转变换，并保证结果在指定位置
    Args:
        coordinates: [..., 2] tensor，原始坐标
        angle: 旋转角度（弧度）
    Returns:
        变换后的坐标
    """
    # 首先将坐标移到原点
    centered_coordinates = coordinates - 0.5

    x, y = centered_coordinates[..., 0], centered_coordinates[..., 1]
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    # 应用旋转
    new_x = cos_angle * x - sin_angle * y
    new_y = sin_angle * x + cos_angle * y

    # 移回(0.5, 0.5)
    final_x = new_x + 0.5
    final_y = new_y + 0.5

    # 确保坐标在 [0, 1] 范围内
    final_x = final_x % 1.0
    final_y = final_y % 1.0

    return torch.stack([final_x, final_y], dim=-1)

def symmetry_augment_batch(batch, device, num_rotations):
    """
    对batch数据进行旋转增强
    Args:
        batch: 原始batch字典
        device: torch设备
        num_rotations: 旋转次数
    Returns:
        augmented_batch: 增强后的batch字典
    """
    augmented_batch = {
        'loc': [],
        'depot': [],
        'demand': [],
        'capacity': [],
        'speed': []
    }

    for i in range(num_rotations):
        # 计算旋转角度
        angle = torch.tensor(2 * torch.pi * i / num_rotations)  # 将360度分成num_rotations
        aug_loc = apply_rotation(batch['loc'].to(device), angle)
        aug_depot = apply_rotation(batch['depot'].to(device), angle)

        augmented_batch['loc'].append(aug_loc)
        augmented_batch['depot'].append(aug_depot)
        # 复制其他不需要变换的特征
        augmented_batch['demand'].append(batch['demand'].to(device))
        augmented_batch['capacity'].append(batch['capacity'].to(device))
        augmented_batch['speed'].append(batch['speed'].to(device))

    for k in augmented_batch:
        augmented_batch[k] = torch.cat(augmented_batch[k], dim=0)

    return augmented_batch

def train_batch(model,
                optimizer,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts,
                log_file_path,
                iteration
                ):
    # 对batch进行对称性增强
    augmented_batch = symmetry_augment_batch(batch, opts.device, num_rotations=opts.aug_num)
    x = move_to(augmented_batch, opts.device)

    # 前向传播，获取增强后的所有实例的结果
    cost, log_likelihood, d_loss = model(x)  # [batch_size * 8]
    log_likelihood = log_likelihood.reshape(opts.aug_num, -1)
    # 计算每组对称实例的平均cost作为基线
    cost = cost.view(opts.aug_num, -1).permute(1, 0)
    log_likelihood = log_likelihood.view(opts.aug_num, -1).permute(1, 0)

    # 计算强化学习损失
    advantage = cost - cost.mean(dim=1).view(-1,1)
    # 为了让模型学习到更好的特征，应该突出最好的那个几个实例结果！！todo
    loss_common = ((advantage) * log_likelihood).mean()
    re_loss = (0.08 * d_loss).mean()
    loss = loss_common + re_loss
    return loss, cost.mean(), d_loss.mean()