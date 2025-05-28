# used after model is completely trained, and test for results

import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
import warnings

mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, veh_lists, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx + 1, ...], cost[idx:idx + 1, ...], veh_lists[idx:idx + 1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result], [
        veh_lists[i] if i >= 0 else None for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model, opts.obj)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model, opts.obj)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, veh_lists, durations = zip(*results)  # Not really costs since they should be negative
    print("Test DataSet is .{}".format(dataset_path))
    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))
    print("*"*60)


    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            softmax_temp, opts.offset, opts.offset + len(costs), ext
        ))
    else:
        out_file = opts.o

    assert opts.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)

    return np.mean(costs), tours, np.mean(durations) / parallelism


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):
    # print('data', dataset[0])
    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)
        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs, veh_lists = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                # reg_loc = batch['loc']
                # sequences, costs, veh_lists = apply_symmetry_transformations_and_select_best(reg_loc, model, batch,batch_rep, iter_rep)
                print('cost', costs)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
            veh_lists = [None] * batch_size
        else:
            sequences, costs, veh_lists = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(), veh_lists.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )

        duration = time.time() - start
        for seq, cost, veh_list in zip(sequences, costs, veh_lists):
            if model.problem.NAME in ("hcvrp"):
                seq = seq.tolist()  # No need to trim as all are same length
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, seq, veh_list, duration))

    return results


def apply_symmetry_transformations_and_select_best(reg_loc, model, batch, batch_rep, iter_rep):

    aug_loc = augment_xy_data_by_8_fold(batch['loc'])
    aug_depot = augment_xy_data_by_8_fold(batch['depot'].unsqueeze(1))

    # 实例增强
    batch['loc'] = aug_loc
    batch['depot'] = aug_depot.squeeze(1)
    # 货物量，速度，需求 全部增强
    batch['demand'] = batch['demand'].repeat(8, 1)
    batch['capacity'] = batch['capacity'].repeat(8, 1)
    batch['speed'] = batch['speed'].repeat(8, 1)

    # 调用模型批量计算所有增强的结果
    sequences, costs, veh_lists = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)  # Costs shape: (batch*8,)

    # 将 costs 转换成原始 batch 维度 (batch, 8)
    costs = costs.view(-1, 8)  # Reshape to (batch, 8)

    # 找到每个 batch 中最小的 cost 及对应的索引
    min_costs, min_indices = torch.min(costs, dim=1)  # (batch,), (batch,)

    # 根据最小 cost 的索引提取 sequences 和 veh_lists
    best_sequences = sequences[min_indices]# 按增强索引选择序列
    best_veh_lists = veh_lists[min_indices] # 按增强索引选择车辆列表

    return best_sequences, min_costs, best_veh_lists

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data



if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", default="data/hcvrp/hcvrp_v3_100_seed26410.pkl", help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1000,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results_with_loss_rebuttal_k1', help="Name of results directory")
    parser.add_argument('--obj', default=['min-max', 'min-sum'])
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    widths = opts.width if opts.width is not None else [0]

    # 指定要获取文件名的目录
    directory = './data/hcvrp'
    # 获取目录下的所有文件名
    file_names = os.listdir(directory)
    # 过滤出文件名（排除子目录）
    file_names = [f for f in file_names if os.path.isfile(os.path.join(directory, f))]
    # 打印文件名
    cost_list = []
    average_time_list = []
    dataset_name_list = []
    for width in widths:
        for dataset_path in file_names:
            print(dataset_path)
            cost, _, average_time = eval_dataset("./data/hcvrp/" + dataset_path, width, opts.softmax_temperature, opts)
            cost_list.append(cost)
            average_time_list.append(average_time)
            import re
            result = re.match(r"([^-]+_[^-]+)", dataset_path).group(1)  # 匹配前两个 `_` 之间的内容
            dataset_name_list.append(result)

    # 生成 dataframe
    import pandas as pd

    res = pd.DataFrame(data={"dataset":dataset_name_list, "cost":cost_list, "time":average_time_list})
    res.to_csv(opts.model + "_res_rebuttal.csv", index=False)
    print("success!")