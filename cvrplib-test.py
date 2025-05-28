# coding=utf-8
# @Author : LJR
# @Time :  2025/1/28 15:21
# @Description :
import time

import numpy as np
import pickle
import os
import vrplib  # Assuming you have this module for reading VRP files
from tqdm import tqdm


class CVRPLibConverter:
    def __init__(self, cvrplib_data):
        """Initialize with CVRPLib data."""
        self.cvrplib_data = cvrplib_data

    def scale_coordinates(self, coords):
        """Scale all coordinates to be in the range [0, 1]."""
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        scaled_coords = (coords - min_coords) / (max_coords - min_coords)
        return scaled_coords

    def scale_demands(self, demands, old_capacity, new_capacity):
        """Scale demands to match the new vehicle capacity."""
        scale_factor = new_capacity / old_capacity
        scaled_demands = demands * scale_factor
        return scaled_demands

    def convert(self):
        """Convert CVRPLib dataset to HCVRP format."""
        old_capacity = self.cvrplib_data['capacity']  # Original vehicle capacity
        new_capacity = 35  # Set vehicle capacity to 35

        # Extract required data from CVRPLib data
        coords = self.cvrplib_data['node_coord']  # All coordinates including depot at index 0
        demands = self.cvrplib_data['demand'][1:]  # Demands, excluding depot

        # Apply transformations
        scaled_coords = self.scale_coordinates(coords)  # Scale all coordinates together
        scaled_demands = self.scale_demands(demands, old_capacity, new_capacity)

        # Prepare data using numpy arrays
        loc_array = scaled_coords[1:, :]  # Locations (excluding depot)
        depot_array = scaled_coords[0, :]  # Depot coordinates
        demand_array = scaled_demands.reshape(-1)  # Ensure demands are 1D

        # Add a batch dimension (batch_size=1)
        batch_size = 1
        depot_array = np.expand_dims(depot_array, axis=0)  # Shape becomes (1, 2) for depot
        loc_array = np.expand_dims(loc_array, axis=0)    # Shape becomes (1, N, 2) for locations
        demand_array = np.expand_dims(demand_array, axis=0)  # Shape becomes (1, N) for demands

        # Build dataset structure according to HCVRP format
        dataset = {
            'depot': depot_array,
            'loc': loc_array,
            'demand': demand_array,
            'capacity': np.array([[new_capacity]]),  # Capacity as 2D array with batch dimension
            'speed': np.array([[1]])  # Speed as 2D array with batch dimension
        }

        return dataset

    def save_as_pickle(self, output_filename):
        """Save converted dataset as a pickle file."""
        dataset = self.convert()
        with open(output_filename, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved as {output_filename}")
        return output_filename

def convert_all_vrp_to_pkl(input_directory, output_directory, model):
    """Convert all VRP files in the input directory to PKL format."""
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)


    # filenameList = ['data/hcvrp-lib\\Golden_1.vrp', 'data/hcvrp-lib\\Golden_13.vrp', 'data/hcvrp-lib\\Golden_14.vrp', 'data/hcvrp-lib\\Golden_15.vrp', 'data/hcvrp-lib\\Golden_16.vrp', 'data/hcvrp-lib\\Golden_17.vrp', 'data/hcvrp-lib\\Golden_18.vrp', 'data/hcvrp-lib\\Golden_2.vrp', 'data/hcvrp-lib\\Golden_9.vrp', 'data/hcvrp-lib\\Li_22.vrp', 'data/hcvrp-lib\\Li_25.vrp', 'data/hcvrp-lib\\Loggi-n401-k23.vrp', 'data/hcvrp-lib\\M-n101-k10.vrp', 'data/hcvrp-lib\\M-n121-k7.vrp', 'data/hcvrp-lib\\M-n151-k12.vrp', 'data/hcvrp-lib\\M-n200-k16.vrp', 'data/hcvrp-lib\\M-n200-k17.vrp', 'data/hcvrp-lib\\X-n101-k25.vrp', 'data/hcvrp-lib\\X-n106-k14.vrp', 'data/hcvrp-lib\\X-n110-k13.vrp', 'data/hcvrp-lib\\X-n115-k10.vrp', 'data/hcvrp-lib\\X-n120-k6.vrp', 'data/hcvrp-lib\\X-n125-k30.vrp', 'data/hcvrp-lib\\X-n129-k18.vrp', 'data/hcvrp-lib\\X-n134-k13.vrp', 'data/hcvrp-lib\\X-n139-k10.vrp', 'data/hcvrp-lib\\X-n143-k7.vrp', 'data/hcvrp-lib\\X-n148-k46.vrp', 'data/hcvrp-lib\\X-n153-k22.vrp', 'data/hcvrp-lib\\X-n157-k13.vrp', 'data/hcvrp-lib\\X-n162-k11.vrp', 'data/hcvrp-lib\\X-n167-k10.vrp', 'data/hcvrp-lib\\X-n172-k51.vrp', 'data/hcvrp-lib\\X-n176-k26.vrp', 'data/hcvrp-lib\\X-n181-k23.vrp', 'data/hcvrp-lib\\X-n186-k15.vrp', 'data/hcvrp-lib\\X-n190-k8.vrp', 'data/hcvrp-lib\\X-n195-k51.vrp', 'data/hcvrp-lib\\X-n200-k36.vrp', 'data/hcvrp-lib\\X-n204-k19.vrp', 'data/hcvrp-lib\\X-n209-k16.vrp', 'data/hcvrp-lib\\X-n223-k34.vrp', 'data/hcvrp-lib\\X-n228-k23.vrp', 'data/hcvrp-lib\\X-n233-k16.vrp', 'data/hcvrp-lib\\X-n237-k14.vrp', 'data/hcvrp-lib\\X-n242-k48.vrp', 'data/hcvrp-lib\\X-n247-k50.vrp', 'data/hcvrp-lib\\X-n251-k28.vrp', 'data/hcvrp-lib\\X-n256-k16.vrp', 'data/hcvrp-lib\\X-n261-k13.vrp', 'data/hcvrp-lib\\X-n266-k58.vrp', 'data/hcvrp-lib\\X-n270-k35.vrp', 'data/hcvrp-lib\\X-n275-k28.vrp', 'data/hcvrp-lib\\X-n280-k17.vrp', 'data/hcvrp-lib\\X-n289-k60.vrp', 'data/hcvrp-lib\\X-n294-k50.vrp', 'data/hcvrp-lib\\X-n298-k31.vrp', 'data/hcvrp-lib\\X-n303-k21.vrp', 'data/hcvrp-lib\\X-n313-k71.vrp', 'data/hcvrp-lib\\X-n322-k28.vrp', 'data/hcvrp-lib\\X-n327-k20.vrp', 'data/hcvrp-lib\\X-n336-k84.vrp', 'data/hcvrp-lib\\X-n344-k43.vrp', 'data/hcvrp-lib\\X-n351-k40.vrp', 'data/hcvrp-lib\\X-n359-k29.vrp', 'data/hcvrp-lib\\X-n376-k94.vrp', 'data/hcvrp-lib\\X-n384-k52.vrp', 'data/hcvrp-lib\\X-n393-k38.vrp', 'data/hcvrp-lib\\X-n401-k29.vrp', 'data/hcvrp-lib\\X-n420-k130.vrp', 'data/hcvrp-lib\\X-n429-k61.vrp', 'data/hcvrp-lib\\X-n439-k37.vrp', 'data/hcvrp-lib\\X-n449-k29.vrp', 'data/hcvrp-lib\\X-n469-k138.vrp', 'data/hcvrp-lib\\X-n480-k70.vrp', 'data/hcvrp-lib\\X-n491-k59.vrp', 'data/hcvrp-lib\\X-n502-k39.vrp', 'data/hcvrp-lib\\X-n524-k153.vrp', 'data/hcvrp-lib\\X-n536-k96.vrp', 'data/hcvrp-lib\\X-n573-k30.vrp', 'data/hcvrp-lib\\X-n586-k159.vrp', 'data/hcvrp-lib\\X-n599-k92.vrp', 'data/hcvrp-lib\\X-n613-k62.vrp', 'data/hcvrp-lib\\X-n627-k43.vrp', 'data/hcvrp-lib\\X-n641-k35.vrp', 'data/hcvrp-lib\\X-n655-k131.vrp', 'data/hcvrp-lib\\X-n685-k75.vrp', 'data/hcvrp-lib\\X-n733-k159.vrp', 'data/hcvrp-lib\\X-n749-k98.vrp', 'data/hcvrp-lib\\X-n819-k171.vrp', 'data/hcvrp-lib\\X-n876-k59.vrp', 'data/hcvrp-lib\\X-n916-k207.vrp']
    # print(len(filenameList))
    # Iterate through all files in the input directory
    opt_name = []
    opt_cost = []
    cost_list = []
    for filename in os.listdir(input_directory):
        if filename.endswith('.vrp'):  # Only process .vrp files
            vrp_file_path = os.path.join(input_directory, filename)

            # Read instance from the VRP file
            instance = vrplib.read_instance(vrp_file_path)

            # Create the converter and save the dataset as pickle
            converter = CVRPLibConverter(instance)
            output_filename = os.path.splitext(filename)[0] + '.pkl'  # Change .vrp to .pkl
            output_file_path = os.path.join(output_directory, output_filename)

            hcvrp_data = converter.save_as_pickle(output_file_path)
            dataset = model.problem.make_dataset(filename=hcvrp_data, num_samples=1, offset=0)

            sol_name = vrp_file_path[:-4] + '.sol'
            solution = vrplib.read_solution(sol_name)
            # cvrplib 的 cost
            best_cost = solution['cost']
            opt_cost.append(best_cost)
            edge_weight = instance['edge_weight']

            # 引入模型，产生解序列，乘以edge权重，得出与cost差距
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=1)

            for batch in tqdm(dataloader):
                batch = move_to(batch, device)
                with torch.no_grad():
                    # This returns (batch_size, iter_rep shape)
                    start = time.time()
                    sequences, costs_sig, veh_lists = model.sample_many(batch, batch_rep=1, iter_rep=1)
                    print(time.time()-start)
                    model_cost_sig = calculate_route_cost(sequences.cpu().numpy()[0], edge_weight)
                    print(model_cost_sig, best_cost)
                    sequences_2, costs, veh_lists = apply_symmetry_transformations_and_select_best(model, batch)
                    model_cost = calculate_route_cost(sequences_2.cpu().numpy()[0], edge_weight)
                    cost_list.append(model_cost)
                    gap = (model_cost - best_cost) / best_cost
                    print("best-cost is :{}, greedy-cost is :{}, model-cost-8 is :{}".format(best_cost, model_cost_sig, model_cost))

                    # if batch['loc'].shape[1] <= 200:
                    #     # results["100-200"].append((model_cost - best_cost) / best_cost)
                    #     # results["100-200"].append((model_cost_sig - best_cost) / best_cost)
                    #
                    #     results["100-200"].append(model_cost)
                    #     results_sig["100-200"].append(model_cost_sig)
                    #     results["100-200-best"].append(best_cost)
                    #     filenameList.append(vrp_file_path)
                    # elif 200 < batch['loc'].shape[1] <= 500:
                    #     # results["200-500"].append((model_cost - best_cost) / best_cost)
                    #     results["200-500"].append(model_cost)
                    #     results_sig["200-500"].append(model_cost_sig)
                    #     results["200-500-best"].append(best_cost)
                    #     filenameList.append(vrp_file_path)
                    # else:
                    #     filenameList.append(vrp_file_path)
                    #     results["500-1000"].append(model_cost)
                    #     results_sig["500-1000"].append(model_cost_sig)
                    #     results["500-1000-best"].append(best_cost)
    # import pandas as pd
    # pd.DataFrame(data={'Instance':opt_name, 'opt-cost':opt_cost, 'RANS-cost':cost_list}).to_csv("RANS.csv", index=False, encoding='gbk')
    # print("[100-200) average 88cost is :{}".format(np.mean(results["100-200"])))
    # print("[200-500) average 88cost is :{}".format(np.mean(results["200-500"])))
    # print("[500-1000) average 88cost is :{}".format(np.mean(results["500-1000"])))
    #
    # print("[100-200) average sig-cost is :{}".format(np.mean(results_sig["100-200"])))
    # print("[200-500) average sig-cost is :{}".format(np.mean(results_sig["200-500"])))
    # print("[500-1000) average sig-cost is :{}".format(np.mean(results_sig["500-1000"])))
    #
    # print("[100-200) average best-cost is :{}".format(np.mean(results["100-200-best"])))
    # print("[200-500) average best-cost is :{}".format(np.mean(results["200-500-best"])))
    # print("[500-1000) average best-cost is :{}".format(np.mean(results["500-1000-best"])))
    #
    # print("[100-200) min cost is :{}".format(np.min(results["100-200"])))
    # print("[200-500) min cost is :{}".format(np.min(results["200-500"])))
    # print("[500-1000) min cost is :{}".format(np.min(results["500-1000"])))
    #
    # print("[100-200) min cost is :{}".format(np.max(results["100-200"])))
    # print("[200-500) min cost is :{}".format(np.max(results["200-500"])))
    # print("[500-1000) min cost is :{}".format(np.max(results["500-1000"])))
    #
    # print(results["100-200"])
    # print(results["200-500"])
    # print(results["500-1000"])
    #
    # print(filenameList)
    return None

def calculate_route_cost(route, edge_weights):
#     """
#     计算给定路线的成本
#     :param route: 包含服务节点的列表
#     :param edge_weights: 边权重矩阵
#     :return: 路径成本
#     """
    # 在开头和结尾添加仓库节点
    full_route = [0] + route + [0]  # 0 表示仓库在节点数组中的索引
    total_cost = 0.0

    # 计算路径成本
    for i in range(len(full_route) - 1):
        total_cost += edge_weights[full_route[i], full_route[i + 1]]

    return total_cost


def apply_symmetry_transformations_and_select_best( model, batch):

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
    sequences, costs, veh_lists = model.sample_many(batch, batch_rep=1, iter_rep=1)  # Costs shape: (batch*8,)

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

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4,dat5,dat6,dat7,dat8), dim=0)
    # aug_xy_data = torch.cat((dat1, dat2, dat3, dat4), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data



if __name__ == "__main__":
    input_dir = "data/hcvrp-lib"  # Replace with your input folder containing .vrp files
    output_dir = "data/CVRPData_2"  # Desired output directory for .pkl files
    output_sl_dir = "data/CVRPDataSL_2"  # Desired output directory for .pkl files
    from utils import load_model, move_to
    import torch

    model_name = 'outputs/hcvrp_v3_40/hcvrp_v3_40_rollout_LJR-Model/RALM.pt'
    model, _ = load_model(model_name, 'mix-max')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    model.eval()
    model.set_decode_type("greedy",temp=1)

    convert_all_vrp_to_pkl(input_dir, output_dir, model)

