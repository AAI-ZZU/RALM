import math

import torch

class HcvrpEnv:
    def __init__(self,input,scale=(1,40,1)):
        '''
        :param input:
        input:{
            'loc':  batch_size, graph_size, 2
            'demand': batch_size, graph_size
            'depot': batch_size, 2
            'capacity': batch_size, vehicle_num
            'speed': batch_size, vehicle_num
        }
        :param scale: used to output normalized state (coords,demand,speed)
        '''

        self.device = input['loc'].device
        self.batch_size = input['loc'].shape[0]
        self.bs_index = torch.arange(self.batch_size,device = self.device)
        self.step = 0
        self.scale_coords,self.scale_demand,self.scale_speed = scale
        self.initial_node_state(input['loc'],input['demand'],input['depot'])
        self.initial_veh_state(input['capacity'], input['speed'])
        # 车辆路径掩码
        self.vehicle_node_mask = torch.zeros(
            (self.batch_size, self.veh_num, self.N),
            dtype=torch.bool,
            device=self.device
        )

    def initial_node_state(self,loc,demand,depot):
        '''
        :param loc:  customer coordinates [batch_size, graph_size,2]
        :param demand: customer demands [batch_size, graph_size]
        :param depot: depot coordinates [batch_size, 2]
        :return:
        '''
        assert loc.shape[:2] == demand.shape, "The custumer's loc and demand shape do not match"
        self.customer_num = loc.shape[1]
        self.N = loc.shape[1]+1 # Let N represent the graph size
        self.coords = torch.cat([depot.unsqueeze(1),
                                 loc],dim=1) # batch_size, N, 2
        self.demand = torch.cat([torch.zeros_like(demand[:,[0]]),
                                 demand],dim=1) # batch_size, N
        self.visited = torch.zeros_like(self.demand).bool() # batch_size, N
        self.visited[:,0] = True # start from depot, so depot is visited

    def all_finished(self):
        '''
        :return: Are all tasks finished?
        '''
        return self.visited.all()

    def finished(self):
        '''
        :return: [bs],true or false, is each task finished?
        '''
        return self.visited.all(-1)

    def get_all_node_state(self):
        '''
        :return: [bs,N+1,3], get node initial features
        '''
        return torch.cat([self.coords/self.scale_coords,
                          self.demand.unsqueeze(-1)/self.scale_demand],dim = -1) # batch_size, N, 3

    def initial_veh_state(self,capacity,speed):
        '''
        :param capacity:  batch_size, veh_num
        :param speed: batch_size, veh_num
        :return
        '''
        assert capacity.size() == speed.size(), "The vehicle's speed and capacity shape do not match"
        self.veh_capacity = capacity
        self.veh_speed = speed
        self.veh_num = capacity.shape[1]
        self.veh_time = torch.zeros_like(capacity)  # batch_size, veh_num
        self.veh_cur_node = torch.zeros_like(capacity).long() # batch_size, veh_num
        self.veh_used_capacity = torch.zeros_like(capacity)
        # a util vector
        self.veh_index = torch.arange(self.veh_num, device=self.device)

    def min_max_norm(self,data):
        '''
        deprecated
        :param data:
        :return:
        '''
        # bs，M
        min_data = data.min(-1,keepdim=True)[0]
        max_data = data.max(-1, keepdim=True)[0]
        return (data-min_data)/(max_data-min_data)
    def get_all_veh_state(self):
        '''
        :return: [bs,M,4]
        # time，capacity，usage capacity，speed
        '''
        return torch.cat([
                          self.veh_time.unsqueeze(-1),
                          self.veh_capacity.unsqueeze(-1)/self.scale_demand,
            self.veh_used_capacity.unsqueeze(-1) / self.scale_demand,
            # (self.veh_used_capacity.unsqueeze(-1) / self.veh_capacity.unsqueeze(-1)) / self.scale_demand,
                          self.veh_speed.unsqueeze(-1)/self.scale_speed,
        ],dim=-1)

    def get_veh_state(self,veh):
        # deprecated
        '''
        :param veh: veh_index，batch_size
        :return:
        '''
        all_veh_state = self.get_all_veh_state() # bs,veh_num,4
        return all_veh_state[self.bs_index,veh] # bs,4


    def action_is_legal(self,veh,next_node):
        # deprecated
        return self.demand[self.bs_index, next_node] <= (self.veh_capacity - self.veh_used_capacity)[self.bs_index, veh]

    def update(self, veh, next_node):
        '''
        input action tuple and update the env
        :param veh: [batch_size,]
        :param next_node: [batch_size,]
        :return:
        '''
        # select node must be unvisited,except depot
        assert not self.visited[self.bs_index, next_node][
            next_node != 0].any(), "Wrong solution: node has been selected !"
        # Note that demand<=remaining_capacity==capacity-usage_capacity
        assert (self.demand[self.bs_index, next_node] <=
                (self.veh_capacity - self.veh_used_capacity)[
                    self.bs_index, veh]).all(), "Wrong solution: the remaining capacity of the vehicle cannot satisfy the node !"

        # update vehicle time，
        last_node = self.veh_cur_node[self.bs_index, veh]
        old_coords, new_coords = self.coords[self.bs_index, last_node], self.coords[self.bs_index, next_node]
        length = torch.norm(new_coords - old_coords, p=2, dim=1)
        time_add = length / self.veh_speed[self.bs_index, veh]
        self.veh_time[self.bs_index, veh] += time_add

        # update the used_capacity
        new_veh_used_capacity = self.veh_used_capacity[self.bs_index, veh] + self.demand[self.bs_index, next_node]
        new_veh_used_capacity[next_node == 0] = 0  # 回到仓库后装满车辆
        self.veh_used_capacity[self.bs_index, veh] = new_veh_used_capacity

        # update the node index where the vehicle stands
        self.veh_cur_node[self.bs_index, veh] = next_node
        self.step += 1
        # print(self.step)

        # update visited vector
        self.visited[self.bs_index, next_node] = True


    def all_go_depot(self):
            '''
            All vehicle go back the depot
            :return:
            '''
            veh_list = torch.arange(self.veh_num,device = self.device)
            depot = torch.zeros_like(self.bs_index)
            for i in veh_list:
                self.update(i.expand(self.batch_size),depot)

    def get_cost(self,obj):
        self.all_go_depot()
        # if obj=='min-max':
        #     return self.veh_time.max(-1)[0]
        # elif obj=='min-sum':
        #     return self.veh_time.sum(-1)
        return self.veh_time.sum(-1)

        # return self.veh_time.sum(-1)

    def get_cost_cvrp(self,obj):
        self.all_go_depot()
        return self.veh_time.sum(-1) / self.veh_num

    def get_action_mask(self):
        # cannot select a visited node except the depot
        visited_mask = self.visited.clone() # bs,N+1
        visited_mask[:,0]=False
        # Here, clone() is important for avoiding the bug from expand()
        visited_mask = visited_mask.unsqueeze(1).expand(self.batch_size, self.veh_num, self.N).clone() # bs,M,N+1
        # Vehicle cannot stay in place to avoid visiting the depot twice,
        # otherwise an infinite loop will easily occur
        visited_mask[self.bs_index.unsqueeze(-1),self.veh_index.unsqueeze(0),self.veh_cur_node]=True
        # capacity constraints
        demand_mask = (self.veh_capacity - self.veh_used_capacity).unsqueeze(-1) < self.demand.unsqueeze(1) # bs,M,N+1
        mask = visited_mask | demand_mask
        # Special setting for batch processing,
        # because the finished task will have a full mask and raise an error
        mask[self.finished(),0,0]=False
        return mask

    @staticmethod
    def caculate_cost(input,solution,obj):
        '''
        :param input: equal to __init__
        :param solution: (veh,next_node): [total_step, batch_size],[total_step, batch_size]
        :param obj: 'min-max' or 'min-sum'
        :return: cost : batch_size
        '''

        env = HcvrpEnv(input)
        for veh,next_node in zip(*solution):
            env.update(veh,next_node)
        return env.get_cost(obj)


    """
    计算车辆到节点的时间
    """
    def get_veh_node_travel_time(self):
        # 获取当前车辆的位置坐标
        veh_cur_coords = self.coords[self.bs_index.unsqueeze(-1), self.veh_cur_node]  # [batch_size, veh_num, 2]

        # 计算每辆车到所有节点的距离
        distance_matrix = torch.norm(veh_cur_coords.unsqueeze(2) - self.coords.unsqueeze(1), p=2,
                                     dim=-1)  # [batch_size, veh_num, N]
        # 获取车辆的速度
        v_speed = self.veh_speed  # [batch_size, veh_num]

        # 将速度的维度扩展到 [batch_size, veh_num, 1] 以便进行广播
        v_speed_expanded = v_speed.unsqueeze(-1)  # [batch_size, veh_num, 1]

        # 计算距离除以速度
        time = distance_matrix / v_speed_expanded  # [batch_size, veh_num, N]

        return time


    # 每一个车辆 当前负载 与 满载负载 的比值
    def get_vehicle_load_ratio(self):
        """
        计算每辆车当前的装载率（已用容量 / 总容量）
        Returns:
        - load_ratio: [batch_size, vehicle_num] 每辆车的装载率
        """
        # 计算装载率：已用容量 / 总容量
        load_ratio = (self.veh_capacity - self.veh_used_capacity) / self.veh_capacity

        return load_ratio

    # todo
    # 注意力矩阵的掩码规则，（bs, veh_nums, node_nums）
    # 注意力掩码 分两种情况：
        # 1， 如果当前车辆货物剩余量 / 满载剩余量<0.97, 那么只有离它最近50个节点才可用有ATTENTION分数，其他的用很小很小的数来代替（注意可用节点数与50的大小，要选择更小的作为实际值）
        # 2,  如果当前车辆货物剩余量 / 满载剩余量>=0.97, 那么那么没有限制
    def generate_attention_mask(self, k=80): # k=100:12.412 k=80:12.412  k=60:12.413  k=40:12.414
        """):
        完全矩阵化注意力掩码生成

        Args:
        - k (int近邻节点数限制，默认为50

        Returns:
        - attention_mask (torch.Tensor): 形状为 [batch_size, veh_num, node_nums]
        """
        # 计算装载率
        load_ratio = self.get_vehicle_load_ratio()  # [batch_size, vehicle_num]

        # 获取距离矩阵
        distance_matrix = self.get_veh_node_travel_time()  # [batch_size, veh_num, N]

        # 初始化注意力掩码，标记已访问节点
        visited_mask = self.get_action_mask()

        # 将已访问节点的距离设为正无穷
        distance_matrix[visited_mask] = float(10000000.0)

        # 创建低装载率掩码
        low_load_mask = load_ratio <= 0.98

        if self.customer_num >500:
            k = 1

        k_real = min(k, self.customer_num)
        # 对低装载率车辆，选择最近的k个未访问节点
        _, topk_indices = torch.topk(distance_matrix, k_real, dim=-1, largest=False)

        # 创建掩码基础
        neighbor_mask = torch.ones_like(distance_matrix, dtype=torch.bool)
        # 为低装载率车辆的近邻节点创建掩码
        bs, veh_num, node_num = distance_matrix.shape
        batch_indices = torch.arange(bs, device=distance_matrix.device).unsqueeze(-1).unsqueeze(-1)
        veh_indices = torch.arange(veh_num, device=distance_matrix.device).unsqueeze(0).unsqueeze(-1)
        # 克隆原始mask
        updated_mask = neighbor_mask.clone()
        # 使用高维索引将对应位置设为False
        updated_mask[batch_indices, veh_indices, topk_indices] = False

        # 高装载率车辆设为全False
        updated_mask[~low_load_mask,:] = False
        updated_mask = updated_mask | visited_mask
        return updated_mask

    def get_k_nearest_neighbors(self, k=30):
        """
        获取所有节点的K近邻节点下标

        Args:
            k (int): 要获取的最近邻节点数量，默认为10

        Returns:
            torch.Tensor: 形状为 (batch_size, node_nums, k) 的张量，
                          每个节点的K个最近邻节点的索引
        """
        # 获取节点间距离矩阵
        distance_matrix = self.get_node_dist()  # (batch_size, node_nums, node_nums)

        # 将对角线设置为很大的值，避免选择自身作为最近邻
        batch_size, node_nums = distance_matrix.shape[:2]
        eye_mask = torch.eye(node_nums, device=distance_matrix.device).bool()
        distance_matrix = distance_matrix.masked_fill(eye_mask, float('inf'))

        # 使用topk获取最近的k个节点的索引
        # 注意：这里使用 largest=False 表示获取最小的k个值的索引
        _, k_nearest_indices = torch.topk(distance_matrix, k=min(k, node_nums - 1), dim=-1, largest=False)

        return k_nearest_indices


    def get_node_dist(self):
        # 计算 coords 之间的距离
        # self.coords 的形状是 (batch_size, nums, 2)
        # 使用 unsqueeze 来获得 (batch_size, nums, 1, 2) 和 (batch_size, 1, nums, 2)
        diff = self.coords.unsqueeze(2) - self.coords.unsqueeze(1)  # (batch_size, nums, nums, 2)
        # 计算最后一个维度的 L2 范数
        distance_matrix = torch.norm(diff, p=2, dim=-1)  # (batch_size, nums, nums)
        return distance_matrix


    def get_node_damand(self):

        return self.demand.unsqueeze(-1)/self.scale_demand

    def generate_local_neighbor_mask(self, k=30):
        """
        生成局部邻接关系的mask

        Args:
            k (int): 每个节点要保留的最近邻节点数量

        Returns:
            torch.Tensor: shape (batch_size, node_nums, node_nums) 的布尔类型tensor
                         False表示保留的k个最近邻节点，True表示需要mask的节点
        """
        # 获取节点间距离矩阵
        distance_matrix = self.get_node_dist()  # (batch_size, node_nums, node_nums)

        # 将对角线设置为很大的值，避免选择自身作为最近邻
        batch_size, node_nums = distance_matrix.shape[:2]
        eye_mask = torch.eye(node_nums, device=distance_matrix.device).bool()
        distance_matrix = distance_matrix.masked_fill(eye_mask, float('inf'))

        # 获取每个节点最近的k个邻居的索引
        # 使用topk获取最小的k个距离的索引
        _, topk_indices = torch.topk(-distance_matrix, k=min(k, node_nums - 1), dim=-1)

        # 创建初始mask，全部设为True
        mask = torch.ones_like(distance_matrix, dtype=torch.bool)

        # 使用高级索引将k个最近邻设置为False
        batch_indices = torch.arange(batch_size, device=distance_matrix.device).unsqueeze(-1).unsqueeze(-1)
        node_indices = torch.arange(node_nums, device=distance_matrix.device).unsqueeze(0).unsqueeze(-1)
        mask[batch_indices, node_indices, topk_indices] = False

        return mask

    def get_vehicle_node_mask(self):
        """
        获取每辆车走过的节点掩码

        Returns:
        - vehicle_node_mask (torch.Tensor): 形状为 [batch_size, vehicle_num, node_num]
            被访问过的节点为True
        """

        # 对于每辆车，将其当前位置标记为True
        batch_indices = self.bs_index.unsqueeze(-1)
        vehicle_indices = torch.arange(self.veh_num, device=self.device).unsqueeze(0)
        self.vehicle_node_mask[batch_indices, vehicle_indices, self.veh_cur_node] = True

        return self.vehicle_node_mask