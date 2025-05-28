import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy

from problems import HCVRP
from train import rollout, get_inner_model

class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, ):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):

        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.n_epochs)
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c): # x is data and c is cost in actor network

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):

    def __init__(self, model, problem, opts, epoch=0):
        super(Baseline, self).__init__()

        self.problem = problem
        self.opts = opts
        self.epoch_count = 10

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset

        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (dataset[0] if self.problem.NAME == 'tsp' else dataset[0]['loc']).size(0) != self.opts.graph_size:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            # """创建包含多种规模的验证集"""
            # configs = [
            #     {'size': 40, 'veh_num': 3},
            #     {'size': 60, 'veh_num': 3},
            #     {'size': 80, 'veh_num': 3},
            #     {'size': 40, 'veh_num': 5},
            #     {'size': 60, 'veh_num': 5},
            #     {'size': 80, 'veh_num': 5},
            #     {'size': 40, 'veh_num': 7},
            #     {'size': 60, 'veh_num': 7},
            #     {'size': 80, 'veh_num': 7}
            # ]
            #
            # # 为每种规模创建验证数据，然后合并
            # val_datasets = []
            # for config in configs:
            #     dataset = HCVRP.make_dataset(
            #         size=config['size'],
            #         veh_num=config['veh_num'],
            #         num_samples=5000
            #     )
            #     val_datasets.append(dataset)
            #
            #     # 合并所有验证数据
            # self.dataset = torch.utils.data.ConcatDataset(val_datasets)
            print("正在生成baseline专用测评数据......")
            self.dataset = self.problem.make_dataset(
                 size = self.opts.graph_size, veh_num = self.opts.veh_num, num_samples=self.opts.val_size, distribution=self.opts.data_distribution)
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))  # [epoch_size, 1] (num_samples)

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        # 定义反射矩阵
        reflection_matrices = torch.tensor([
            [[1, 0], [0, 1]],  # (x, y)
            [[0, 1], [1, 0]],  # (y, x)
            [[1, 0], [0, -1]],  # (x, 1-y)
            [[0, -1], [1, 0]],  # (y, 1-x)
            [[-1, 0], [0, 1]],  # (1-x, y)
            [[0, 1], [-1, 0]],  # (1-y, x)
            [[-1, 0], [0, -1]],  # (1-x, 1-y)
            [[0, -1], [-1, 0]]  # (1-y, 1-x)
        ]).to(x.device, torch.float)
        costs = torch.zeros((x['loc'].size(0), len(reflection_matrices))).to(x.device)
        # 将C进行对称性变换
        loc = x['loc']  # 假设loc是二维坐标的张量
        depot = x['depot']  # 假设depot也是二维坐标的张量
        capacity_loss = None
        # Loop through each reflection matrix
        for i, matrix in enumerate(reflection_matrices):
            # 进行变换
            transformed_loc = loc @ matrix.T
            transformed_depot = depot @ matrix.T
            x['loc'] = transformed_loc
            x['depot'] = transformed_depot
            # Use volatile mode for efficient inference (single batch so we don't use rollout function)
            with torch.no_grad():
                v, _, _ = self.model(x)  # 返回baseline和cost
            costs[:, i] = v
            break
        # 返回最小成本以及其对应的实例
        return costs.min(dim=1).values

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))

        # if candidate model have smaller cost than current baseline model
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)
                self.epoch_count = 10
            else:
                 self.epoch_count =  self.epoch_count - 1

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)
