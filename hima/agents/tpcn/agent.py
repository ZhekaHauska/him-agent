import torch
import numpy as np
from enum import Enum, auto
from hima.modules.tpcn.tpcn import TemporalPCN, HierarchicalPCN
from hima.experiments.successor_representations.runners.base import BaseAgent
from hima.common.utils import safe_divide, softmax
from hima.common.sdr import sparse_to_dense
from hima.modules.tpcn.constants import EPS

class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()

class tPCNAgent(BaseAgent):
    def __init__(self, 
                n_obs_states: int,
                n_actions: int,
                tpc_model: TemporalPCN | None = None, 
                init_model: HierarchicalPCN | None = None,
                init_model_config: dict | None = None,
                tpc_model_config: dict | None = None,
                test_inf_iters: int = 5,
                inf_iters: int = 5,
                inf_lr: int = 0.001,
                device: str = 'cpu',
                hidden_dim: int = 512,
                batch_size: int = 64,
                learning_rate: float = 0.005,
                gamma: float = 0.9,
                learn: bool = True,
                reward_lr: float = 0.01,
                learn_rewards_from_state: bool = True,
                plan_steps: int = 1,
                inverse_temp: float = 1.0,
                exploration_eps: float = -1,
                sr_estimate_planning: str = 'uniform',
                sr_early_stop_uniform: float | None = None,
                sr_early_stop_goal: float | None = None,
                seed: int = 42):

        if init_model is None:
            if init_model_config is not None:
                self.init_model = HierarchicalPCN(**init_model_config)
            else:
                raise ValueError('One of the arguments {init_model, init_model_config} must not be None')
        else:
            self.init_model = init_model
        if tpc_model is None:
            if tpc_model_config is not None:
                self.tpc_model = TemporalPCN(**tpc_model_config)
            else:
                raise ValueError('One of the arguments {tpc_model, tpc_model_config} must not be None')
        else:
            self.tpc_model = tpc_model

        self.lr = learning_rate
        self.cum_reward = 0
        self.n_actions = n_actions
        self.n_obs_states = n_obs_states
        self.hidden_dim = hidden_dim
        self.exploration_eps = exploration_eps
        self.device = device
        self.observations = [[]]
        self.actions = [[]]
        self.episode = 1
        self.is_first = True
        self.learn = learn
        self.batch_size = batch_size
        self.inverse_temp = inverse_temp
        self.plan_steps = plan_steps
        self.seed = seed
        self.sr_estimate_planning = sr_estimate_planning
        self.sr_early_stop_uniform = sr_early_stop_uniform
        self.sr_early_stop_goal = sr_early_stop_goal

        self.test_inf_iters = test_inf_iters
        self.inf_lr = inf_lr
        self.inf_iters = inf_iters
        self.tpc_optimizer = torch.optim.Adam(
            self.tpc_model.parameters(), 
            lr=self.lr,
        )
        self.init_optimizer = torch.optim.Adam(
            self.init_model.parameters(),
            lr=self.lr,
        )

        self.learn_rewards_from_state = learn_rewards_from_state
        if self.learn_rewards_from_state:
            rewards = np.zeros((self.hidden_dim))
        else:
            rewards = np.zeros((self.n_obs_states))
        
        self.gamma = gamma
        self.rewards = rewards.flatten()
        self.reward_lr = reward_lr
        self.prev_hidden = None
        self.encoded_action = None
        self.action = None
        self.energy = 0
        self.loss = 0
        self.sf_steps = 0

        if self.exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = self.exploration_eps
        self._rng = np.random.default_rng(seed=seed)
        
    def observe(self, obs, action, reward):
        
        self.encoded_action = torch.tensor(action, dtype=torch.float32).reshape(1, 1, -1)
        encoded_obs = torch.tensor(obs, dtype=torch.float32).reshape(1, 1, -1)
        self.observations[-1].append(encoded_obs)
        if self.encoded_action is not None:
            self.actions[-1].append(self.encoded_action)

        if self.is_first:
            self.init_model.eval()
            with torch.no_grad():
                self.init_model.inference(self.inf_iters, self.inf_lr, encoded_obs)
            
            self.is_first = False
            self.prev_hidden = self.init_model.z.clone().detach()

        else:
            self.tpc_model.eval()
            with torch.no_grad():
                self.tpc_model.inference(self.inf_iters, self.inf_lr, 
                                        self.encoded_action, self.prev_hidden, encoded_obs)
            
            # update the hidden state
            self.prev_hidden = self.tpc_model.z.clone().detach()

    def generate_sf(self, init_hidden, n_steps=5, gamma=0.95, learn_sr=False):
        zs = [init_hidden]
        sr = init_hidden
        discounts = [1.0]
        v = (torch.ones([self.tpc_model.Win.weight.shape[1]]) / self.tpc_model.Win.weight.shape[1]).reshape(1, -1)
        sf = self.init_model.decode(zs[-1])
        self.sf_steps = 1
        for _ in range(n_steps):
            if self.learn_rewards_from_state:
                early_stop = self._early_stop_planning(
                    sr.reshape(-1, 1)
                )
            else:
                early_stop = self._early_stop_planning(
                    sf.reshape(-1, 1)
                )
            
            zs.append(self.tpc_model.g(v, zs[-1]))

            if learn_sr:
                sr += discounts[-1] * zs[-1]
            
            sf += discounts[-1] * self.tpc_model.decode(zs[-1])

            discounts.append(discounts[-1] * gamma)

            self.sf_steps += 1
            
            if early_stop:
                break
        
        if learn_sr:
            return sf.detach().numpy(), sr.detach().numpy()
        else:
            return sf.detach().numpy()

    def predict(self, action: int):

        action_tensor = torch.zeros(self.n_actions, dtype=torch.float32)
        action_tensor[action] = 1.0
        action_tensor = action_tensor.reshape(1, 1, -1)
        pred_hidden = self.tpc_model.g(action_tensor, self.prev_hidden)
        return self.tpc_model.decode(pred_hidden), pred_hidden
    
    def sample_action(self):
        """Evaluate and sample actions."""
        self.action_values = self.evaluate_actions(n_steps=self.plan_steps, gamma=self.gamma)
        self.action_dist = self._get_action_selection_distribution(
            self.action_values, on_policy=True
        )

        self.action = self._rng.choice(self.n_actions, p=self.action_dist)
        self.encoded_action = torch.zeros(self.n_actions, dtype=torch.float32)
        self.encoded_action[self.action] = 1.0
        self.encoded_action = self.encoded_action.reshape(1, 1, -1)
        return self.action
    
    def reinforce(self, reward):
        if self.learn_rewards_from_state:
            deltas = self.prev_hidden.detach().numpy().squeeze() * (reward - self.rewards)
        else:
            deltas = self.tpc_model.decode(self.prev_hidden).detach().numpy().squeeze() * (reward - self.rewards)
        
        self.rewards += self.reward_lr * deltas
    
    def evaluate_actions(self, n_steps=5, gamma=0.95):
        
        n_actions = self.n_actions
        action_values = np.zeros(n_actions)
        dense_action = torch.tensor(np.zeros_like(action_values), dtype=torch.float32)

        for action in range(n_actions):
            dense_action[action - 1] = 0
            dense_action[action] = 1

            pred_hidden = self.tpc_model.g(dense_action, self.prev_hidden)
            if self.learn_rewards_from_state:
                sf, sr = self.generate_sf(
                    init_hidden=pred_hidden, 
                    n_steps=n_steps, 
                    gamma=gamma,
                    learn_sr=True)

                action_values[action] = np.sum(
                        sr * self.rewards
                    ) / self.hidden_dim
            else:
                sf = self.generate_sf(
                init_hidden=pred_hidden, 
                n_steps=n_steps, 
                gamma=gamma)

                action_values[action] = np.sum(
                        sf * self.rewards
                    ) / self.n_obs_states
        return action_values

    def memorize(self):
        self.actions[-1] = self._list_to_tensor(self.actions[-1])
        self.observations[-1] = self._list_to_tensor(self.observations[-1])
        
        sequence_length = self.observations.shape[1]
        if self.learn and self.episode >= self.batch_size:
            self.actions = self._list_to_tensor(self.actions)
            self.observations = self._list_to_tensor(self.observations)
        
        self.tpc_model.train()
        self.init_model.train()
        total_loss = 0 # average loss across time steps
        total_energy = 0 # average energy across time steps

        self.init_optimizer.zero_grad()

        init_actv = self.observations[:, 0, :]
        self.init_model.inference(self.inf_iters, self.inf_lr, init_actv)
        energy, obs_loss = self.init_model.get_energy()
        energy.backward()
        self.init_optimizer.step()
        
        total_loss += obs_loss.item()
        total_energy += energy.item()

        prev_inds = torch.all(~init_actv.isnan(), dim=1).nonzero().flatten()
        prev_hidden = self.init_model.z.clone().detach()
        for k in range(self.actions.shape[1]):
            p = self.observations[:, k+1].to(self.device)
            mask = torch.all(~p.isnan(), dim=1)
            mask = mask[prev_inds]
            prev_hidden = prev_hidden[mask]
            prev_inds = torch.all(~p.isnan(), dim=1).nonzero().flatten()
            p = p[~p.isnan()].reshape(-1, self.n_obs)
            v = self.actions[:, k].to(self.device)
            v = v[~v.isnan()].reshape(-1, self.n_actions)

            self.tpc_optimizer.zero_grad()
            self.tpc_model.inference(self.inf_iters, self.inf_lr, v, prev_hidden, p)
            energy, obs_loss = self.tpc_model.get_energy()
            energy.backward()
            self.tpc_optimizer.step()

            # update the hidden state
            prev_hidden = self.tpc_model.z.clone().detach()

            # add up the loss value at each time step
            total_loss += obs_loss.item()
            total_energy += energy.item()

        self.energy = total_energy / sequence_length
        self.loss = total_loss / sequence_length

    def reset(self):

        if self.episode > self.batch_size:
            self.memorize()
            self.episode = 1
            self.observations = [[]]
            self.actions = [[]]
        else:
            self.episode += 1
            self.actions.append([])
            self.observations.append([])
        
        self.is_first = True
        self.cum_reward = 0
        self.prev_hidden = None
        self.encoded_action = None
        self.action = None

    def _get_action_selection_distribution(
            self, action_values, on_policy: bool = True
    ) -> np.ndarray:
        # off policy means greedy, on policy — with current exploration strategy
        if on_policy and self.exploration_policy == ExplorationPolicy.SOFTMAX:
            # normalize values before applying softmax to make the choice
            # of the softmax temperature scale invariant
            action_values = safe_divide(action_values, np.abs(action_values.sum()))
            action_dist = softmax(action_values, beta=self.inverse_temp)
        else:
            # greedy off policy or eps-greedy
            best_action = np.argmax(action_values)
            # make greedy policy
            # noinspection PyTypeChecker
            action_dist = sparse_to_dense([best_action], like=action_values)

            if on_policy and self.exploration_policy == ExplorationPolicy.EPS_GREEDY:
                # add uniform exploration
                action_dist[best_action] = 1 - self.exploration_eps
                action_dist[:] += self.exploration_eps / self.n_actions

        return action_dist
    
    def _check_shapes(self, lst: list) -> bool:
        if lst[0].shape[:2] == (1, 1):
            return False
        else:
            return True
    
    def _concat_with_padding(self, tensor_list: list) -> torch.Tensor:
        # Находим максимальный размер по оси N
        max_n = max(tensor.shape[1] for tensor in tensor_list)
        
        padded_tensors = []
        for tensor in tensor_list:
            current_n = tensor.shape[1]
            if current_n < max_n:
                # Создаем тензор с NaN значениями для дополнения
                pad_size = max_n - current_n
                nan_padding = torch.full((1, pad_size, tensor.shape[2]), float('nan'),
                                    dtype=tensor.dtype, device=tensor.device)
                # Конкатенируем по оси N
                padded_tensor = torch.cat([tensor, nan_padding], dim=1)
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor)
        
        # Объединяем все тензоры в один
        return torch.cat(padded_tensors, dim=0)

    def _list_to_tensor(self, tensors: list) -> torch.Tensor:
        
        is_train = self._check_shapes(tensors)

        if not is_train:
            return torch.cat(tensors, dim=1)
        else:
            return self._concat_with_padding(tensors)

    def _early_stop_planning(self, states: torch.Tensor) -> bool:
        n_vars, n_states = states.shape

        if self.sr_early_stop_uniform is not None:
            uni_dkl = (
                    torch.sum(
                        states * torch.log(
                            torch.clip(
                                states, EPS, None
                            )
                        ),
                        dim=-1
                    )
            )

            uniform = uni_dkl.mean() < self.sr_early_stop_uniform
        else:
            uniform = False

        if self.sr_early_stop_goal is not None:
            goal = torch.any(
                torch.sum(
                    (states.flatten().detach() * (self.rewards > 0)).reshape(
                        n_vars, -1
                    ),
                    dim=-1
                ) > self.sr_early_stop_goal
            )
        else:
            goal = False

        return uniform or goal
