import gym
import torch as th
import dgl
import numpy as np
import os
from torch.distributions import Categorical
import Enviroments
import tianshou as ts
import PPOMask_Network
import time
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import math

global_device = 'cuda'

def construct_envs(custom_env, path_of_part_train, truncate, parallel, tau):
    env = gym.make(custom_env, part_file=path_of_part_train, device=global_device, max_repeat_step=truncate, tau=tau)

    train_envs = ts.env.DummyVectorEnv([lambda: gym.wrappers.AutoResetWrapper(
        gym.make(custom_env, part_file=path_of_part_train, device=global_device,
                 max_repeat_step=truncate, tau=tau))] * parallel)

    test_envs = ts.env.DummyVectorEnv([lambda: gym.wrappers.AutoResetWrapper(
        gym.make(custom_env, part_file=path_of_part_train, device=global_device,
                 max_repeat_step=truncate, tau=tau))] * 60)

    return env, train_envs, test_envs

def construct_policy_PPO_all_attr(env):
    in_node_feats = env.observation_space.node_space.shape[0]
    in_edge_feats = gym.spaces.utils.flatdim(env.observation_space.edge_space)
    num_nodes = env.num_nodes
    out_tool_size = env.tool_num
    out_process_size = env.action_process_num
    Actor_net = PPOMask_Network.Masked_Actor_Net_PNAConv(in_node_feats, in_edge_feats, num_nodes,
                                                                       out_tool_size, out_process_size).to(global_device)
    Critic_net = PPOMask_Network.Critic_Net_PNAConv(in_node_feats, in_edge_feats, num_nodes).to(global_device)

    optim = th.optim.Adam([{'params': Actor_net.parameters()}, {'params': Critic_net.parameters()}], lr=3e-4)

    dist_fn = th.distributions.Categorical
    policy = ts.policy.PPOPolicy(actor=Actor_net, critic=Critic_net, optim=optim, dist_fn=dist_fn)
    return policy

def train_PPO_on_policy(train_file):
    graphs, _ = dgl.load_graphs(train_file)
    tau = 0.8
    truncate = 50
    parallel = 200
    epoch = 20
    step_per_epoch = 42000
    repeat_per_collect = 5
    batch_size = 1024
    test_num = 60
    buffer_size = 40000
    step_per_collect = 4200
    logdir = r"...\DRL_MPP\exps"
    task = "WoPPO_without_rough_task_1"

    print("truncate: " + str(truncate))
    print("repeat_per_collect: " + str(repeat_per_collect))
    print("batch_size: " + str(batch_size))
    print("PPO")
    print(task)

    env, train_envs, test_envs = construct_envs("Enviroments/WorldofPPv3_7_without_rough", train_file, truncate, parallel, tau)

    train_envs.seed(1626)
    test_envs.seed(1626)
    policy = construct_policy_PPO_all_attr(env)


    time_start = time.time()
    train_collector = ts.data.Collector(policy=policy, env=train_envs,
                                        buffer=ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=train_envs.env_num),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy=policy, env=test_envs, exploration_noise=True)

    log_path = os.path.join(logdir, task, 'ppo')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        th.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint_" + str(epoch)+".pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        th.save(
            {
                "model": policy.state_dict(),
            }, ckpt_path
        )
        return ckpt_path

    # def stop_fn(mean_rewards):
    #     return mean_rewards >= 15

    result = ts.trainer.onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        epoch,
        step_per_epoch,
        repeat_per_collect,
        test_num,
        batch_size,
        step_per_collect= step_per_collect,
        #stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )

    time_end = time.time()
    print(str(time_end - time_start))
    return result

if __name__ == "__main__":
    train_file = r"...\DRL_MPP\data\ZH_test11_with_cons.dgl"
    train_PPO_on_policy(train_file)