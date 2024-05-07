import os
import torch as th
import gym
import tianshou as ts
import Train_PPO_mask_net
import numpy as np
import os
import copy
import Enviroments
import time
import dgl

time_of_change_tool = 1
punish_of_change_process = 1
time_of_change_tool_real = 20
punish_of_change_process_real = 20
fate_speed = 10000  # mm/min

global_device="cuda"
tool_num=23
operation_num=10 # the number of machining semantic content

tool_lib = ['D50 R1', 'D30 R5', 'D30 R3', 'D30 R1', 'D25 R5', 'D25 R3', 'D25 R1', 'D20 R5', 'D20 R3', 'D20 R1', 'D18 R5',
            'D18 R3', 'D18 R1', 'D16 R5', 'D16 R3', 'D16 R1', 'D12 R5', 'D12 R3', 'D12 R1', 'D10 R3', 'D10 R1', 'D8 R3',
            'D8 R1']

process_lib_c = ['开粗', '半精铣侧壁','五轴半精铣侧壁','半精铣底面','精铣侧壁','五轴精铣侧壁','精铣底面','精铣转角','五轴精铣转角', '二次开粗']
process_lib = ['Rough_milling','Semi-finish_milling_side','Semi-finish_milling_side_with_5-axis_CNC',
               'Semi-finish_milling_bottom','Finish_milling_side','Finish_milling_side_with_5-axis_CNC',
               'Finish_milling_bottom','Finish_milling_corner','Finish_milling_corner_with_5-axis_CNC',
               'Second_rough_milling']

def read_txt(origin_path):
    origin_ids = []
    files = os.listdir(origin_path)
    temp_file = origin_path+"\\"+files[0]
    file = open(temp_file)
    lines = file.readlines()
    for line in lines:
        line = line.strip().strip("\n")
        line = line.split(" ")
        if line[0] == "Node":
            origin_ids.append(int(line[2].split(":")[1]))
    return origin_ids

class PolicyandValueNet():
    def __init__(self, para_file, customenv, evaluation_file, tau, real_time=0):
        self.env = gym.make(customenv, part_file=evaluation_file, device=global_device, max_repeat_step=50, tau=tau, real_time=real_time)
        self.paras = Train_PPO_mask_net.construct_policy_PPO_all_attr(self.env)
        self.paras.load_state_dict(th.load(para_file))

    def get_paras(self):
        self.paras.eval()
        return self.paras

    def get_env(self):
        return self.env

    def get_actor_network(self):
        policy_network = self.paras.actor
        policy_network.eval()
        return policy_network

    def get_critic_network(self):
        value_network = self.paras.critic
        value_network.eval()
        return value_network


class Predict():
    def __init__(self, policy_network, value_network, environment, state_dim_mode=232, real_time=0):
        self.p_n = policy_network
        self.v_n = value_network
        self.env = environment
        self.state_dim_mode = state_dim_mode
        self.initial_state = self.env.reset()
        self.time_matrix = self.env.time_matrix
        self.dis_matrix = self.env.dis_matrix
        self.tau = self.env.tau
        if real_time:
            self.time_matrix_org = self.env.time_matrix_origin
            self.dis_matrix_org = self.env.dis_matrix_org

    def get_actions(self, state):
        temp_state = np.array([state])
        policy = self.p_n(temp_state)[0].squeeze(0)
        sorted_logits_desc, sorted_indices_desc = th.sort(policy, descending=True)
        sorted_logits_desc_np = sorted_logits_desc.to('cpu').detach().numpy()
        sorted_indices_desc_np = sorted_indices_desc.to('cpu').detach().numpy()

        return sorted_logits_desc_np, sorted_indices_desc_np

    def get_one_action(self, num_nodes, action_id):
        serial_tensor = np.arange(0, num_nodes * tool_num * operation_num).reshape(
            (num_nodes, tool_num, operation_num))

        indices = np.where(serial_tensor == action_id)
        action = [indices[0].item(), indices[1].item(), indices[2].item()]
        return action

    def get_next_state(self, state, action, graph_attr):
        dynamic_info = state.dstdata["NodeAttr"][:, self.state_dim_mode:]
        self.env.reset()
        mystate = copy.deepcopy(graph_attr)
        mystate.dstdata["NodeAttr"] = th.concat([mystate.dstdata["NodeAttr"], dynamic_info], dim=1)

        dynamic_graph = copy.deepcopy(state)
        dynamic_graph.dstdata["NodeAttr"] = state.dstdata["NodeAttr"][:, self.state_dim_mode:]
        self.env.mode_state(mystate, dynamic_graph)

        next_state, reward, _, _ = self.env.step(action)
        return next_state, reward

    def get_state_value(self, state, action, graph_attr):
        next_state, reward = self.get_next_state(state, action, graph_attr)
        v = self.v_n(np.array([next_state]))[0][0]
        q = (v + reward) / 2
        return q, reward

    def get_state_value_2(self, state, action, graph_attr):
        next_state, reward = self.get_next_state(state, action, graph_attr)
        v = self.v_n(np.array([next_state]))[0][0]
        q = (v + reward) / 2
        return q, reward, v

    def calculate_real_time(self, action_0, action_1, action_2):
        machining_time = self.time_matrix_org[action_0][action_1][action_2]
        return machining_time.to('cpu')

    def calculate_prepare_time(self, feature_id_ac, env_process_ac_id, env_tool_ac_id, temp_previous_feature_id,
                                temp_previous_process_st_id, temp_previous_tool_st):
        previous_feature_id = temp_previous_feature_id
        t_2 = self.dis_matrix_org[previous_feature_id, feature_id_ac]/fate_speed

        t_3 = 0
        env_previous_tool_ac = temp_previous_tool_st
        if env_tool_ac_id != env_previous_tool_ac:
            t_3 = time_of_change_tool_real

        t_5 = 0
        index_last = temp_previous_process_st_id
        index_ac = env_process_ac_id
        if (index_last == 0) | (index_last == 9):
            if (index_ac != 0) & (index_ac != 9):
                t_5 = punish_of_change_process_real
        if (index_last != 0) & (index_last != 9):
            if (index_ac == 0) | (index_ac == 9):
                t_5 = punish_of_change_process_real

        if index_last != index_ac:
            t_5 += punish_of_change_process_real
        t = t_2 + t_3 + t_5
        return t.to('cpu')

    def forward_policy(self, num_nodes):
        state = self.initial_state
        all_actions = []
        all_max_vs = []
        all_time = 0
        all_reward = 0
        process_num = 0
        graph_attr = self.env.T_graph
        while True:
            iteration = 0
            actions_logits_des, actions_ids_des = self.get_actions(state)
            finish = 0
            while True:
                action = actions_ids_des[iteration]
                action_pro = actions_logits_des[iteration]
                q, reward, _ = self.get_state_value_2(state, action, graph_attr)
                iteration += 1
                if (reward != 0):
                    break
                elif action_pro==0:
                    finish = 1
                    break
            if finish == 1:
                break
            real_action = self.get_one_action(num_nodes, action)
            t = 0
            if reward!=0:
                t = (1-reward)*(1+self.tau)
            res = self.translate(real_action, q)
            print(real_action)
            print(res)
            print("final_reward:" + str(reward))
            print("max_q: " + str(q.item()))

            dynamic_info = state.dstdata["NodeAttr"][:, self.state_dim_mode:]

            self.env.reset()
            mystate = copy.deepcopy(graph_attr)
            mystate.dstdata["NodeAttr"] = th.concat([mystate.dstdata["NodeAttr"], dynamic_info], dim=1)

            dynamic_graph = copy.deepcopy(state)
            dynamic_graph.dstdata["NodeAttr"] = state.dstdata["NodeAttr"][:, self.state_dim_mode:]

            self.env.mode_state(mystate, dynamic_graph)

            state, reward, done, info = self.env.step(action)
            next_state_obs = state.dstdata["NodeAttr"][:, self.state_dim_mode:].to('cpu').numpy()
            all_actions.append(real_action)
            all_max_vs.append(q)

            # all_time += final_time
            process_num += 1
            all_time += t
            all_reward += reward
            print(process_num)

            if done == 1:
                break

        return all_actions, all_time, all_max_vs, all_reward

    def forward_policy_real_time(self, num_nodes):
        state = self.initial_state
        all_actions = []
        all_max_vs = []
        all_time_real = 0
        all_time_real_without_non_cutting = 0
        all_reward = 0
        process_num = 0
        graph_attr = self.env.T_graph
        while True:
            iteration = 0
            actions_logits_des, actions_ids_des = self.get_actions(state)
            finish = 0
            while True:
                action = actions_ids_des[iteration]
                action_pro = actions_logits_des[iteration]
                q, reward, _ = self.get_state_value_2(state, action, graph_attr)
                iteration += 1
                # if (reward != 0)&(action_pro!=0):
                #     break
                if (reward != 0):
                    break
                elif action_pro==0:
                    finish = 1
                    break
            if finish == 1:
                break

            real_action = self.get_one_action(num_nodes, action)
            t_real = 0
            t_w_non = 0


            if reward!=0:
                real_action_feature = real_action[0]
                real_action_tool = real_action[1]
                real_action_semantic = real_action[2]
                t_real = self.calculate_real_time(real_action_feature, real_action_tool, real_action_semantic)
                t_w_non = copy.deepcopy(t_real)
                if process_num==0:
                    t_pre = 0
                    real_action_last = copy.deepcopy(real_action)
                else:
                    t_pre = self.calculate_prepare_time(real_action_feature,real_action_semantic,real_action_tool,real_action_last[0], real_action_last[2],real_action_last[1])
                    real_action_last = copy.deepcopy(real_action)
                t_real += t_pre

            res = self.translate(real_action, q)
            print(real_action)
            print(res)
            print("final_reward:" + str(reward))
            print("max_q: " + str(q.item()))

            dynamic_info = state.dstdata["NodeAttr"][:, self.state_dim_mode:]

            self.env.reset()
            mystate = copy.deepcopy(graph_attr)
            mystate.dstdata["NodeAttr"] = th.concat([mystate.dstdata["NodeAttr"], dynamic_info], dim=1)

            dynamic_graph = copy.deepcopy(state)
            dynamic_graph.dstdata["NodeAttr"] = state.dstdata["NodeAttr"][:, self.state_dim_mode:]

            self.env.mode_state(mystate, dynamic_graph)

            state, reward, done, info = self.env.step(action)
            all_actions.append(real_action)
            all_max_vs.append(q)

            # all_time += final_time
            process_num += 1
            all_time_real += t_real
            all_time_real_without_non_cutting += t_w_non
            all_reward += reward
            print(process_num)

            if done == 1:
                break

        return all_actions, all_time_real, all_time_real_without_non_cutting, all_max_vs, all_reward

    def translate(self, action, max_q):
        node_id = action[0]
        node_real_id = node_id
        tool_type = tool_lib[action[1]]
        process_type = process_lib[action[2]]
        result = tool_type + ' ' + process_type + ' Feature_' + str(node_real_id)

        return result

    def write_txt(self, all_actions, all_max_vs, file_name, all_time=0, all_reward=0, cal_time=0):
        with open(file_name, "w") as f:
            for i in range(len(all_actions)):
                result = self.translate(all_actions[i], all_max_vs[i])
                f.write(str(i + 1) + ': ' + result + '\n')
            f.write('machine_time: ' + str(all_time) + '\n')
            # f.write('all_reward: ' + str(all_reward) + '\n')
            f.write('cal_time: ' + str(cal_time) + '\n')

def Get_result_real_time(result_file):

    '''
    Part: ZH_test11_with_cons_all_time.dgl, policy: WoPPO_test_11
    Part: ZH_test3_with_cons_all_time.dgl, policy: WoPPO_test_3

    Part: ZH_test18_with_cons_all_time.dgl, policy: WoPPO_test_18_only_rough
    Part: ZH_test18_with_cons_all_time.dgl, policy: WoPPO_test_18_without_rough
    '''
    part_file_with_all_time = r"...\DRL_MPP\data\ZH_test11_with_cons_all_time.dgl"
    policy_file = r"...\DRL_MPP\exps\WoPPO_test_11\ppo\policy.pth"

    tau = 0.8

    '''
    Consider all machining operations: WorldofPPv3_1_predict
    Only consider roughing: WorldofPPv3_6_predict
    Don't consider roughing: WorldofPPv3_7_predict
    '''

    PVn = PolicyandValueNet(policy_file,
                            "Enviroments/WorldofPPv3_1_predict",
                            part_file_with_all_time,
                            tau,
                            real_time=1)

    graphs, _ = dgl.load_graphs(part_file_with_all_time)
    graph = graphs[0]
    num_nodes = graph.num_nodes()

    time_start = time.time()
    a_a = PVn.get_actor_network()
    a_c = PVn.get_critic_network()
    par = PVn.get_paras()
    env = PVn.get_env()
    Pre = Predict(a_a, a_c, env, 232, real_time=1)
    all_actions, all_time, all_time_real_without_non_cutting,all_max_vs, all_reward = Pre.forward_policy_real_time(num_nodes)

    print("all_time: " + str(all_time))
    print("all_time_real_without_non_cutting: " + str(all_time_real_without_non_cutting))
    print("all_reward: " + str(all_reward))

    time_end = time.time()
    print("Compute time: " + str(time_end - time_start))

    Pre.write_txt(all_actions, all_max_vs, result_file, all_time, all_reward, str(time_end - time_start))

if __name__ == "__main__":
    result_file = r"...\DRL_MPP\results\prediction_results\test_11_results.txt"
    Get_result_real_time(result_file)