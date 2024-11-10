import gym
from gym import spaces
import torch as th
import copy
import dgl

tool_num = 23
operation_num = 10

time_of_change_tool = 1
punish_of_change_process = 1

class WorldofPPv3_1(gym.Env):
    def __init__(self, part_file, device='cuda', tool_num=23, operation_num=10, state_process_num=10,
                 max_repeat_step=100, base_reward=0, tau=0.8):
        self.tau = tau
        graphs, _ = dgl.load_graphs(part_file)
        graph = graphs[0].to(device)
        self.graph = graph

        self.num_nodes = graph.num_nodes()
        self.feasible_vector = graph.dstdata["NodeAttr"][:, :tool_num * operation_num]
        self.time_vector = graph.dstdata["NodeAttr"][:, tool_num * operation_num:-1]
        self.if_father_node = graph.dstdata["NodeAttr"][:, -1]

        self.dis_matrix = graph.dstdata["Dis_matrix"]
        self.if_entire_benchmark = graph.dstdata["If_entire_benchmark"]

        graph_all_attr = copy.deepcopy(graph)
        graph_all_attr.dstdata["NodeAttr"] = th.concat(
            (self.time_vector, self.if_father_node.unsqueeze(-1), self.if_entire_benchmark.unsqueeze(-1)),
            dim=-1)  # 260+1+1

        self.T_graph = graph_all_attr
        self.mask_vector = self.feasible_vector
        self.T_graph.dstdata["Mask_vector"] = self.mask_vector

        self.father_node_id = self.T_graph.dstdata["Father_id"]
        self.benchmark_node_id = self.T_graph.dstdata["Benchmark_id"]
        self.entire_benchmark_id = th.argwhere(self.if_entire_benchmark == 1).item()
        self.feasible_matrix = self.feasible_vector.reshape((self.feasible_vector.shape[0], tool_num, operation_num))
        self.time_matrix = self.time_vector.reshape((self.time_vector.shape[0], tool_num, operation_num))

        self.device = device

        self.tool_num = tool_num
        self.action_process_num = operation_num
        self.state_process_num = state_process_num
        self.max_repeat_step = max_repeat_step
        self.base_reward = base_reward

        self.observation_space = spaces.Graph(
            node_space=spaces.Box(low=-1, high=1, shape=(
                self.T_graph.dstdata["NodeAttr"].shape[1] + 2 * self.state_process_num + self.tool_num + 2, 1)),
            edge_space=spaces.Discrete(5)
        )
        self.action_space = spaces.MultiBinary(self.num_nodes*self.tool_num*self.state_process_num)

        self.serial_tensor = th.arange(0, self.num_nodes * self.tool_num * self.state_process_num).reshape(
            (self.num_nodes, self.tool_num, self.state_process_num))

    def _get_obs(self):
        final_graph = copy.deepcopy(self._state)
        return final_graph.to(self.device)

    def _get_info(self):
        return {}

    def reset(self, seed=None, return_info=False, options=None):
        super(WorldofPPv3_1, self).reset(seed=seed)
        self.step_num = 0
        self._dynamic_info = th.zeros((self.T_graph.num_nodes(), 2 * self.state_process_num + self.tool_num + 2)).to(
            self.device)
        self._dynamic_graph = copy.deepcopy(self.T_graph)
        self._dynamic_graph.dstdata["NodeAttr"] = self._dynamic_info
        self._state = copy.deepcopy(self.T_graph).to(self.device)
        self._state.dstdata["NodeAttr"] = th.concat([self.T_graph.dstdata["NodeAttr"], self._dynamic_info], 1)
        self._state.dstdata["Dis_matrix"] = self.dis_matrix

        observation = self._get_obs()
        info = self._get_info()

        return (observation, info) if return_info else observation

    def step(self, action):
        indices = th.where(self.serial_tensor == action)
        action = [indices[0].item(),indices[1].item(), indices[2].item()]
        if (action[0] >= self.T_graph.num_nodes()) | (action[1] >= self.tool_num) | (
                action[2] >= self.action_process_num):
            reward = 0
            temp_state = copy.deepcopy(self._state).to(self.device)
            temp_dynamic_info = copy.deepcopy(self._dynamic_info).to(self.device)
            new_dynamic_graph = temp_dynamic_info  # The action is not executed.
        else:
            action_feature = self.one_hot(action[0], self.T_graph.num_nodes()).to(
                self.device)  # Select a machining feature, e.g. "action_feature = [0,0,0,0,1,0,0,...,0]" represents the 4th feature.
            network_action_tool = self.one_hot(action[1], self.tool_num).to(
                self.device)  # Select a cutting tool, e.g. action_tool = [0,1,0,0,0,0,0,0,...,0]
            network_action_process = self.one_hot(action[2], self.action_process_num).to(
                self.device)  # Select a machining semantic content, e.g. action_process = [0,0,0,0,1,0,0,0,0,0]

            env_tool_ac_id = th.argwhere(network_action_tool == 1).item()  # The id of the selected cutting tool
            env_process_ac_id = th.argwhere(network_action_process == 1).item()  # The id of the selected  machining semantic content
            feature_id_ac = th.argwhere(action_feature == 1).item()  # The id of the selected machining features

            env_process_ac = network_action_process

            temp_state = copy.deepcopy(self._state).to(self.device)
            temp_dynamic_graph = copy.deepcopy(self._dynamic_graph).to(self.device)
            temp_tool_st = temp_dynamic_graph.dstdata["NodeAttr"][:, :self.tool_num]  # The cutting tool at the previous step
            temp_process_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                              self.tool_num:self.tool_num + self.state_process_num]  # Machining semantic content applied
            temp_feature_terminal_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                                       self.tool_num + self.state_process_num]  # The flag of fully machined
            temp_previous_feature_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                                       self.tool_num + self.state_process_num + 1]  # The machining feature at the previous step
            temp_previous_process_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                                       self.tool_num + self.state_process_num + 2:]  # The machining semantic content at the previous step

            if self.judge_first_rough(feature_id_ac, temp_process_st) == 0:  # Judge whether to machine the datum feature first.
                reward = self.base_reward
                new_dynamic_graph = temp_dynamic_graph  # The action is not executed.
            else:
                flag = self.judge_satisfy_all_conditions(feature_id_ac, env_tool_ac_id,
                                                         env_process_ac_id,
                                                         temp_process_st,
                                                         temp_feature_terminal_st,
                                                         )
                if flag == 0:
                    reward = self.base_reward
                    new_dynamic_graph = temp_dynamic_graph  # The action is not executed.
                else:
                    reward = self.calculate_reward_v3(feature_id_ac, env_process_ac, temp_previous_process_st,
                                                      temp_previous_feature_st,
                                                      env_tool_ac_id, temp_tool_st,
                                                      env_process_ac_id)
                    new_dynamic_graph, _ = self.perform_action(temp_dynamic_graph, feature_id_ac,
                                                               network_action_tool, env_process_ac,
                                                               env_tool_ac_id)

        done, reward = self.judge_part_done_for_reward_v3(new_dynamic_graph,
                                                          reward)

        static_attr_num = self.T_graph.dstdata["NodeAttr"].shape[1]
        temp_state.dstdata["NodeAttr"][:, static_attr_num:] = new_dynamic_graph.dstdata["NodeAttr"]

        if reward == self.base_reward:
            self.step_num += 1
            if self.step_num >= self.max_repeat_step:
                done = 1
                # reward = self.get_dead_reward()
                self.step_num = 0
        else:
            self.step_num = 0

        self._state = temp_state
        self._dynamic_graph = new_dynamic_graph

        observation = self._get_obs()
        info = self._get_info()

        if isinstance(reward, th.Tensor):
            reward = reward.item()

        return observation, reward, done, info

    def one_hot(self, num, size):
        o_h = th.zeros(size)
        o_h[num] = 1
        return o_h

    '''
    Judge whether the entire part has been machined.
    '''
    def judge_part_done(self, temp_selected_feature_graph):
        done = 0
        if th.sum(temp_selected_feature_graph.dstdata["NodeAttr"][:,
                  33]).item() == temp_selected_feature_graph.num_nodes():
            done = 1
        return done


    def judge_part_done_for_reward_v3(self, temp_dynamic_graph, reward):
        done = 0
        if th.sum(temp_dynamic_graph.dstdata["NodeAttr"][:,
                  33]).item() == temp_dynamic_graph.num_nodes():
            done = 1
        if done == 1:
            reward += (1 - self.base_reward) / 0.367
        return done, reward

    def get_dead_reward(self):
        return - (1 - self.base_reward) / 0.605 # 100: 0.367; 50: 0.605


    def perform_action(self, temp_dynamic_graph, feature_id_ac, network_action_tool, env_process_ac, env_tool_ac_id):
        network_action_tool_matrix = network_action_tool.repeat((temp_dynamic_graph.num_nodes(), 1))
        temp_dynamic_graph.dstdata["NodeAttr"][:, :23] = network_action_tool_matrix
        temp_dynamic_graph.dstdata["NodeAttr"][feature_id_ac][23:33] += env_process_ac
        env_process_ac_matrix = env_process_ac.repeat((temp_dynamic_graph.num_nodes(), 1))

        temp_dynamic_graph.dstdata["NodeAttr"][:, 34] = th.zeros(temp_dynamic_graph.num_nodes())
        temp_dynamic_graph.dstdata["NodeAttr"][feature_id_ac][34] = 1
        temp_dynamic_graph.dstdata["NodeAttr"][:, 35:] = env_process_ac_matrix

        env_process_ac_id = th.argwhere(env_process_ac == 1).item()
        if env_process_ac_id == 4:  # Finishing sides
            if th.sum(self.feasible_matrix[feature_id_ac, :, 7]) != 0:  # It may use finishing corner operation
                if self.feasible_matrix[feature_id_ac][env_tool_ac_id][4] + \
                        self.feasible_matrix[feature_id_ac][env_tool_ac_id][7] == 2:  # During finishing the sides, the corners were also finishing.
                    temp_dynamic_graph.dstdata["NodeAttr"][feature_id_ac][30] = 1
        elif env_process_ac_id == 5:  # 5-axis finishing sides
            if th.sum(self.feasible_matrix[feature_id_ac, :, 8]) != 0:  # It may use finishing corner operation with 5 axis-CNC
                if self.feasible_matrix[feature_id_ac][env_tool_ac_id][5] + \
                        self.feasible_matrix[feature_id_ac][env_tool_ac_id][8] == 2:  # During finishing the sides with 5 axis-CNC, the corners were also finishing.
                    temp_dynamic_graph.dstdata["NodeAttr"][feature_id_ac][31] = 1
        elif env_process_ac_id == 0:  # Roughing
            if th.sum(self.feasible_matrix[feature_id_ac, :, 9]) != 0:  # It may require secondary roughing operation.
                if self.feasible_matrix[feature_id_ac][env_tool_ac_id][0] + \
                        self.feasible_matrix[feature_id_ac][env_tool_ac_id][9] == 2:  # The cutting tool that was used for roughing the first time can also be used for the second roughing, so it only needs to be roughed once.
                    temp_dynamic_graph.dstdata["NodeAttr"][feature_id_ac][32] = 1

        finish, _ = self.judge_complete_feature(feature_id_ac, temp_dynamic_graph)
        if finish:
            temp_dynamic_graph.dstdata["NodeAttr"][feature_id_ac][33] = 1

        return temp_dynamic_graph, finish

    '''
    Judge whether to machine the datum feature first.
    '''
    def judge_first_rough(self, feature_id_ac, temp_process_st):
        flag = 1
        if th.sum(temp_process_st) == 0:
            if feature_id_ac != self.entire_benchmark_id:
                flag = 0
        return flag

    '''
    Determine whether a certain machining feature has been fully machined.
    '''
    def judge_complete_feature(self, feature_id_ac, temp_dynamic_graph):
        process_finish_vec, _ = th.max(self.feasible_matrix[feature_id_ac], dim=0)
        temp_process_vec = temp_dynamic_graph.dstdata["NodeAttr"][feature_id_ac][23:33]
        finish = False
        # if th.sum(temp_process_vec)>=th.sum(process_finish_vec): # Old version
        #     finish = True
        if (temp_process_vec>=process_finish_vec).all():  # New version
            finish = True

        if finish:
            temp_dynamic_graph.dstdata["NodeAttr"][feature_id_ac][33] = 1

        return finish, temp_dynamic_graph

    '''
    Determine whether the machining operation meets the rules.
    '''
    def judge_satisfy_all_conditions(self, feature_id_ac, env_tool_ac_id, env_process_ac_id, temp_process_st,
                                     temp_feature_terminal_st):

        flag_1 = self.judge_machine_finished_feature(feature_id_ac, temp_feature_terminal_st)
        if flag_1 == 0: return 0
        flag_2 = self.judge_process_repeat(feature_id_ac, env_process_ac_id, temp_process_st)
        if flag_2 == 0: return 0
        flag_3 = self.judge_operation_feasibility(env_tool_ac_id, feature_id_ac, env_process_ac_id)
        if flag_3 == 0: return 0
        # flag_4 = self.judge_roughing_all_features(env_process_ac_id, temp_process_st)
        # if flag_4 == 0: return 0
        flag_5 = self.judge_process_master_slave_features(feature_id_ac, temp_process_st)
        if flag_5 == 0: return 0
        flag_6 = self.judge_process_benchmark_features(feature_id_ac, temp_feature_terminal_st, env_process_ac_id)
        if flag_6 == 0: return 0
        flag_10 = self.judge_process_order_for_feature(feature_id_ac, env_process_ac_id, temp_process_st)
        if flag_10 == 0:
            return 0
        else:
            return 1

    '''
    Determine whether a feature repeats the same machining semantic content.
    '''
    def judge_process_repeat(self, feature_id_ac, env_process_ac_id, temp_process_st):
        flag = 1
        if temp_process_st[feature_id_ac, env_process_ac_id] == 1:
            flag = 0
        return flag

    '''
    Determine whether a feature is roughed first and then finished.
    '''
    def judge_process_order_for_feature(self, feature_id_ac, env_process_ac_id, temp_process_st):
        flag = 1
        if temp_process_st[feature_id_ac, 0] == 0:  # Did not perform the roughing operation.
            if env_process_ac_id != 0:  # The action is not roughing.
                flag = 0
        elif (temp_process_st[feature_id_ac, 9] == 0) & (
                th.sum(self.feasible_matrix[feature_id_ac, :, 9]) != 0):  # Did not perform the second roughing operation.
            if env_process_ac_id != 9:  # The action is not second roughing.
                flag = 0
        else:
            if env_process_ac_id == 4:  # Finishing sides
                if th.sum(self.feasible_matrix[feature_id_ac, :, 1]) != 0:  # Semi-finishing sides is required.
                    if temp_process_st[feature_id_ac, 1] != 1:  # Semi-finishing sides is not performed.
                        flag = 0
            elif env_process_ac_id == 5:  # Finishing sides with 5-axis CNC
                if th.sum(self.feasible_matrix[feature_id_ac, :, 2]) != 0:  # Semi-finishing sides with 5-axis CNC is required.
                    if temp_process_st[feature_id_ac, 2] != 1:  # Semi-finishing sides with 5-axis CNC is not performed.
                        flag = 0
            elif env_process_ac_id == 6:  # Finishing bottom
                if th.sum(self.feasible_matrix[feature_id_ac, :, 3]) != 0:  # Semi-finishing bottom is required
                    if temp_process_st[feature_id_ac, 3] != 1:  # Semi-finishing bottom is not performed.
                        flag = 0
            elif (env_process_ac_id == 7) & (temp_process_st[feature_id_ac, 4] != 1):  # The sides must be finished before finishing the corners.
                flag = 0
            elif (env_process_ac_id == 8) & (temp_process_st[feature_id_ac, 5] != 1):  # The sides must be 5-axis finished before 5-axis finishing the corners.
                flag = 0
        return flag

    '''
    If a feature has been fully machined, but the machining operation is still performed to the feature, then the reward is 0.
    '''
    def judge_machine_finished_feature(self, feature_id_ac, temp_feature_terminal_st):
        if temp_feature_terminal_st[feature_id_ac] == 1:
            flag = 0
        else:
            flag = 1
        return flag

    '''
    Determine the correctness of tool and machining semantic content selection.
    '''
    def judge_operation_feasibility(self, env_tool_ac, feature_id_ac, env_process_ac_id):
        flag = 1
        if self.feasible_matrix[feature_id_ac][env_tool_ac][env_process_ac_id] != 1:
            flag = 0
        return flag

    '''
    Determine whether to rough the master feature first and then machine the slave feature.
    '''
    def judge_process_master_slave_features(self, feature_id_ac, temp_process_st):
        flag = 1
        father_node_id = self.father_node_id[feature_id_ac].int()
        if father_node_id != -1:
            if th.sum(self.feasible_matrix[father_node_id, :, 9]) == 0:
                if temp_process_st[father_node_id][0].item() == 0:
                    flag = 0
            else:
                if temp_process_st[father_node_id][9].item() == 0:
                    flag = 0

        return flag

    '''
    Determine whether to finish finishing the datum first (it can also be understood as the 
    datum feature being fully machined), and then semi-finish or finish other features based 
    on this feature.
    '''
    def judge_process_benchmark_features(self, feature_id_ac, temp_feature_terminal_st, env_process_ac_id):
        flag = 1
        if (env_process_ac_id!=0)&(env_process_ac_id!=9):
            benchmark_node_id = self.benchmark_node_id[feature_id_ac].int()
            if benchmark_node_id != -1:
                if temp_feature_terminal_st[self.entire_benchmark_id].item() != 1:
                    flag = 0
                if temp_feature_terminal_st[benchmark_node_id].item() != 1:
                    flag = 0
        return flag

    '''
    Finishing features without roughing all features.
    '''
    def judge_roughing_all_features(self, env_process_ac_id, temp_process_st):
        flag = 1
        if (env_process_ac_id!=0)&(env_process_ac_id!=9): # 不是开粗的加工操作
            if th.sum(temp_process_st[:, 0])!=temp_process_st.shape[0]: # 没有完成所有特征的第一次开粗
                flag = 0
        return flag

    def calculate_machining_time(self, feature_id_ac, env_process_ac_id, env_tool_ac_id):
        time = self.time_matrix[feature_id_ac][env_tool_ac_id][env_process_ac_id]
        return time.item()

    def calculate_prepare_time(self, feature_id_ac, temp_previous_process_st, temp_previous_feature_st, env_process_ac,
                               env_tool_ac_id, temp_tool_st):
        if th.sum(temp_previous_feature_st) == 0:
            l = 0
        else:
            previous_feature_id = th.argwhere(temp_previous_feature_st == 1).item()
            l = self.dis_matrix[previous_feature_id, feature_id_ac]

        t_2 = l
        t_3 = 0
        if th.sum(temp_tool_st[0]) != 0:  # Any feature use the same tool
            env_previous_tool_ac = th.argwhere(temp_tool_st[0] == 1).item()
            if env_tool_ac_id != env_previous_tool_ac:
                t_3 = time_of_change_tool

        env_previous_process_ac = temp_previous_process_st[0]
        t_5 = 0
        if th.sum(env_previous_process_ac) != 0:
            index_last = th.argwhere(env_previous_process_ac == 1).item()
            index_ac = th.argwhere(env_process_ac == 1).item()
            if (index_last == 0) | (index_last == 9):
                if (index_ac != 0) & (index_ac != 9):
                    t_5 = punish_of_change_process+time_of_change_tool
            if (index_last != 0) & (index_last != 9):
                if (index_ac == 0) | (index_ac == 9):
                    t_5 = punish_of_change_process+time_of_change_tool

        if th.sum(env_previous_process_ac) != 0:
            index_last = th.argwhere(env_previous_process_ac == 1).item()
            index_ac = th.argwhere(env_process_ac == 1).item()
            if index_last != index_ac:
                t_5 += punish_of_change_process + time_of_change_tool

        t = t_2 + t_3 + t_5
        return t

    def calculate_reward_v3(self, feature_id_ac, env_process_ac, temp_previous_process_st, temp_previous_feature_st,
                            env_tool_ac_id, temp_tool_st,
                            env_process_ac_id):
        time_p = self.calculate_prepare_time(feature_id_ac, temp_previous_process_st, temp_previous_feature_st,
                                             env_process_ac,
                                             env_tool_ac_id, temp_tool_st)
        time_m = self.calculate_machining_time(feature_id_ac, env_process_ac_id, env_tool_ac_id)
        time = (time_m + time_p * self.tau / 3) / (1 + self.tau)
        reward = -time + 1

        return reward

# Only consider rough machining operations.
class WorldofPPv3_6_only_rough(WorldofPPv3_1):
    def __init__(self, part_file, device='cuda', tool_num=23, operation_num=10, state_process_num=10,
                 max_repeat_step=100,
                 base_reward=0, tau=0.4): #base_reward<=0
        super(WorldofPPv3_6_only_rough, self).__init__(part_file, device, tool_num, operation_num, state_process_num,
                 max_repeat_step, base_reward, tau)
        graph_t = self.graph

        self.num_nodes = graph_t.num_nodes()

        f_v = graph_t.dstdata["NodeAttr"][:, :tool_num * operation_num]
        f_m = f_v.reshape((f_v.shape[0], tool_num, operation_num))
        f_m_m = copy.copy(f_m)
        self.need_second_rough = th.zeros(self.num_nodes)

        t_v = graph_t.dstdata["NodeAttr"][:, tool_num * operation_num:-1]
        t_m = t_v.reshape((t_v.shape[0], tool_num, operation_num))
        t_m_m = copy.copy(t_m)

        f_m_m[:, :, 1:9] = th.zeros((f_m_m.shape[0], f_m_m.shape[1], 8))
        t_m_m[:, :, 1:9] = th.zeros((f_m_m.shape[0], f_m_m.shape[1], 8))

        f_m_m_v = f_m_m.reshape((f_v.shape[0], tool_num*operation_num))

        t_m_m_v = t_m_m.reshape((t_v.shape[0], tool_num * operation_num))
        i_f_n = graph_t.dstdata["NodeAttr"][:, -1].unsqueeze(-1)

        all_reset_attr = th.concat((f_m_m_v, t_m_m_v, i_f_n), dim=-1)

        graph_reset = copy.deepcopy(graph_t)
        graph_reset.dstdata["NodeAttr"] = all_reset_attr

        self.feasible_vector = graph_reset.dstdata["NodeAttr"][:, :tool_num * operation_num]
        self.time_vector = graph_reset.dstdata["NodeAttr"][:, tool_num * operation_num:-1]
        self.if_father_node = graph_reset.dstdata["NodeAttr"][:, -1]

        self.dis_matrix = graph_reset.dstdata["Dis_matrix"]
        self.if_entire_benchmark = graph_reset.dstdata["If_entire_benchmark"]

        graph_all_attr = copy.deepcopy(graph_reset)
        graph_all_attr.dstdata["NodeAttr"] = th.concat(
            (self.time_vector, self.if_father_node.unsqueeze(-1), self.if_entire_benchmark.unsqueeze(-1)),
            dim=-1)  # 260+1+1

        self.T_graph = graph_all_attr
        self.mask_vector = self.feasible_vector
        self.T_graph.dstdata["Mask_vector"] = self.mask_vector

        self.father_node_id = self.T_graph.dstdata["Father_id"]
        self.benchmark_node_id = self.T_graph.dstdata["Benchmark_id"]
        self.entire_benchmark_id = th.argwhere(self.if_entire_benchmark == 1).item()
        self.feasible_matrix = self.feasible_vector.reshape((self.feasible_vector.shape[0], tool_num, operation_num))
        self.time_matrix = self.time_vector.reshape((self.time_vector.shape[0], tool_num, operation_num))

    def step(self, action):
        indices = th.where(self.serial_tensor == action)
        action = [indices[0].item(),indices[1].item(), indices[2].item()]
        if (action[0] >= self.T_graph.num_nodes()) | (action[1] >= self.tool_num) | (
                action[2] >= self.action_process_num):
            reward = 0
            temp_state = copy.deepcopy(self._state).to(self.device)
            temp_dynamic_info = copy.deepcopy(self._dynamic_info).to(self.device)
            new_dynamic_graph = temp_dynamic_info
        else:
            action_feature = self.one_hot(action[0], self.T_graph.num_nodes()).to(
                self.device)
            network_action_tool = self.one_hot(action[1], self.tool_num).to(
                self.device)
            network_action_process = self.one_hot(action[2], self.action_process_num).to(
                self.device)

            env_tool_ac_id = th.argwhere(network_action_tool == 1).item()
            env_process_ac_id = th.argwhere(network_action_process == 1).item()
            feature_id_ac = th.argwhere(action_feature == 1).item()

            env_process_ac = network_action_process

            temp_state = copy.deepcopy(self._state).to(self.device)
            temp_dynamic_graph = copy.deepcopy(self._dynamic_graph).to(self.device)
            temp_tool_st = temp_dynamic_graph.dstdata["NodeAttr"][:, :self.tool_num]
            temp_process_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                              self.tool_num:self.tool_num + self.state_process_num]
            temp_feature_terminal_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                                       self.tool_num + self.state_process_num]
            temp_previous_feature_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                                       self.tool_num + self.state_process_num + 1]
            temp_previous_process_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                                       self.tool_num + self.state_process_num + 2:]

            if self.judge_first_rough(feature_id_ac, temp_process_st) == 0:
                reward = self.base_reward
                new_dynamic_graph = temp_dynamic_graph
            else:
                flag = self.judge_satisfy_all_conditions(feature_id_ac, env_tool_ac_id,
                                                         env_process_ac_id,
                                                         temp_process_st,
                                                         temp_feature_terminal_st,
                                                         )
                if flag == 0:
                    reward = self.base_reward
                    new_dynamic_graph = temp_dynamic_graph
                else:
                    reward = self.calculate_reward_v3(feature_id_ac, env_process_ac, temp_previous_process_st,
                                                      temp_previous_feature_st,
                                                      env_tool_ac_id, temp_tool_st,
                                                      env_process_ac_id)
                    new_dynamic_graph, _ = self.perform_action(temp_dynamic_graph, feature_id_ac,
                                                               network_action_tool, env_process_ac,
                                                               env_tool_ac_id)

        done, reward = self.judge_part_done_for_reward_v3(new_dynamic_graph,
                                                          reward)

        static_attr_num = self.T_graph.dstdata["NodeAttr"].shape[1]
        temp_state.dstdata["NodeAttr"][:, static_attr_num:] = new_dynamic_graph.dstdata["NodeAttr"]

        if reward == self.base_reward:
            self.step_num += 1
            if self.step_num >= self.max_repeat_step:
                done = 1
                self.step_num = 0
        else:
            self.step_num = 0

        self._state = temp_state
        self._dynamic_graph = new_dynamic_graph

        observation = self._get_obs()
        info = self._get_info()

        if isinstance(reward, th.Tensor):
            reward = reward.item()

        return observation, reward, done, info

    def judge_part_done_for_reward_v4(self, temp_dynamic_graph, reward):
        done = 0
        if th.sum(temp_dynamic_graph.dstdata["NodeAttr"][:,
                  23]).item() == temp_dynamic_graph.num_nodes():
            if th.sum(temp_dynamic_graph.dstdata["NodeAttr"][:,
                  32]).item() == th.sum(self.need_second_rough):
                done = 1
        if done == 1:
            reward += (1 - self.base_reward) / 0.367
        return done, reward

# Do not consider rough machining operations.
class WorldofPPv3_7_without_rough(WorldofPPv3_1):
    def __init__(self, part_file, device='cuda', tool_num=23, operation_num=10, state_process_num=10,
                 max_repeat_step=100,
                 base_reward=0, tau=0.4): #base_reward<=0
        super(WorldofPPv3_7_without_rough, self).__init__(part_file, device, tool_num, operation_num, state_process_num,
                 max_repeat_step, base_reward, tau)
        graph_t = self.graph
        self.num_nodes = graph_t.num_nodes()
        f_v = graph_t.dstdata["NodeAttr"][:, :tool_num * operation_num]
        f_m = f_v.reshape((f_v.shape[0], tool_num, operation_num))
        f_m_m = copy.copy(f_m)
        t_v = graph_t.dstdata["NodeAttr"][:, tool_num * operation_num:-1]
        t_m = t_v.reshape((t_v.shape[0], tool_num, operation_num))
        t_m_m = copy.copy(t_m)

        f_m_m[:, :, 0] = th.zeros((f_m_m.shape[0], f_m_m.shape[1]))
        f_m_m[:, :, 9] = th.zeros((f_m_m.shape[0], f_m_m.shape[1]))
        t_m_m[:, :, 0] = th.zeros((f_m_m.shape[0], f_m_m.shape[1]))
        t_m_m[:, :, 9] = th.zeros((f_m_m.shape[0], f_m_m.shape[1]))
        f_m_m_v = f_m_m.reshape((f_v.shape[0], tool_num*operation_num))

        t_m_m_v = t_m_m.reshape((t_v.shape[0], tool_num * operation_num))
        i_f_n = graph_t.dstdata["NodeAttr"][:, -1].unsqueeze(-1)

        all_reset_attr = th.concat((f_m_m_v, t_m_m_v, i_f_n), dim=-1)

        graph_reset = copy.deepcopy(graph_t)
        graph_reset.dstdata["NodeAttr"] = all_reset_attr

        self.feasible_vector = graph_reset.dstdata["NodeAttr"][:, :tool_num * operation_num]
        self.time_vector = graph_reset.dstdata["NodeAttr"][:, tool_num * operation_num:-1]
        self.if_father_node = graph_reset.dstdata["NodeAttr"][:, -1]
        self.dis_matrix = graph_reset.dstdata["Dis_matrix"]
        self.if_entire_benchmark = graph_reset.dstdata["If_entire_benchmark"]

        graph_all_attr = copy.deepcopy(graph_reset)
        graph_all_attr.dstdata["NodeAttr"] = th.concat(
            (self.time_vector, self.if_father_node.unsqueeze(-1), self.if_entire_benchmark.unsqueeze(-1)),
            dim=-1)  # 260+1+1

        self.T_graph = graph_all_attr
        self.mask_vector = self.feasible_vector
        self.T_graph.dstdata["Mask_vector"] = self.mask_vector

        self.father_node_id = self.T_graph.dstdata["Father_id"]
        self.benchmark_node_id = self.T_graph.dstdata["Benchmark_id"]
        self.entire_benchmark_id = th.argwhere(self.if_entire_benchmark == 1).item()
        self.feasible_matrix = self.feasible_vector.reshape((self.feasible_vector.shape[0], tool_num, operation_num))
        self.time_matrix = self.time_vector.reshape((self.time_vector.shape[0], tool_num, operation_num))

    def step(self, action):
        indices = th.where(self.serial_tensor == action)
        action = [indices[0].item(),indices[1].item(), indices[2].item()]
        if (action[0] >= self.T_graph.num_nodes()) | (action[1] >= self.tool_num) | (
                action[2] >= self.action_process_num):
            reward = 0
            temp_state = copy.deepcopy(self._state).to(self.device)
            temp_dynamic_info = copy.deepcopy(self._dynamic_info).to(self.device)
            new_dynamic_graph = temp_dynamic_info
        else:
            action_feature = self.one_hot(action[0], self.T_graph.num_nodes()).to(
                self.device)
            network_action_tool = self.one_hot(action[1], self.tool_num).to(
                self.device)
            network_action_process = self.one_hot(action[2], self.action_process_num).to(
                self.device)

            env_tool_ac_id = th.argwhere(network_action_tool == 1).item()
            env_process_ac_id = th.argwhere(network_action_process == 1).item()
            feature_id_ac = th.argwhere(action_feature == 1).item()

            env_process_ac = network_action_process

            temp_state = copy.deepcopy(self._state).to(self.device)
            temp_dynamic_graph = copy.deepcopy(self._dynamic_graph).to(self.device)
            temp_tool_st = temp_dynamic_graph.dstdata["NodeAttr"][:, :self.tool_num]
            temp_process_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                              self.tool_num:self.tool_num + self.state_process_num]
            temp_feature_terminal_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                                       self.tool_num + self.state_process_num]
            temp_previous_feature_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                                       self.tool_num + self.state_process_num + 1]
            temp_previous_process_st = temp_dynamic_graph.dstdata["NodeAttr"][:,
                                       self.tool_num + self.state_process_num + 2:]

            if self.judge_first_rough(feature_id_ac, temp_process_st) == 0:
                reward = self.base_reward
                new_dynamic_graph = temp_dynamic_graph
            else:
                flag = self.judge_satisfy_all_conditions_v3(feature_id_ac, env_tool_ac_id,
                                                         env_process_ac_id,
                                                         temp_process_st,
                                                         temp_feature_terminal_st,
                                                         )
                if flag == 0:
                    reward = self.base_reward
                    new_dynamic_graph = temp_dynamic_graph
                else:
                    reward = self.calculate_reward_v3(feature_id_ac, env_process_ac, temp_previous_process_st,
                                                      temp_previous_feature_st,
                                                      env_tool_ac_id, temp_tool_st,
                                                      env_process_ac_id)
                    new_dynamic_graph, _ = self.perform_action(temp_dynamic_graph, feature_id_ac,
                                                               network_action_tool, env_process_ac,
                                                               env_tool_ac_id)

        done, reward = self.judge_part_done_for_reward_v3(new_dynamic_graph,
                                                          reward)

        static_attr_num = self.T_graph.dstdata["NodeAttr"].shape[1]
        temp_state.dstdata["NodeAttr"][:, static_attr_num:] = new_dynamic_graph.dstdata["NodeAttr"]

        if reward == self.base_reward:
            self.step_num += 1
            if self.step_num >= self.max_repeat_step:
                done = 1
                self.step_num = 0
        else:
            self.step_num = 0

        self._state = temp_state
        self._dynamic_graph = new_dynamic_graph

        observation = self._get_obs()
        info = self._get_info()

        if isinstance(reward, th.Tensor):
            reward = reward.item()

        return observation, reward, done, info


    def judge_satisfy_all_conditions_v3(self, feature_id_ac, env_tool_ac_id, env_process_ac_id, temp_process_st,
                                     temp_feature_terminal_st):
        flag_1 = self.judge_machine_finished_feature(feature_id_ac, temp_feature_terminal_st)
        if flag_1 == 0: return 0
        flag_2 = self.judge_process_repeat(feature_id_ac, env_process_ac_id, temp_process_st)
        if flag_2 == 0: return 0
        flag_3 = self.judge_operation_feasibility(env_tool_ac_id, feature_id_ac, env_process_ac_id)
        if flag_3 == 0: return 0
        # flag_4 = self.judge_roughing_all_features(env_process_ac_id, temp_process_st)
        # if flag_4 == 0: return 0
        # flag_5 = self.judge_process_master_slave_features(feature_id_ac, temp_process_st)
        # if flag_5 == 0: return 0
        flag_6 = self.judge_process_benchmark_features(feature_id_ac, temp_feature_terminal_st, env_process_ac_id)
        if flag_6 == 0: return 0
        flag_10 = self.judge_process_order_for_feature_v3(feature_id_ac, env_process_ac_id, temp_process_st)
        if flag_10 == 0:
            return 0
        else:
            return 1

    def judge_process_order_for_feature_v3(self, feature_id_ac, env_process_ac_id, temp_process_st):
        flag = 1
        if env_process_ac_id == 4:
            if th.sum(self.feasible_matrix[feature_id_ac, :, 1]) != 0:
                if temp_process_st[feature_id_ac, 1] != 1:
                    flag = 0
        elif env_process_ac_id == 5:  # 五轴精侧壁
            if th.sum(self.feasible_matrix[feature_id_ac, :, 2]) != 0:
                if temp_process_st[feature_id_ac, 2] != 1:
                    flag = 0
        elif env_process_ac_id == 6:  # 精底面
            if th.sum(self.feasible_matrix[feature_id_ac, :, 3]) != 0:
                if temp_process_st[feature_id_ac, 3] != 1:
                    flag = 0
        elif (env_process_ac_id == 7) & (temp_process_st[feature_id_ac, 4] != 1):
            flag = 0
        elif (env_process_ac_id == 8) & (temp_process_st[feature_id_ac, 5] != 1):
            flag = 0
        return flag

class WorldofPPv3_1_predict(WorldofPPv3_1):
    def __init__(self, part_file, device='cuda', tool_num=23, operation_num=10, state_process_num=10,
                 max_repeat_step=100,
                 base_reward=0, tau=0.5, real_time=0):  # base_reward<=0
        super(WorldofPPv3_1_predict, self).__init__(part_file, device, tool_num, operation_num, state_process_num,
                                            max_repeat_step, base_reward, tau)
        if real_time:
            self.dis_matrix_org = self.graph.dstdata["Dis_matrix_org"]
            self.time_vector_origin = self.graph.dstdata["Time_origin"]
            self.time_matrix_origin = self.time_vector_origin.reshape((self.time_vector.shape[0], tool_num, operation_num))

    def mode_state(self, state, dynamic_graph):
        self._state = state
        self._dynamic_graph = dynamic_graph

class WorldofPPv3_6_predict(WorldofPPv3_6_only_rough):
    def __init__(self, part_file, device='cuda', tool_num=23, operation_num=10, state_process_num=10,
                 max_repeat_step=100,
                 base_reward=0, tau=0.5, real_time=0):  # base_reward<=0
        super(WorldofPPv3_6_predict, self).__init__(part_file, device, tool_num, operation_num, state_process_num,
                                            max_repeat_step, base_reward, tau)
        if real_time:
            self.dis_matrix_org = self.graph.dstdata["Dis_matrix_org"]
            self.time_vector_origin = self.graph.dstdata["Time_origin"]
            self.time_matrix_origin = self.time_vector_origin.reshape((self.time_vector.shape[0], tool_num, operation_num))

    def mode_state(self, state, dynamic_graph):
        self._state = state
        self._dynamic_graph = dynamic_graph

class WorldofPPv3_7_predict(WorldofPPv3_7_without_rough):
    def __init__(self, part_file, device='cuda', tool_num=23, operation_num=10, state_process_num=10,
                 max_repeat_step=100,
                 base_reward=0, tau=0.5, real_time=0):  # base_reward<=0
        super(WorldofPPv3_7_predict, self).__init__(part_file, device, tool_num, operation_num, state_process_num,
                                            max_repeat_step, base_reward, tau)
        if real_time:
            self.dis_matrix_org = self.graph.dstdata["Dis_matrix_org"]
            self.time_vector_origin = self.graph.dstdata["Time_origin"]
            self.time_matrix_origin = self.time_vector_origin.reshape((self.time_vector.shape[0], tool_num, operation_num))

    def mode_state(self, state, dynamic_graph):
        self._state = state
        self._dynamic_graph = dynamic_graph
