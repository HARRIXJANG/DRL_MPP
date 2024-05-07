import torch as th
import gym
import tianshou as ts
import Train_PPO_mask_net
import numpy as np
import random
import math
import dgl
import time
import copy

global_device="cuda"
tool_num=23
operation_num=10
fate_speed = 10000  # mm/min

tool_lib = ['D50 R1', 'D30 R5', 'D30 R3', 'D30 R1', 'D25 R5', 'D25 R3', 'D25 R1', 'D20 R5', 'D20 R3', 'D20 R1', 'D18 R5',
            'D18 R3', 'D18 R1', 'D16 R5', 'D16 R3', 'D16 R1', 'D12 R5', 'D12 R3', 'D12 R1', 'D10 R3', 'D10 R1', 'D8 R3',
            'D8 R1']

process_lib_c = ['开粗', '半精铣侧壁','五轴半精铣侧壁','半精铣底面','精铣侧壁','五轴精铣侧壁','精铣底面','精铣转角','五轴精铣转角', '二次开粗']
process_lib = ['Rough_milling','Semi-finish_milling_side','Semi-finish_milling_side_with_5-axis_CNC',
               'Semi-finish_milling_bottom','Finish_milling_side','Finish_milling_side_with_5-axis_CNC',
               'Finish_milling_bottom','Finish_milling_corner','Finish_milling_corner_with_5-axis_CNC',
               'Second_rough_milling']
time_of_change_tool = 1
punish_of_change_process = 1

time_of_change_tool_real = 20
punish_of_change_process_real = 20


class PolicyandValueNet():
    def __init__(self, para_file, customenv, evaluation_file, tau):
        self.env = gym.make(customenv, part_file=evaluation_file, device=global_device, max_repeat_step=50, tau=tau)
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

class Node():
    def __init__(self, state, policy_pro, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.v_n_value = 0
        self.policy_pro = policy_pro
        self.action = None
        self.is_terminal = False
        self.reward = 1.1

    def is_leaf_node(self):
        return len(self.children) == 0

class MCTs():
    def __init__(self, policy_network, value_network, custom_env, c_puct=1, threshold=0.0, sim_num=100,
                 state_dim_mode=232, lamda=1, mrs=0, num_expand_action=20, sim_max_step=20):
        self.p_n = policy_network
        self.v_n = value_network
        self.env = custom_env
        self.c_puct = c_puct
        self.state_dim_mode = state_dim_mode
        self.graph_attr = self.env.T_graph
        self.threshold = threshold
        self.num_expand_action = num_expand_action
        self.lamda = lamda
        self.sim_num = sim_num
        self.mrs = mrs
        self.sim_max_step = sim_max_step

    def search(self, initial_state, epoch):
        root_node = Node(initial_state, 1)
        i = 1
        while True:
            node = root_node
            path = [node]
            # Selection
            while not node.is_leaf_node():
                action, node = self.select_child(node)
                path.append(node)
            # Expansion
            if not node.is_terminal:
                new_node = self.expand_node(node)
                if len(new_node.children)!=0:
                    action, child_node = self.select_child(new_node)
                    path.append(child_node)
                    node = child_node
                else:
                    break
            else:
                new_node = node
            # Simulation
            if len(new_node.children) == 1:  # Only one action to choose from does not need to be simulated
                value = self.get_state_value(node.state)
                if i == 1:
                    break
            else:
                #value = self.simulate(node)
                value = self.simulate_mod(node)
            # Backpropagation
            self.backpropagate(path, value)

            if len(root_node.children) != 0:
                if i>=epoch:
                    break
            i += 1

        if len(root_node.children) != 0:
            best_child = max(root_node.children.values(), key=lambda child: child.visit_count)
        else:
            best_child = root_node
        return best_child

    def select_child(self, node):
        scores = {}
        # N = sum(child.visit_count for child in node.children.values())
        for action_id, child_node in node.children.items():
            # scores[action_id] = child_node.total_value + self.c_puct/(1 + child_node.visit_count)*child_node.policy_pro
            if child_node.visit_count == 0:
                v_c_c = 0.001
                f_n_tv = 1
            else:
                v_c_c = child_node.visit_count
                f_n_tv = node.total_value
            f_n_v = node.v_n_value
            # scores[action_id] = child_node.total_value/(f_n_v*v_c_c) + math.sqrt(math.log(node.visit_count+1) / v_c_c)
            scores[action_id] = child_node.total_value/(f_n_tv*v_c_c) + self.c_puct / (v_c_c+1) * child_node.policy_pro

        max_ac = max(scores, key=scores.get) # Get the action with the highest probability.
        max_child_node = node.children[max_ac] # Get the child node with the largest score.
        return max_ac, max_child_node

    def expand_node(self, node):
        temp_state = copy.deepcopy(node.state)
        des_candidate_ac_pro, des_candidate_ac_id, last_des_candidate_ac_pro, last_des_candidate_ac_id = self.get_possible_actions(temp_state)
        node.v_n_value = self.get_state_value(node.state)
        flag_reward = 0
        for i in range(des_candidate_ac_pro.shape[0]):
            temp_action = des_candidate_ac_id[i]
            temp_action_pro = des_candidate_ac_pro[i]
            new_state, reward, done = self.get_next_state(temp_state, temp_action)
            if reward != 0:
                flag_reward = 1
                new_node = Node(new_state, policy_pro=temp_action_pro, parent=node)
                new_node.action = temp_action
                new_node.reward = reward
                if done == 1:
                    new_node.is_terminal = True
                node.children[temp_action] = new_node

        if flag_reward == 0:
            for i in range(last_des_candidate_ac_pro.shape[0]):
                temp_action = last_des_candidate_ac_id[i]
                temp_action_pro = last_des_candidate_ac_pro[i]
                new_state, reward, done = self.get_next_state(temp_state, temp_action)
                if reward != 0:
                    new_node = Node(new_state, policy_pro=temp_action_pro, parent=node)
                    new_node.action = temp_action
                    new_node.reward = reward
                    if done == 1:
                        new_node.is_terminal = True
                    node.children[temp_action] = new_node
        return node

    def simulate(self, node):
        temp_state = node.state
        initial_reward = node.reward
        total_q = 0
        iteration = 0

        while iteration<self.sim_num:
            iteration += 1
            new_state = temp_state
            temp_mrs = 0
            step = 0
            each_sim_path_reward = initial_reward
            while True:
                step += 1
                des_candidate_ac_pro, des_candidate_ac_id,_,_ = self.get_possible_actions(new_state)
                nor_des_candidate_ac_pro = des_candidate_ac_pro/np.sum(des_candidate_ac_pro)
                temp_action = np.random.choice(des_candidate_ac_id, replace=True, p=nor_des_candidate_ac_pro)
                new_state, reward, done = self.get_next_state(new_state, temp_action)
                each_sim_path_reward += reward*(self.lamda**step)
                if done == 1:
                    break
                if reward == 0:
                    temp_mrs += 1
                else:
                    temp_mrs = 0
                if temp_mrs > self.mrs:
                    break
            each_sim_path_q = each_sim_path_reward  # Please modify
            total_q += each_sim_path_q
        q = total_q/self.sim_num
        return q

    def simulate_mod(self, node):
        temp_state = node.state
        initial_reward = node.reward
        v = self.get_state_value(temp_state)
        total_q = 0
        iteration = 0

        while iteration < self.sim_num:
            iteration += 1
            new_state = temp_state
            temp_mrs = 0
            step = 0
            each_sim_path_reward = initial_reward
            while True:
                step += 1
                des_candidate_ac_pro, des_candidate_ac_id, _, _ = self.get_possible_actions(new_state)
                nor_des_candidate_ac_pro = des_candidate_ac_pro / np.sum(des_candidate_ac_pro)
                temp_action = np.random.choice(des_candidate_ac_id, replace=True, p=nor_des_candidate_ac_pro)
                new_state, reward, done = self.get_next_state(new_state, temp_action)
                each_sim_path_reward += reward * (self.lamda ** step)
                if step >= self.sim_max_step:
                    v_temp = self.get_state_value(new_state)
                    each_sim_path_reward += v_temp
                    break
                if done == 1:
                    break
            each_sim_path_q = (each_sim_path_reward + v)/2 # Please modify
            # each_sim_path_q = each_sim_path_reward  # Please modify
            total_q += each_sim_path_q
        q = total_q / self.sim_num
        return q

    def backpropagate(self, path, q):
        for node in path:
            node.visit_count += 1
            if isinstance(q, th.Tensor):
                q = q.item()
            node.total_value += q

    def get_possible_actions(self, state):
        temp_state = np.array([state])
        policy = self.p_n(temp_state)[0].squeeze(0)
        sorted_logits_desc, sorted_indices_desc = th.sort(policy, descending=True)
        sorted_logits_desc_np = sorted_logits_desc.to('cpu').detach().numpy()
        sorted_indices_desc_np = sorted_indices_desc.to('cpu').detach().numpy()

        if sorted_logits_desc_np[sorted_logits_desc_np > self.threshold].shape[0]<self.num_expand_action:
            sorted_logits_desc_np_threshold = sorted_logits_desc_np[:self.num_expand_action]
            sorted_indices_desc_np_threshold = sorted_indices_desc_np[:self.num_expand_action]
        else:
            sorted_logits_desc_np_threshold = sorted_logits_desc_np[sorted_logits_desc_np > self.threshold]
            num_p_a = sorted_logits_desc_np_threshold.shape[0]
            sorted_indices_desc_np_threshold = sorted_indices_desc_np[:num_p_a]

        max_pro = sorted_logits_desc_np_threshold[0]
        min_pro = sorted_logits_desc_np_threshold[-1]

        mid_pro = (max_pro+min_pro)/2

        pre_sorted_logits_desc_np_threshold =  sorted_logits_desc_np_threshold[sorted_logits_desc_np_threshold >= mid_pro]
        num_p_a = pre_sorted_logits_desc_np_threshold.shape[0]
        pre_sorted_indices_desc_np_threshold = sorted_indices_desc_np_threshold[:num_p_a]

        last_sorted_logits_desc_np_threshold = sorted_logits_desc_np_threshold[sorted_logits_desc_np_threshold < mid_pro]
        last_sorted_indices_desc_np_threshold = sorted_indices_desc_np[num_p_a:]

        return pre_sorted_logits_desc_np_threshold, pre_sorted_indices_desc_np_threshold, last_sorted_logits_desc_np_threshold, last_sorted_indices_desc_np_threshold

    def get_state_value(self, state):
        v = self.v_n(np.array([state]))[0][0]
        return v

    def get_next_state(self, state, action):
        dynamic_info = state.dstdata["NodeAttr"][:, self.state_dim_mode:]
        self.env.reset()
        mystate = copy.deepcopy(self.graph_attr)
        mystate.dstdata["NodeAttr"] = th.concat([mystate.dstdata["NodeAttr"], dynamic_info], dim=1)

        dynamic_graph = copy.deepcopy(state)
        dynamic_graph.dstdata["NodeAttr"] = state.dstdata["NodeAttr"][:, self.state_dim_mode:]
        self.env.mode_state(mystate, dynamic_graph)

        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

def Perform_action(state, action, env, state_dim_mode):
    dynamic_info = state.dstdata["NodeAttr"][:, state_dim_mode:]
    env.reset()
    mystate = copy.deepcopy(env.T_graph)
    mystate.dstdata["NodeAttr"] = th.concat([mystate.dstdata["NodeAttr"], dynamic_info], dim=1)

    dynamic_graph = copy.deepcopy(state)
    dynamic_graph.dstdata["NodeAttr"] = state.dstdata["NodeAttr"][:, state_dim_mode:]
    env.mode_state(mystate, dynamic_graph)

    next_state, reward, done, _ = env.step(action)
    return next_state, reward, done

def get_final_action(num_nodes, action_id):
    serial_tensor = np.arange(0, num_nodes * tool_num * operation_num).reshape(
        (num_nodes, tool_num, operation_num))

    indices = np.where(serial_tensor == action_id)
    action = [indices[0].item(), indices[1].item(), indices[2].item()]
    return action

def translate(action, max_q):
    node_id = action[0]
    node_real_id = node_id

    tool_type = tool_lib[action[1]]
    process_type = process_lib[action[2]]
    result = tool_type + ' ' + process_type + ' Feature_' + str(node_real_id)

    return result

def calculate_real_time(action_0, action_1, action_2, time_matrix_origin):
    machining_time = time_matrix_origin[action_0][action_1][action_2]
    if machining_time == 0:
        machining_time = th.max(time_matrix_origin[action_0,:,action_2])
    return machining_time.to('cpu')

def calculate_prepare_time(dis_matrix_org, feature_id_ac, env_process_ac_id, env_tool_ac_id, temp_previous_feature_id,
                                temp_previous_process_st_id, temp_previous_tool_st):
    previous_feature_id = temp_previous_feature_id
    t_2 = dis_matrix_org[previous_feature_id, feature_id_ac]/fate_speed
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

def calculate_time(current_ac, last_ac, dis_matrix, time_matrix_origin, cutting_time = 1):
    all_time = 0
    real_action_feature = current_ac[0]
    real_action_semantic = current_ac[2]
    real_action_tool = current_ac[1]

    real_action_feature_last = last_ac[0]
    real_action_semantic_last = last_ac[2]
    real_action_tool_last = last_ac[1]
    if cutting_time == 1:
        real_time =calculate_real_time(real_action_feature, real_action_tool, real_action_semantic, time_matrix_origin)
        all_time += real_time
    t_pre = calculate_prepare_time(dis_matrix, real_action_feature, real_action_semantic, real_action_tool,
                                   real_action_feature_last, real_action_semantic_last, real_action_tool_last)

    all_time += t_pre
    return all_time

def write_txt(all_actions, all_max_vs, file_name, all_time=0, all_reward=0, info="None", computation_time=.0):
    with open(file_name,"w") as f:
        f.write(info + '\n')
        for i in range(len(all_actions)):
            result = translate(all_actions[i], all_max_vs[i])
            f.write(str(i+1) + ': ' + result + '\n')
        f.write('machine_time: ' + str(all_time) + '\n')
        # f.write('all_reward: ' + str(all_reward) + '\n')
        f.write('computation_time: ' + str(computation_time) + '\n')

def get_action_manual(temp_state, state_dim_mode, num_nodes, tool_num, operation_num):
    fully_finish = temp_state.dstdata["NodeAttr"][:, state_dim_mode+33]
    indices_feature = th.argwhere(fully_finish==0)
    action = 0
    op_id = 0
    tool_id = 0
    serial_tensor = th.arange(0, num_nodes * tool_num * operation_num).reshape(
        (num_nodes, tool_num, operation_num))
    indices_feature = indices_feature.reshape(-1)
    for i in range(indices_feature.shape[0]):
        feature_id = indices_feature[i].to('cpu').numpy()
        static_attr = temp_state.dstdata["NodeAttr"][feature_id, :state_dim_mode-2]
        static_reshape = th.reshape(static_attr, [23, 10])
        need_machining_operation = th.sum(static_reshape, 0)
        indices_operation = th.argwhere(need_machining_operation>0).reshape(-1)
        dynamic_attr = temp_state.dstdata["NodeAttr"][feature_id, state_dim_mode:]
        current_machining_operation = dynamic_attr[23:33]
        for j in range(indices_operation.reshape(-1).shape[0]):
            op_id = indices_operation[j].to('cpu').numpy()
            if current_machining_operation[op_id] == 0:
                static_reshape = static_reshape.to('cpu').numpy()
                static_reshape[static_reshape == 0] = np.nan
                tool_id = np.nanargmin(static_reshape[:, op_id])
                break
        action = serial_tensor[feature_id, tool_id, op_id]
        break
    return action

def get_next_state_manual(state, action, state_dim_mode, env, graph_attr):
    dynamic_info = state.dstdata["NodeAttr"][:, state_dim_mode:]
    env.reset()
    mystate = copy.deepcopy(graph_attr)
    mystate.dstdata["NodeAttr"] = th.concat([mystate.dstdata["NodeAttr"], dynamic_info], dim=1)

    dynamic_graph = copy.deepcopy(state)
    dynamic_graph.dstdata["NodeAttr"] = state.dstdata["NodeAttr"][:, state_dim_mode:]
    env.mode_state(mystate, dynamic_graph)

    next_state, reward, done, _ = env.step(action)
    return next_state, reward, done


if __name__ == "__main__":
    '''
       Part: ZH_test11_with_cons_all_time.dgl, policy: WoPPO_test_11
       Part: ZH_test3_with_cons_all_time.dgl, policy: WoPPO_test_3

       Part: ZH_test18_with_cons_all_time.dgl, policy: WoPPO_test_18_only_rough
       Part: ZH_test18_with_cons_all_time.dgl, policy: WoPPO_test_18_without_rough
    '''
    part_file = r"...\DRL_MPP\data\ZH_test11_with_cons_all_time.dgl"
    policy_file = r"...\DRL_MPP\exps\WoPPO_test_11\ppo\policy.pth"

    tau = 0.8
    max_iterations = 100
    sim_num = 20
    sim_max_step = 20

    mrs = 0
    num_expand_action = 20

    '''
        Consider all machining operations: WorldofPPv3_1_predict
        Only consider roughing: WorldofPPv3_6_predict
        Don't consider roughing: WorldofPPv3_7_predict
    '''

    PVn = PolicyandValueNet(policy_file,
                            "Enviroments/WorldofPPv3_7_predict",
                            part_file,
                            tau)

    graphs, _ = dgl.load_graphs(part_file)
    graph = graphs[0]
    dis_matrix = graph.dstdata["Dis_matrix_org"]
    org_time_vector = graph.dstdata["Time_origin"]
    time_matrix_origin = org_time_vector.reshape((org_time_vector.shape[0], tool_num, operation_num))
    num_nodes = graph.num_nodes()

    time_start = time.time()
    a_a = PVn.get_actor_network()
    a_c = PVn.get_critic_network()
    par = PVn.get_paras()
    env = PVn.get_env()
    iteration = 0
    temp_state = env.reset()
    pre_action = None

    all_actions = []
    all_values = []
    all_time = 0
    all_time_real = 0
    all_reward = 0
    while True:
        Tree = MCTs(a_a, a_c, env, sim_num=sim_num, mrs=mrs, num_expand_action=num_expand_action, sim_max_step=sim_max_step)
        temp_child = Tree.search(temp_state, max_iterations)
        temp_v = temp_child.total_value
        temp_done = temp_child.is_terminal
        temp_ac = temp_child.action

        if (pre_action == temp_ac)|(temp_ac == None):
            temp_ac = get_action_manual(temp_state, 232, num_nodes, 23, 10).item()
            temp_state, reward, done = get_next_state_manual(temp_state, temp_ac, 232, env, env.T_graph)
            temp_done = done
        else:
            temp_state = temp_child.state

        trans_action = get_final_action(num_nodes, temp_ac)
        res = translate(trans_action, temp_v)

        all_actions.append(trans_action)
        all_values.append(temp_v)
        if temp_child.reward != 0:
            if iteration==0:
                pre_trans_action = copy.deepcopy(trans_action)
            t_real = calculate_time(trans_action, pre_trans_action, dis_matrix,time_matrix_origin, cutting_time=1)
            t = (1 - temp_child.reward) * (1 + tau)
            all_time += t
            all_time_real += t_real
            all_reward += temp_child.reward
        iteration += 1
        # temp_state, reward, done = Perform_action(temp_state, temp_ac, env, 232)
        pre_action = temp_ac
        pre_trans_action = copy.deepcopy(trans_action)

        print(iteration)
        print(trans_action)
        print(res)
        print("max_v: " + str(temp_v))

        if temp_done == True:
            print("done")
            break
    print("all_time: " + str(all_time))
    print("all_reward: " + str(all_reward))
    print("all_time_real: " + str(all_time_real))

    file_name = r"...\DRL_MPP\results\mcts_results\test_11_results.txt"

    info = "Strict the simulation time in one episode to "+str(sim_max_step) + ". Max iterations: "+str(max_iterations)+". Simulate number: "+str(sim_num)
    time_end = time.time()
    computation_time = time_end - time_start
    write_txt(all_actions, all_values, file_name, all_time_real, all_reward, info, computation_time)

    print("Compute time: "+ str(computation_time))