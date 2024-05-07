import dgl
import torch as th
import os
import copy
import numpy as np
import csv
import math
import networkx as nx
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


ToolDias = [50,30,30,30,25,25,25,20,20,20,18,18,18,16,16,16,12,12,12,10,10,8,8]
ToolCorners = [1,5,3,1,5,3,1,5,3,1,5,3,1,5,3,1,5,3,1,3,1,3,1]
Features = ["Plane", "Pocket", "Hole", "Profile", "Slot"]

blank_allowance = 5
rough_allowance=1.5
semi_finish_allowance=0.5
finish_allowance=0.05

rough_v_feed = 4000  # 粗加工进给速度， mm/min
semi_finish_v_feed = 3000  # 半精加工进给速度， mm/min
finish_v_feed = 3000  # 精加工进给速度， mm/min

approach_speed = 2000  # 进刀速度 mm/min
retract_speed = 3000  # 退刀速度 mm/min
fate_speed = 10000  # 走空刀 mm/min

safe_place = 10 #安全平面 mm
start_end = 3 #进刀和退刀起点 mm

temp_constraint = 1 #临时控制变量，等于0，不限制可行解范围，可行解范围大；等于1，限制可行解范围，可行解范围小，原文中给1

def one_hot(num, total_num):
    vector = th.zeros(total_num)
    vector[int(num)] = 1
    return vector

def read_txt(txt_name, whether_read_feature_id=0):
    temp_nodes_attr = []
    temp_edges_attr = []

    file = open(txt_name)
    lines = file.readlines()
    for line in lines:
        line = line.strip().strip("\n")
        line = line.split(" ")
        temp_node_attr = []
        temp_edge_attr = []
        if line[0] == "Node":
            if whether_read_feature_id == 1:
                temp_node_attr.append(int(line[2].split(":")[1]))  # 特征实际的id
            temp_node_attr.append(int(line[3].split(":")[1]))  # featuretype
            temp_node_attr.append(int(line[4].split(":")[1]))  # AngleCode
            temp_node_attr.append(float(line[5].split(":")[1]))  # XBoundingBox
            temp_node_attr.append(float(line[6].split(":")[1]))  # YBoundingBox
            temp_node_attr.append(float(line[7].split(":")[1]))  # ZBoundingBox
            temp_node_attr.append(float(line[8].split(":")[1]))  # XBottomCenterCoordinate
            temp_node_attr.append(float(line[9].split(":")[1]))  # YBottomCenterCoordinate
            temp_node_attr.append(float(line[10].split(":")[1]))  # ZBottomCenterCoordinate

            temp_node_attr.append(float(line[11].split(":")[1]))  # Flength
            temp_node_attr.append(float(line[12].split(":")[1]))  # Fwidth
            temp_node_attr.append(float(line[13].split(":")[1]))  # Fdepth

            temp_node_attr.append(float(line[14].split(":")[1]))  # NormalX
            temp_node_attr.append(float(line[15].split(":")[1]))  # NormalY
            temp_node_attr.append(float(line[16].split(":")[1]))  # NormalZ
            temp_node_attr.append(float(line[17].split(":")[1]))  # Bottomthickness
            temp_node_attr.append(float(line[18].split(":")[1]))  # Boundthickness
            corner0 = float(line[19].split(":")[1])
            corner1 = float(line[20].split(":")[1])
            if float(line[19].split(":")[1]) < 4:  # ConerR0，转角必须都大于4
                corner0 = 4
            if float(line[19].split(":")[1]) < 0.1:  # ConerR0，没有转角
                corner0 = 0
            if float(line[19].split(":")[1]) > 12:
                corner0 = 0
            if float(line[20].split(":")[1]) < 4:  # ConerR1，转角必须都大于4
                corner1 = 4
            if float(line[20].split(":")[1]) < 0.1:  # ConerR1，没有转角
                corner1 = 0
            if float(line[20].split(":")[1]) > 12:
                corner1 = 0
            temp_node_attr.append(corner0)  # ConerR0
            temp_node_attr.append(corner1)  # ConerR1
            temp_node_attr.append(float(line[21].split(":")[1]))  # BottomR0
            temp_node_attr.append(float(line[22].split(":")[1]))  # BottomR1
            temp_node_attr.append(float(line[23].split(":")[1]))  # BottomArea
            temp_node_attr.append(float(line[24].split(":")[1]))  # MinChannelDis
            BottomPrecision = int(line[25].split(":")[1])
            if BottomPrecision == 2:
                BottomPrecision = 3
            temp_node_attr.append(BottomPrecision)  # BottomPrecision
            BoundingPrecision = int(line[26].split(":")[1])
            if BoundingPrecision == 2:
                BoundingPrecision = 3
            temp_node_attr.append(BoundingPrecision)  # BoundingPrecision

            temp_node_attr.append(float(line[27].split(":")[1]))  # MaxChannelDis
            temp_node_attr.append(int(line[28].split(":")[1]))  # CornerNum
            temp_nodes_attr.append(temp_node_attr)

        elif line[0] == "Edge":
            temp_edge_attr.append(int(line[1].split(":")[1]))  # source
            temp_edge_attr.append(int(line[2].split(":")[1]))  # target
            edge_type = int(line[3].split(":")[1])
            temp_edge_attr.append(edge_type)  # edgetype
            temp_edge_attr.append(int(line[4].split(":")[1]))  # edgedesigntype
            if edge_type != 3:
                temp_edges_attr.append(temp_edge_attr)

    return temp_nodes_attr, temp_edges_attr


'''
Feature type：
    1:Plane 2:Pocket 3:Hole 4:Profile 5:Slot(without corner)
Open closs code:
    1.全=90 10000000
    2.全<90 01000000
    3.全>90 001000000
    4.>90 + <90 00010000
    5.>90 + =90 00001000
    6.<90 + =90 00000100
    7.>90 + <90 + =90 00000010
    8.无开闭角 0000001 （平面或者孔或者轮廓）#8
Precsion: 3:精度高需要半精, 2:精度低需要半精
Thickness: 1:薄壁, 0:非薄壁
'''

class GetAttrFromNodeEdge():
    def __init__(self,temp_nodes_attr, temp_edges_attr):
        self.temp_nodes_attr = temp_nodes_attr
        self.temp_edges_attr = temp_edges_attr
        self.each_operation_tool_num = 3 #每个操作最多each_operation_tool_num把可行刀具

    '''
    For debug
    '''
    def write_to_cvs(self, file_name, matrix):
        x_axis = [" ", "大开粗", "半侧壁", "半五轴侧壁", "半底面", "精侧壁", "精五轴侧壁", "精底面", "精转角", "精五轴转角", "小开粗"]
        y_axis = ["D50R1", "D30R5", "D30R3", "D30R1", "D25R5", "D25R3", "D25R1", "D20R5", "D20R3", "D20R1", "D18R5",
                  "D18R3", "D18R1", "D16R5", "D16R3", "D16R1", "D12R5", "D12R3", "D12R1", "D10R3", "D10R1",
                  "D8R3", "D8R1"]
        with open(file_name, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(x_axis)
            for i in range(matrix.shape[0]):
                row_all=[y_axis[i]]
                row = matrix[i][:]
                for r in row:
                    row_all.append(str(r.item()))
                writer.writerow(row_all)

    def convert_nodes_attr(self):
        all_node_matrix_tensor = th.zeros((1, 23, 10))
        all_node_time_matrix_tensor = th.zeros((1, 23, 10))
        all_node_nor_time_matrix_tensor = th.zeros((1, 23, 10))
        all_node_Coor_tensor = th.zeros((1, 3))
        for i in range(len(self.temp_nodes_attr)):
            temp_node_attr = self.temp_nodes_attr[i]
            self.AngleCode = temp_node_attr[1]
            self.XBoundingBox = temp_node_attr[2]
            self.YBoundingBox = temp_node_attr[3]
            self.ZBoundingBox = temp_node_attr[4]
            XYZBoundingBox = th.tensor([self.XBoundingBox, self.YBoundingBox, self.ZBoundingBox])
            self.Flength = temp_node_attr[8]
            self.Fwidth = temp_node_attr[9]
            self.Fdepth = temp_node_attr[10]
            self.min_dis = temp_node_attr[21]
            if self.min_dis > min(self.Flength, self.Fwidth):
                self.min_dis = min(self.Flength, self.Fwidth)
            self.max_dis = temp_node_attr[24]
            if self.max_dis > min(self.Flength, self.Fwidth):
                self.max_dis = min(self.Flength, self.Fwidth)
            self.CornerR = max(temp_node_attr[16], temp_node_attr[17])
            self.BottomR = max(temp_node_attr[18], temp_node_attr[19])
            self.BottomPrecision = temp_node_attr[22]
            self.BoundPrecision = temp_node_attr[23]
            self.Bottomthickness = temp_node_attr[14]
            self.Boundthickness = temp_node_attr[15]
            self.CornerNum = temp_node_attr[25]

            if temp_node_attr[0] == 1: # Plane
                node_matrix = self.get_plane_feasible_solutions_matrix()
                node_time_matrix = self.estimate_plane_machining_time(node_matrix)
            elif temp_node_attr[0] == 2: # Pocket
                if self.BottomR >= 5:
                    self.BottomR = 5
                elif self.BottomR >= 3:
                    self.BottomR = 3
                else:
                    self.BottomR = 1
                node_matrix = self.get_pocket_feasible_solutions_matrix()
                node_time_matrix = self.estimate_pocket_machining_time(node_matrix)
            elif temp_node_attr[0] == 3: # Hole
                node_matrix = self.get_hole_feasible_solutions_matrix()
                node_time_matrix = self.estimate_hole_machining_time(node_matrix)
            elif temp_node_attr[0] == 4: # Profile
                node_matrix = self.get_profile_feasible_solutions_matrix()
                node_time_matrix = self.estimate_profile_machining_time(node_matrix)
            else: # Slot(without corner)
                if self.BottomR >= 5:
                    self.BottomR = 5
                elif self.BottomR >= 3:
                    self.BottomR = 3
                else:
                    self.BottomR = 1
                node_matrix = self.get_slot_feasible_solutions_matrix()
                node_time_matrix = self.estimate_slot_machining_time(node_matrix)
            node_time_matrix_copy = copy.deepcopy(node_time_matrix)
            node_normal_time_matrix = self.normalize_time(node_time_matrix_copy)
            node_matrix_tensor = th.tensor(node_matrix)
            node_time_matrix_tensor = th.tensor(node_time_matrix)
            node_normal_time_matrix_tensor = th.tensor(node_normal_time_matrix)
            '''
            debug
            '''
            feature_name_id = str(i)+"_"+Features[temp_node_attr[0]-1]

            ##写入cvs
            self.write_to_cvs(r"F:\ZH\AllDRLPP_MC\Code\Debug_txt\Process_Matrix_2\\" + feature_name_id + ".csv", node_matrix_tensor)
            self.write_to_cvs(
                r"F:\ZH\AllDRLPP_MC\Code\Debug_txt\Process_Matrix_2\\" + feature_name_id + "_process_time.csv",
                node_time_matrix_tensor)
            self.write_to_cvs(
                r"F:\ZH\AllDRLPP_MC\Code\Debug_txt\Process_Matrix_2\\" + feature_name_id + "_nor_process_time.csv",
                node_normal_time_matrix_tensor)

            if i == 0:
                all_node_matrix_tensor[0] = node_matrix_tensor
                all_node_time_matrix_tensor[0] = node_time_matrix_tensor
                all_node_nor_time_matrix_tensor[0] = node_normal_time_matrix_tensor
                all_node_Coor_tensor[0] = XYZBoundingBox
            else:
                all_node_matrix_tensor = th.concat((all_node_matrix_tensor, node_matrix_tensor.unsqueeze(0)), dim=0)
                all_node_time_matrix_tensor = th.concat((all_node_time_matrix_tensor, node_time_matrix_tensor.unsqueeze(0)), dim=0)
                all_node_nor_time_matrix_tensor = th.concat((all_node_nor_time_matrix_tensor, node_normal_time_matrix_tensor.unsqueeze(0)), dim=0)
                all_node_Coor_tensor =th.concat((all_node_Coor_tensor, XYZBoundingBox.unsqueeze(0)), dim=0)

        dis_matrix_nor, dis_matrix_org = self.get_distance_matrix(all_node_Coor_tensor)
        dis_matrix_nor_tensor = th.tensor(dis_matrix_nor)
        dis_matrix_org_tensor = th.tensor(dis_matrix_org)

        return all_node_matrix_tensor, all_node_time_matrix_tensor, all_node_nor_time_matrix_tensor, dis_matrix_nor_tensor, dis_matrix_org_tensor

    def normalize_edges_attr(self):
        temp_edges_attr_tensor = th.tensor(self.temp_edges_attr)
        indices = th.argwhere((temp_edges_attr_tensor[:, -1] == 1) | (temp_edges_attr_tensor[:, -1] == 2)).squeeze(
            -1)
        temp_edges_attr_tensor = th.index_select(temp_edges_attr_tensor, 0, indices)
        edges_attr_tensor = th.zeros(2)
        for i in range(temp_edges_attr_tensor.shape[0]):
            attr = temp_edges_attr_tensor[i]
            temp_edge_attr_tensor = attr[0: 2]
            temp_edge_attr_tensor = th.concat([temp_edge_attr_tensor, one_hot(attr[-2] - 1, 3)],
                                              0)  # 主从，并列，无关系
            temp_edge_attr_tensor = th.concat([temp_edge_attr_tensor, one_hot(attr[-1] - 1, 2)], 0)  # 基准，无关系

            if i == 0:
                edges_attr_tensor = temp_edge_attr_tensor.unsqueeze(0)
            else:
                edges_attr_tensor = th.concat([edges_attr_tensor, temp_edge_attr_tensor.unsqueeze(0)], 0)

        return edges_attr_tensor

    def judge_rough(self, ToolDias, node_matrix, j, constraint=temp_constraint): # constraint=1表面要缩小解空间, 只有两把刀具，constraint=0表示正常
        flag = 0
        for i in range(len(ToolDias)):
            if (ToolDias[i] <= self.max_dis - 2 * rough_allowance) & (ToolDias[i] < 50):
                if ToolCorners[i] == self.BottomR:
                    node_matrix[i][j] = 1
                    flag += 1
            if constraint == 1:
                if flag>=self.each_operation_tool_num:
                    break
        return node_matrix

    def judge_rough_hole(self, ToolDias, node_matrix, j, constraint=temp_constraint):
        flag = 0
        for i in range(len(ToolDias)):
            if (ToolDias[i] <= self.max_dis - 2 * rough_allowance) & (ToolDias[i] < 50):
                if ToolCorners[i] == 1:
                    node_matrix[i][j] = 1
                    flag += 1
            if constraint == 1:
                if flag >= self.each_operation_tool_num:
                    break
        return node_matrix

    def judge_semi(self, ToolDias, node_matrix, j, constraint=temp_constraint):
        flag = 0
        for i in range(len(ToolDias)):
            if (ToolDias[i] <= self.min_dis - 2 * semi_finish_allowance) & (ToolDias[i] < 30):
                if ToolCorners[i] == self.BottomR:
                    node_matrix[i][j] = 1
                    flag += 1
                if constraint == 1:
                    if flag >= self.each_operation_tool_num:
                        break
        if np.sum(node_matrix[:, j]) == 0:
            for i in range(len(ToolDias)):
                if (ToolDias[i] <= self.min_dis - 2 * semi_finish_allowance) & (ToolDias[i] < 30):
                    if ToolCorners[i] < self.BottomR:
                        node_matrix[i][j] = 1
                        flag += 1
                    if constraint == 1:
                        if flag >= self.each_operation_tool_num:
                            break
        return node_matrix

    def judge_semi_hole(self, ToolDias, node_matrix, j, constraint=temp_constraint):
        flag = 0
        for i in range(len(ToolDias)):
            if (ToolDias[i] <= self.min_dis - 2 * semi_finish_allowance) & (ToolDias[i] < 30):
                if ToolCorners[i] == 1:
                    node_matrix[i][j] = 1
                    flag += 1
            if constraint == 1:
                if flag >= self.each_operation_tool_num:
                    break
        return node_matrix

    def judge_finish(self, ToolDias, node_matrix, j, constraint=temp_constraint):
        flag = 0
        for i in range(len(ToolDias)):
            if (ToolDias[i] <= self.min_dis - 2 * finish_allowance) & (ToolDias[i] < 30):
                if ToolCorners[i] == self.BottomR:
                    node_matrix[i][j] = 1
                    flag += 1
            if constraint == 1:
                if flag >= self.each_operation_tool_num:
                    break
        if np.sum(node_matrix[:, j]) == 0:
            for i in range(len(ToolDias)):
                if (ToolDias[i] <= self.min_dis - 2 * finish_allowance) & (ToolDias[i] < 30):
                    if ToolCorners[i] < self.BottomR:
                        node_matrix[i][j] = 1
                        flag += 1
                if constraint == 1:
                    if flag >= self.each_operation_tool_num:
                        break
        return node_matrix

    def judge_finish_hole(self, ToolDias, node_matrix, j, constraint=temp_constraint):
        flag = 0
        for i in range(len(ToolDias)):
            if (ToolDias[i] <= self.min_dis - 2 * finish_allowance) & (ToolDias[i] < 30):
                if ToolCorners[i] == 1:
                    node_matrix[i][j] = 1
                    flag += 1
            if constraint == 1:
                if flag >= self.each_operation_tool_num:
                    break
        return node_matrix

    def judge_finish_corner(self, ToolDias, node_matrix, j, constraint=temp_constraint):
        flag = 0
        for i in range(len(ToolDias)):
            if ToolDias[i] <= self.CornerR*2:
                if ToolCorners[i] == self.BottomR:
                    node_matrix[i][j] = 1
                    flag += 1
            if constraint == 1:
                if flag >= self.each_operation_tool_num:
                    break
        if np.sum(node_matrix[:, j]) == 0:
            for i in range(len(ToolDias)):
                if ToolDias[i] <= self.CornerR*2:
                    if ToolCorners[i] < self.BottomR:
                        node_matrix[i][j] = 1
                        flag += 1
                if constraint == 1:
                    if flag >= self.each_operation_tool_num:
                        break
        return node_matrix

    def judge_second_rough(self, ToolDias, node_matrix, j, constraint=temp_constraint):
        flag = 0
        for i in range(len(ToolDias)):
            if (ToolDias[i] <= self.min_dis - 2 * rough_allowance) & (ToolDias[i] < 50):
                if ToolCorners[i] == self.BottomR:
                    node_matrix[i][j] = 1
                    flag += 1
            if constraint == 1:
                if flag >= self.each_operation_tool_num:
                    break
        return node_matrix

    def get_pocket_feasible_solutions_matrix(self):
        node_matrix = np.zeros((23, 10))  # 23 cutting tools and 10 machining operations
        # 根据精度，薄壁，开闭角判断是需要进行必要的加工操作
        Operations = np.zeros(10)
        Operations[0] = 1  # 第一次开粗
        Operations[9] = 1  # 第二次开粗，在奖励函数中判断是否需要二次开粗
        Operations[6] = 1  # 精底面

        if (self.BottomPrecision == 3)|(self.Bottomthickness == 1):  # 底面需要半精加工
            Operations[3] = 1  # 半精底面

        if (self.AngleCode!=1)&(self.AngleCode!=8):
            Operations[5] = 1 # 需要用五轴铣精侧壁
            Operations[8] = 1 # 需要用五轴铣精转角（判断是否需要需要在奖励函数中判断）
            if (self.BoundPrecision == 3)|(self.Boundthickness == 1):
                Operations[2] = 1  # 侧壁需要半精加工
        else:
            Operations[4] = 1  # 需要用三轴铣精侧壁
            Operations[7] = 1  # 需要用三轴铣精转角（判断是否需要需要在奖励函数中判断）
            if (self.BoundPrecision == 3) | (self.Boundthickness == 1):
                Operations[1] = 1  # 侧壁需要半精加工

        for j in range(Operations.shape[0]):
            if Operations[j]==1:
                if (j == 0): # 粗
                    node_matrix = self.judge_rough(ToolDias, node_matrix, j)
                elif (j == 1)|(j == 2)|(j == 3): # 半侧壁，五轴半侧壁，半底面
                    node_matrix = self.judge_semi(ToolDias, node_matrix, j)
                elif (j == 4)|(j == 5)|(j == 6): # 精铣侧壁，五轴精铣侧壁，精铣底面
                    node_matrix = self.judge_finish(ToolDias, node_matrix, j)
                elif (j == 7)|(j == 8): # 精转角，五轴精转角
                    node_matrix = self.judge_finish_corner(ToolDias, node_matrix, j)
                elif j == 9:
                    node_matrix = self.judge_second_rough(ToolDias, node_matrix, j)
        return node_matrix

    def get_slot_feasible_solutions_matrix(self):
        node_matrix = np.zeros((23, 10))  # 23 cutting tools and 10 machining operations
        # 根据精度，薄壁，开闭角判断是需要进行必要的加工操作
        Operations = np.zeros(10)
        Operations[0] = 1  # 第一次开粗
        Operations[9] = 1  # 第二次开粗，在奖励函数中判断是否需要二次开粗
        Operations[6] = 1  # 精底面

        if (self.BottomPrecision == 3) | (self.Bottomthickness == 1):  # 底面需要半精加工
            Operations[3] = 1  # 半精底面

        if (self.AngleCode != 1) & (self.AngleCode != 8):
            Operations[5] = 1  # 需要用五轴铣精侧壁
            if (self.BoundPrecision == 3) | (self.Boundthickness == 1):
                Operations[2] = 1  # 侧壁需要半精加工
        else:
            Operations[4] = 1  # 需要用三轴铣精侧壁
            if (self.BoundPrecision == 3) | (self.Boundthickness == 1):
                Operations[1] = 1  # 侧壁需要半精加工

        for j in range(Operations.shape[0]):
            for j in range(Operations.shape[0]):
                if Operations[j] == 1:
                    if (j == 0):  # 粗
                        node_matrix = self.judge_rough(ToolDias, node_matrix, j)
                    elif (j == 1) | (j == 2) | (j == 3):  # 半侧壁，五轴半侧壁，半底面
                        node_matrix = self.judge_semi(ToolDias, node_matrix, j)
                    elif (j == 4) | (j == 5) | (j == 6):  # 精铣侧壁，五轴精铣侧壁，精铣底面
                        node_matrix = self.judge_finish(ToolDias, node_matrix, j)
                    elif j == 9:
                        node_matrix = self.judge_second_rough(ToolDias, node_matrix, j)
            return node_matrix

    def get_plane_feasible_solutions_matrix(self, constraint=temp_constraint):
        node_matrix = np.zeros((23, 10))  # 23 cutting tools and 10 machining operations
        # 根据精度，薄壁，开闭角判断是需要进行必要的加工操作
        Operations = np.zeros(10)
        Operations[0] = 1  # 第一次开粗
        Operations[6] = 1  # 精底面

        if (self.BottomPrecision == 3) | (self.Bottomthickness == 1):  # 底面需要半精加工
            Operations[3] = 1  # 半精底面

        for j in range(Operations.shape[0]):
            for j in range(Operations.shape[0]):
                if Operations[j] == 1:
                    if (j == 0):  # 粗
                        flag = 0
                        for i in range(len(ToolDias)):
                            if ToolCorners[i] == 1:
                                node_matrix[i][j] = 1
                                flag += 1
                            if constraint==1:
                                if flag >= self.each_operation_tool_num:
                                    break
                    elif (j == 1) | (j == 2) | (j == 3):  # 半侧壁，五轴半侧壁，半底面
                        flag = 0
                        for i in range(len(ToolDias)):
                            if ToolCorners[i] == 1:
                                node_matrix[i][j] = 1
                                flag += 1
                            if constraint == 1:
                                if flag >= self.each_operation_tool_num:
                                    break
                    elif (j == 4) | (j == 5) | (j == 6):  # 精铣侧壁，五轴精铣侧壁，精铣底面
                        flag = 0
                        for i in range(len(ToolDias)):
                            if ToolCorners[i] == 1:
                                node_matrix[i][j] = 1
                                flag += 1
                            if constraint == 1:
                                if flag >= self.each_operation_tool_num:
                                    break
            return node_matrix

    def get_profile_feasible_solutions_matrix(self, constraint=temp_constraint):
        node_matrix = np.zeros((23, 10))  # 23 cutting tools and 10 machining operations
        # 根据精度，薄壁，开闭角判断是需要进行必要的加工操作
        Operations = np.zeros(10)
        Operations[0] = 1  # 第一次开粗

        if (self.AngleCode != 1) & (self.AngleCode != 8):
            Operations[5] = 1  # 需要用五轴铣精侧壁
            if (self.BoundPrecision == 3) | (self.Boundthickness == 1):
                Operations[2] = 1  # 侧壁需要半精加工
        else:
            Operations[4] = 1  # 需要用三轴铣精侧壁
            if (self.BoundPrecision == 3) | (self.Boundthickness == 1):
                Operations[1] = 1  # 侧壁需要半精加工

        for j in range(Operations.shape[0]):
            for j in range(Operations.shape[0]):
                if Operations[j] == 1:
                    if (j == 0):  # 粗
                        flag = 0
                        for i in range(len(ToolDias)):
                            if ToolDias[i] < 50:
                                if ToolCorners[i] == 1:
                                    node_matrix[i][j] = 1
                                    flag += 1
                            if constraint == 1:
                                if flag >= self.each_operation_tool_num:
                                    break
                    elif (j == 1) | (j == 2) | (j == 3):  # 半侧壁，五轴半侧壁，半底面
                        flag = 0
                        for i in range(len(ToolDias)):
                            if ToolDias[i] < 30:
                                if ToolCorners[i] == 1:
                                    node_matrix[i][j] = 1
                                    flag += 1
                            if constraint == 1:
                                if flag >= self.each_operation_tool_num:
                                    break
                    elif (j == 4) | (j == 5) | (j == 6):  # 精铣侧壁，五轴精铣侧壁，精铣底面
                        flag = 0
                        for i in range(len(ToolDias)):
                            if ToolDias[i] < 30:
                                if ToolCorners[i] == 1:
                                    node_matrix[i][j] = 1
                                    flag += 1
                            if constraint == 1:
                                if flag >= self.each_operation_tool_num:
                                    break
            return node_matrix

    def get_hole_feasible_solutions_matrix(self):
        node_matrix = np.zeros((23, 10))  # 23 cutting tools and 10 machining operations
        # 根据精度，薄壁，开闭角判断是需要进行必要的加工操作
        Operations = np.zeros(10)
        Operations[0] = 1  # 第一次开粗

        Operations[4] = 1  # 需要用三轴铣精侧壁
        if (self.BoundPrecision == 3) | (self.Boundthickness == 1):
            Operations[1] = 1  # 侧壁需要半精加工

        for j in range(Operations.shape[0]):
            if Operations[j]==1:
                if (j == 0): # 粗
                    node_matrix = self.judge_rough_hole(ToolDias, node_matrix, j)
                elif (j == 1)|(j == 2)|(j == 3): # 半侧壁，五轴半侧壁，半底面
                    node_matrix = self.judge_semi_hole(ToolDias, node_matrix, j)
                elif (j == 4)|(j == 5)|(j == 6): # 精铣侧壁，五轴精铣侧壁，精铣底面
                    node_matrix = self.judge_finish_hole(ToolDias, node_matrix, j)
        return node_matrix

    def estimate_plane_machining_time(self, node_matrix):
        node_time_matrix = np.zeros_like(node_matrix)
        for i in range(node_matrix.shape[0]):
            temp_tool_dia = ToolDias[i]
            for j in range(node_matrix.shape[1]):
                if node_matrix[i][j]==1:
                    if j == 0: # 第一次开粗
                        node_time_matrix[i][j] = self.calculate_roughing_time(temp_tool_dia, 1)
                    elif j == 3: # 半精底面
                        node_time_matrix[i][j] = self.calculate_bottom_finishing_time(temp_tool_dia, 1)
                    elif j == 6: # 精底面
                        node_time_matrix[i][j] = self.calculate_bottom_finishing_time(temp_tool_dia, 0)
        return node_time_matrix

    def estimate_pocket_machining_time(self, node_matrix):
        node_time_matrix = np.zeros_like(node_matrix)
        for i in range(node_matrix.shape[0]):
            temp_tool_dia = ToolDias[i]
            for j in range(node_matrix.shape[1]):
                if node_matrix[i][j] == 1:
                    if j == 0:  # 第一次开粗
                        node_time_matrix[i][j] = self.calculate_roughing_time(temp_tool_dia, 0)
                    elif (j == 1)|(j == 2): # 半精侧壁
                        node_time_matrix[i][j] = self.calculate_bound_finishing_time(temp_tool_dia, 1)
                    elif (j == 4)|(j == 5): # 精侧壁
                        node_time_matrix[i][j] = self.calculate_bound_finishing_time(temp_tool_dia, 0)
                    elif j == 3:  # 半精底面
                        node_time_matrix[i][j] = self.calculate_bottom_finishing_time(temp_tool_dia, 1)
                    elif j == 6:  # 精底面
                        node_time_matrix[i][j] = self.calculate_bottom_finishing_time(temp_tool_dia, 0)
                    elif (j == 7)|(j == 8): # 精转角
                        node_time_matrix[i][j] = self.calculate_corner_finishing_time(temp_tool_dia)
                    elif j == 9: # 第二次开粗
                        node_time_matrix[i][j] = self.calculate_second_roughing_time(temp_tool_dia)
        return node_time_matrix

    def estimate_profile_machining_time(self, node_matrix):
        node_time_matrix = np.zeros_like(node_matrix)
        for i in range(node_matrix.shape[0]):
            temp_tool_dia = ToolDias[i]
            for j in range(node_matrix.shape[1]):
                if node_matrix[i][j] == 1:
                    if j == 0:  # 第一次开粗
                        node_time_matrix[i][j] = self.calculate_roughing_time(temp_tool_dia, 0)
                    elif (j == 1) | (j == 2):  # 半精侧壁
                        node_time_matrix[i][j] = self.calculate_bound_finishing_time(temp_tool_dia, 1)
                    elif (j == 4) | (j == 5):  # 精侧壁
                        node_time_matrix[i][j] = self.calculate_bound_finishing_time(temp_tool_dia, 0)
        return node_time_matrix

    def estimate_hole_machining_time(self, node_matrix): #铣削孔
        node_time_matrix = np.zeros_like(node_matrix)
        for i in range(node_matrix.shape[0]):
            temp_tool_dia = ToolDias[i]
            for j in range(node_matrix.shape[1]):
                if node_matrix[i][j] == 1:
                    if j == 0:  # 第一次开粗
                        node_time_matrix[i][j] = self.calculate_roughing_time(temp_tool_dia, 0)
                    elif (j == 1) | (j == 2):  # 半精侧壁
                        node_time_matrix[i][j] = self.calculate_bound_finishing_time(temp_tool_dia, 0) #孔的另算，因为太小
                    elif (j == 4) | (j == 5):  # 精侧壁
                        node_time_matrix[i][j] = self.calculate_bound_finishing_time(temp_tool_dia, 0)
        return node_time_matrix

    def estimate_slot_machining_time(self, node_matrix):
        node_time_matrix = np.zeros_like(node_matrix)
        for i in range(node_matrix.shape[0]):
            temp_tool_dia = ToolDias[i]
            for j in range(node_matrix.shape[1]):
                if node_matrix[i][j] == 1:
                    if j == 0:  # 第一次开粗
                        node_time_matrix[i][j] = self.calculate_roughing_time(temp_tool_dia, 0)
                    elif (j == 1) | (j == 2):  # 半精侧壁
                        node_time_matrix[i][j] = self.calculate_bound_finishing_time(temp_tool_dia, 1)
                    elif (j == 4) | (j == 5):  # 精侧壁
                        node_time_matrix[i][j] = self.calculate_bound_finishing_time(temp_tool_dia, 0)
                    elif j == 3:  # 半精底面
                        node_time_matrix[i][j] = self.calculate_bottom_finishing_time(temp_tool_dia, 1)
                    elif j == 6:  # 精底面
                        node_time_matrix[i][j] = self.calculate_bottom_finishing_time(temp_tool_dia, 0)
        return node_time_matrix

    def calculate_roughing_time(self, tool_dia, plane): # plane=1
        virtual_length = 2 * (self.Flength + self.Fwidth - 4 * rough_allowance)
        if plane == 1:  # 是平面特征
            machine_depth = blank_allowance
        else:
            machine_depth = abs(self.Fdepth + blank_allowance-rough_allowance)
        a_e = tool_dia * 2 / 3  # 切宽
        a_p = tool_dia * 1 / 2  # 切深
        l_1 = virtual_length/4
        if l_1 <= 0:
            l_1 = 1
        n_1 = math.ceil(l_1 / a_e) + 1
        n_2 = math.ceil(machine_depth / a_p)
        l = (n_1 * l_1 + l_1) * n_2
        time = 60 * l / rough_v_feed
        return time

    '''
    计算侧壁的半/精加时间
    '''
    def calculate_bound_finishing_time(self, env_tool_ac, semi):  # semi=1半精,semi=0精
        if semi == 1:
            allowance = semi_finish_allowance
            v_f = semi_finish_v_feed
        else:
            allowance = finish_allowance
            v_f = finish_v_feed

        virtual_length = 2 * (self.Flength + self.Fwidth - 4 * allowance)
        machine_depth = abs(self.Fdepth - allowance)
        l_1 = virtual_length - env_tool_ac
        if l_1 <= 0:
            l_1 = 1
        a_p = env_tool_ac * 1 / 3  # 切深
        n = math.ceil(machine_depth / a_p)
        l = l_1 * 4 * n
        time = 60 * l / v_f
        return time

    '''
    计算底面的半/精加时间
    '''
    def calculate_bottom_finishing_time(self, tool_dia, semi):  # semi=1半精,semi=0精
        if semi == 1:
            allowance = semi_finish_allowance
            v_f = semi_finish_v_feed
        else:
            allowance = finish_allowance
            v_f = finish_v_feed

        virtual_length = 2 * (self.Flength + self.Fwidth - 4 * allowance)
        a_e = tool_dia * 1 / 3  # 切宽

        l_1 = virtual_length / 4  # 20230525
        if l_1 <= 0:
            l_1 = 1
        n = math.ceil(l_1 / a_e) + 1
        l = n * l_1 + l_1
        time = 60 * l / v_f
        return time

    '''
    计算精铣转角的时间
    '''
    def calculate_corner_finishing_time(self, env_tool_ac):
        num_corners = self.CornerNum
        r = self.CornerR
        l_1 = (2 + math.sqrt(2)) * r

        machine_depth = abs(self.Fdepth - finish_allowance)
        a_p = env_tool_ac * 1 / 3  # 切深
        n = math.ceil(machine_depth / a_p)

        l = l_1 * n * num_corners
        time_1 = 60 * l / finish_v_feed

        virtual_length = 2 * (self.Flength + self.Fwidth - 4 * finish_allowance)

        time_app = 60 * start_end/approach_speed*num_corners
        time_ret = 60 * start_end/retract_speed*num_corners
        time_fate = 60 * (virtual_length)/fate_speed*num_corners

        return time_app+time_ret+time_fate+time_1

    '''
    计算二次开粗时间
    '''
    def calculate_second_roughing_time(self, tool_dia):
        rate = 0.2 # 0.2的区域需要二次开粗
        virtual_length = 2 * (self.Flength + self.Fwidth - 4 * rough_allowance) * rate
        machine_depth = abs(self.Fdepth + blank_allowance-rough_allowance)
        a_e = tool_dia * 2 / 3  # 切宽
        a_p = tool_dia * 1 / 2  # 切深
        l_1 = virtual_length/4
        if l_1 <= 0:
            l_1 = 1
        n_1 = math.ceil(l_1 / a_e) + 1
        n_2 = math.ceil(machine_depth / a_p)
        l = (n_1 * l_1 + l_1) * n_2
        time = 60 * l / rough_v_feed
        return time

    '''
    将每个加工操作的加工时间看作为独立的，并对每个加工操作的加工时间进行归一化。每个加工操作的时间最大为1， 最小为0，换刀时间给1。另外，一次开粗
    和二次开粗的时间放在一块计算，精侧壁和精转角的时间放在一块计算，五轴精侧壁和五轴精转角的时间放在一块计算。另换刀时间可以看作为1。
    x = (x-x_min)/(x_max-x_min)+alpha; alpha = 0.1, 因此归一化后的时间最大为1.1
    '''
    def normalize_time(self, node_time_matrix, padding = 0, alpha=0.1): #padding <= 0
        node_nor_time_matrix = np.ones_like(node_time_matrix)*padding
        s = np.sum(node_time_matrix, axis=0)
        for i in range(s.shape[0]):
            if s[i] != 0:
                operation_i = node_time_matrix[:,i]
                if i == 0: # 第一次开粗和第二次开粗放在一块进行计算
                    operation_9 = node_time_matrix[:,9]
                    if s[9]!= 0: # 有第二次开粗的时间
                        min_i = np.min(np.concatenate((operation_i, operation_9), axis=0))
                        if min_i == 0:
                            min_i = np.partition(np.unique(np.concatenate((operation_i, operation_9), axis=0)), 1)[1] # 最小的值可能为0，此时取第二小的值
                        max_i = np.max(np.concatenate((operation_i, operation_9), axis=0))
                        if max_i == min_i:
                            operation_i[operation_i != 0] = operation_i[operation_i != 0] / max_i + alpha
                            operation_9[operation_9 != 0] = operation_9[operation_9 != 0] / max_i + alpha
                        else:
                            operation_i[operation_i != 0] = (operation_i[operation_i != 0] - min_i) / (
                                        max_i - min_i) + alpha
                            operation_9[operation_9 != 0] = (operation_9[operation_9 != 0] - min_i) / (
                                        max_i - min_i) + alpha
                    else:
                        min_i = np.min(operation_i)
                        if min_i == 0:
                            min_i = np.partition(np.unique(operation_i), 1)[1]  # 最小的值可能为0，此时取第二小的值
                        max_i = np.max(operation_i)
                        if max_i==min_i:
                            operation_i[operation_i != 0] = operation_i[operation_i != 0]/max_i+alpha
                        else:
                            operation_i[operation_i != 0] = (operation_i[operation_i != 0] - min_i) / (
                                        max_i - min_i) + alpha

                    node_nor_time_matrix[:, i] = operation_i
                    node_nor_time_matrix[:, 9] = operation_9
                elif i == 4:  # 精铣侧壁和精铣转角放在一块计算
                    operation_7 = node_time_matrix[:,7] #精转角
                    if s[7] != 0:  # 有精转角的时间
                        min_i = np.min(np.concatenate((operation_i, operation_7), axis=0))
                        if min_i == 0:
                            min_i = np.partition(np.unique(np.concatenate((operation_i, operation_7), axis=0)), 1)[
                                1]  # 最小的值可能为0，此时取第二小的值
                        max_i = np.max(np.concatenate((operation_i, operation_7), axis=0))
                        if max_i == min_i:
                            operation_i[operation_i != 0] = operation_i[operation_i != 0] / max_i + alpha
                            operation_7[operation_7 != 0] = operation_7[operation_7 != 0] / max_i + alpha
                        else:
                            operation_i[operation_i != 0] = (operation_i[operation_i != 0] - min_i) / (
                                    max_i - min_i) + alpha
                            operation_7[operation_7 != 0] = (operation_7[operation_7 != 0] - min_i) / (
                                    max_i - min_i) + alpha
                    else:
                        min_i = np.min(operation_i)
                        if min_i == 0:
                            min_i = np.partition(np.unique(operation_i), 1)[1]  # 最小的值可能为0，此时取第二小的值
                        max_i = np.max(operation_i)
                        if max_i == min_i:
                            operation_i[operation_i != 0] = operation_i[operation_i != 0] / max_i + alpha
                        else:
                            operation_i[operation_i != 0] = (operation_i[operation_i != 0] - min_i) / (
                                    max_i - min_i) + alpha

                    node_nor_time_matrix[:, i] = operation_i
                    node_nor_time_matrix[:, 7] = operation_7
                elif i == 5:  # 五轴精铣侧壁和五轴精铣转角放在一块计算
                    operation_8 = node_time_matrix[:, 8] #五轴精转角
                    if s[8] != 0:  # 有五轴精转角的时间
                        min_i = np.min(np.concatenate((operation_i, operation_8), axis=0))
                        if min_i == 0:
                            min_i = np.partition(np.unique(np.concatenate((operation_i, operation_8), axis=0)), 1)[
                                1]  # 最小的值可能为0，此时取第二小的值
                        max_i = np.max(np.concatenate((operation_i, operation_8), axis=0))
                        if max_i == min_i:
                            operation_i[operation_i != 0] = operation_i[operation_i != 0] / max_i + alpha
                            operation_8[operation_8 != 0] = operation_8[operation_8 != 0] / max_i + alpha
                        else:
                            operation_i[operation_i != 0] = (operation_i[operation_i != 0] - min_i) / (
                                    max_i - min_i) + alpha
                            operation_8[operation_8 != 0] = (operation_8[operation_8 != 0] - min_i) / (
                                    max_i - min_i) + alpha
                    else:
                        min_i = np.min(operation_i)
                        if min_i == 0:
                            min_i = np.partition(np.unique(operation_i), 1)[1]  # 最小的值可能为0，此时取第二小的值
                        max_i = np.max(operation_i)
                        if max_i == min_i:
                            operation_i[operation_i != 0] = operation_i[operation_i != 0] / max_i + alpha
                        else:
                            operation_i[operation_i != 0] = (operation_i[operation_i != 0] - min_i) / (
                                    max_i - min_i) + alpha
                    node_nor_time_matrix[:, i] = operation_i
                    node_nor_time_matrix[:, 8] = operation_8
                elif (i!=7)&(i!=8)&(i!=9):
                    min_i = np.min(operation_i)
                    if min_i == 0:
                        min_i = np.partition(np.unique(operation_i), 1)[1] # 最小的值可能为0，此时取第二小的值
                    max_i = np.max(operation_i)
                    if max_i == min_i:
                        operation_i[operation_i != 0] = operation_i[operation_i != 0] / max_i + alpha
                    else:
                        operation_i[operation_i!=0] = (operation_i[operation_i!=0]-min_i)/(max_i-min_i)+alpha
                    node_nor_time_matrix[:, i] = operation_i
        return node_nor_time_matrix

    def get_distance_matrix(self, all_node_Coor_tensor):
        all_node_Coor_np = all_node_Coor_tensor.numpy()
        dis_matrix = cdist(all_node_Coor_np, all_node_Coor_np, 'euclidean')
        nor_dis_matrix = dis_matrix/(np.max(dis_matrix))
        return nor_dis_matrix, dis_matrix



class GraphConstruction():  # 不包括原始的时间和原始的距离矩阵
    def __init__(self, nor_nodes_feasible_matrix, nor_nodes_time_matrix, edges_attr, dis_matrix_tensor):
        self.nor_nodes_feasible_matrix = nor_nodes_feasible_matrix
        self.nor_nodes_time_matrix = nor_nodes_time_matrix
        self.edges_attr = edges_attr
        self.dis_matrix_tensor = dis_matrix_tensor

        nodes_num = self.nor_nodes_feasible_matrix.shape[0]
        tools_num = self.nor_nodes_feasible_matrix.shape[1]
        operations_num = self.nor_nodes_feasible_matrix.shape[2]
        self.nor_nodes_feasible_vector = nor_nodes_feasible_matrix.reshape((nodes_num, tools_num * operations_num, -1)).squeeze()
        self.nor_nodes_time_vector = nor_nodes_time_matrix.reshape((nodes_num, tools_num * operations_num, -1)).squeeze()

    #father_node为0表示该节点没有父节点, 存在节点的入边中属性为-1，那么认为该节点没有父节点
    #该函数仅判断主从结构的父特征，而不对基准进行判断
    def judge_terminal_father_nodes(self, graph):
        father_node = th.ones(graph.num_nodes())
        father_node_id = th.zeros(graph.num_nodes())
        for node_id in range(graph.num_nodes()):
            in_edges = graph.in_edges(node_id, form='all')
            in_edges_attr = th.index_select(graph.edata["EdgeAttr"], 0, in_edges[2])[:, :3]
            indices = th.argwhere(in_edges_attr == 1)
            if 0 not in indices[:, 1]:
                father_node[node_id] = 0 #没有父节点
                father_node_id[node_id] = -1 #没有父节点的给-1
            else:
                indices_0 = th.argwhere(indices[:, 1] == 0).squeeze(-1)
                father_node_id[node_id] = in_edges[0][indices_0[0]]

        return father_node, father_node_id

    def get_node_grandfather(self, node_id, father_node_ids, all_benchmark_nodes):
        t = 0
        grandfather_id = -2
        while True:
            t += 1
            father_id = int(father_node_ids[node_id].item())
            if father_id not in all_benchmark_nodes:
                node_id = father_id
                if t == 100:
                    break
            else:
                grandfather_id = father_id
                break

        return grandfather_id

    # 找到基准作为父节点并增加一条边，记录某个特征的对应的基准特征的ID（假设每个节点只有一个基准特征），这个id并不用于训练只是用于环境中快速进行确定
    def judge_benchmark_father_nodes(self, graph, father_node_ids):
        # 把所有没有父节点的特征看作为基准
        all_benchmark_nodes = th.argwhere(father_node_ids == -1).squeeze(-1)
        benchmark_node_id = th.zeros(graph.num_nodes())
        for node_id in range(graph.num_nodes()):
            if node_id in all_benchmark_nodes:
                benchmark_node_id[node_id] = -1 # 本身就是基准特征
            else:
                benchmark_node_id[node_id] = self.get_node_grandfather(node_id, father_node_ids, all_benchmark_nodes)
                graph.add_edges(int(benchmark_node_id[node_id].item()), node_id, {"EdgeAttr": th.tensor([[0,0,0,1,0]]).float()})

        return graph, benchmark_node_id

    def construct_graph(self, all_attr):
        source, target = self.edges_attr[:, 0].int(), self.edges_attr[:, 1].int()
        g = dgl.graph((source, target))
        g.ndata["NodeAttr"] = all_attr
        g.edata["EdgeAttr"] = self.edges_attr[:, 2:]

        father_node, father_node_id = self.judge_terminal_father_nodes(g)
        g.ndata["NodeAttr"] = th.concat([g.ndata["NodeAttr"], father_node.unsqueeze(-1)], 1) # (N, 230*2+1)
        g.ndata["Father_id"] = father_node_id
        g, benchmark_id = self.judge_benchmark_father_nodes(g, father_node_id)
        g.ndata["Benchmark_id"] = benchmark_id
        g.ndata["Dis_matrix"] = self.dis_matrix_tensor
        benchmark_id_np = benchmark_id.numpy()
        benchmark_id_np_final = benchmark_id_np[benchmark_id_np!=-1]
        # 使用numpy.unique找到数组中唯一的元素和它们的计数
        unique_elements, counts = np.unique(benchmark_id_np_final, return_counts=True)
        # 找到出现次数最多的元素
        most_common_index = np.argmax(counts)
        most_common_value = unique_elements[most_common_index]
        most_common_value_one_hot =  one_hot(most_common_value, benchmark_id.shape)
        g.ndata["If_entire_benchmark"] = most_common_value_one_hot
        return g

    def construct_attr_graph_with_time(self):
        '''
        构建包括加工时间在内的图
        '''
        all_attr = th.concat((self.nor_nodes_feasible_vector, self.nor_nodes_time_vector), dim=-1) # (N, 230*2)
        return self.construct_graph(all_attr)

    def construct_attr_graph_without_time(self):
        '''
        构建包括加工时间在内的图
        '''
        all_attr =self.nor_nodes_feasible_vector # (N, 230)
        return self.construct_graph(all_attr)

class GraphConstruction_all_time(): # 包括原始的时间和原始的距离矩阵
    def __init__(self, nor_nodes_feasible_matrix, org_nodes_time_matrix, nor_nodes_time_matrix, edges_attr, dis_matrix_tensor_nor, dis_matrix_tensor_org):
        self.nor_nodes_feasible_matrix = nor_nodes_feasible_matrix
        self.org_nodes_time_matrix = org_nodes_time_matrix
        self.nor_nodes_time_matrix = nor_nodes_time_matrix
        self.edges_attr = edges_attr
        self.dis_matrix_tensor_nor = dis_matrix_tensor_nor
        self.dis_matrix_tensor_org = dis_matrix_tensor_org

        nodes_num = self.nor_nodes_feasible_matrix.shape[0]
        tools_num = self.nor_nodes_feasible_matrix.shape[1]
        operations_num = self.nor_nodes_feasible_matrix.shape[2]
        self.nor_nodes_feasible_vector = nor_nodes_feasible_matrix.reshape((nodes_num, tools_num * operations_num, -1)).squeeze()
        self.nor_nodes_time_vector = nor_nodes_time_matrix.reshape((nodes_num, tools_num * operations_num, -1)).squeeze()
        self.org_nodes_time_vector = org_nodes_time_matrix.reshape((nodes_num, tools_num * operations_num, -1)).squeeze()
        # father_node为0表示该节点没有父节点, 存在节点的入边中属性为-1，那么认为该节点没有父节点
        # 该函数仅判断主从结构的父特征，而不对基准进行判断

    def judge_terminal_father_nodes(self, graph):
        father_node = th.ones(graph.num_nodes())
        father_node_id = th.zeros(graph.num_nodes())
        for node_id in range(graph.num_nodes()):
            in_edges = graph.in_edges(node_id, form='all')
            in_edges_attr = th.index_select(graph.edata["EdgeAttr"], 0, in_edges[2])[:, :3]
            indices = th.argwhere(in_edges_attr == 1)
            if 0 not in indices[:, 1]:
                father_node[node_id] = 0  # 没有父节点
                father_node_id[node_id] = -1  # 没有父节点的给-1
            else:
                indices_0 = th.argwhere(indices[:, 1] == 0).squeeze(-1)
                father_node_id[node_id] = in_edges[0][indices_0[0]]

        return father_node, father_node_id

    def get_node_grandfather(self, node_id, father_node_ids, all_benchmark_nodes):
        t = 0
        grandfather_id = -2
        while True:
            t += 1
            father_id = int(father_node_ids[node_id].item())
            if father_id not in all_benchmark_nodes:
                node_id = father_id
                if t == 100:
                    break
            else:
                grandfather_id = father_id
                break

        return grandfather_id

        # 找到基准作为父节点并增加一条边，记录某个特征的对应的基准特征的ID（假设每个节点只有一个基准特征），这个id并不用于训练只是用于环境中快速进行确定

    def judge_benchmark_father_nodes(self, graph, father_node_ids):
        # 把所有没有父节点的特征看作为基准
        all_benchmark_nodes = th.argwhere(father_node_ids == -1).squeeze(-1)
        benchmark_node_id = th.zeros(graph.num_nodes())
        for node_id in range(graph.num_nodes()):
            if node_id in all_benchmark_nodes:
                benchmark_node_id[node_id] = -1  # 本身就是基准特征
            else:
                benchmark_node_id[node_id] = self.get_node_grandfather(node_id, father_node_ids, all_benchmark_nodes)
                graph.add_edges(int(benchmark_node_id[node_id].item()), node_id,
                                {"EdgeAttr": th.tensor([[0, 0, 0, 1, 0]]).float()})

        return graph, benchmark_node_id

    def construct_graph(self, all_attr):
        source, target = self.edges_attr[:, 0].int(), self.edges_attr[:, 1].int()
        g = dgl.graph((source, target))
        g.ndata["NodeAttr"] = all_attr
        g.edata["EdgeAttr"] = self.edges_attr[:, 2:]

        father_node, father_node_id = self.judge_terminal_father_nodes(g)
        g.ndata["NodeAttr"] = th.concat([g.ndata["NodeAttr"], father_node.unsqueeze(-1)], 1)  # (N, 230*2+1)
        g.ndata["Father_id"] = father_node_id
        g, benchmark_id = self.judge_benchmark_father_nodes(g, father_node_id)
        g.ndata["Benchmark_id"] = benchmark_id
        g.ndata["Dis_matrix"] = self.dis_matrix_tensor_nor
        g.ndata["Dis_matrix_org"] = self.dis_matrix_tensor_org
        g.ndata["Time_origin"] = self.org_nodes_time_vector
        benchmark_id_np = benchmark_id.numpy()
        benchmark_id_np_final = benchmark_id_np[benchmark_id_np != -1]
        # 使用numpy.unique找到数组中唯一的元素和它们的计数
        unique_elements, counts = np.unique(benchmark_id_np_final, return_counts=True)
        # 找到出现次数最多的元素
        most_common_index = np.argmax(counts)
        most_common_value = unique_elements[most_common_index]
        most_common_value_one_hot = one_hot(most_common_value, benchmark_id.shape)
        g.ndata["If_entire_benchmark"] = most_common_value_one_hot
        return g

    def construct_attr_graph_with_time(self): #只有归一化后的时间
        '''
        构建包括加工时间在内的图
        '''
        all_attr = th.concat((self.nor_nodes_feasible_vector, self.nor_nodes_time_vector), dim=-1)  # (N, 230*2)
        return self.construct_graph(all_attr)

    def construct_attr_graph_without_time(self):
        '''
        构建不包括加工时间在内的图
        '''
        all_attr = self.nor_nodes_feasible_vector  # (N, 230)
        return self.construct_graph(all_attr)


def draw_graph(graph):

    nx_g = dgl.to_networkx(graph)
    C = nx.spring_layout(nx_g)
    # nx.draw(nx_g, pos, with_labels=True, node_color='skyblue', node_size=1200, font_size=10, font_color='black',
    #         font_weight='bold', edge_color='gray', linewidths=1, alpha=0.7)
    nx.draw(nx_g, with_labels=True, pos=nx.circular_layout(nx_g), node_color='r', edge_color='b')
    plt.show()

def save_graph(graph, dgl_file):
    dgl.save_graphs(dgl_file, graph)

'''
    x = (x-x_min)/(x_max-x_min)+alpha; alpha = 0.1, 因此归一化后的时间最大为1.1
'''
def change_time_matrix_tensor(all_node_nor_time_matrix_tensor):
    new_m = np.sqrt(all_node_nor_time_matrix_tensor)
    max_i = np.max(new_m)
    min_i = np.min(new_m)
    all_node_nor_time_matrix_tensor_flatten = all_node_nor_time_matrix_tensor.flatten()
    if min_i == 0:
        min_i = np.partition(np.unique(all_node_nor_time_matrix_tensor_flatten), 1)[1] # 最小的值可能为0，此时取第二小的值
    zero_mask = (new_m != 0)
    new_m = (np.where(zero_mask, new_m, 0)-min_i)/(max_i-min)
    return new_m

if __name__ == "__main__":
    # only for test Zhang et al. 2024 "F:\ZH\AllDRLPP\Code\Code\Alldata\OriginalData_3\WZPart_ZH_T4.txt"
    # F:\ZH\AllDRLPP_MC\catdata\txt\part_data\PartOriginalGraph\txt_ZH_test4.txt
    txt_name = r"F:\ZH\AllDRLPP_MC\catdata\txt\part_data\PartOriginalGraph\WZPart_ZH_T20.txt"
    dgl_file = r"F:\ZH\AllDRLPP_MC\catdata\txt\part_data\MachiningFeatureProcessGraph\ZH_test20_with_cons.dgl"
    # ZH_test4_with_cons_all_time_no_cons_only_2024
    dgl_file_with_all_time = r"F:\ZH\AllDRLPP_MC\catdata\txt\part_data\MachiningFeatureProcessGraph\ZH_test20_with_cons_all_time.dgl"

    temp_nodes_attr, temp_edges_attr = read_txt(txt_name)
    GAFNE = GetAttrFromNodeEdge(temp_nodes_attr, temp_edges_attr)
    all_node_matrix_tensor, all_node_time_matrix_tensor, all_node_nor_time_matrix_tensor, dis_matrix_nor_tensor, dis_matrix_org_tensor = GAFNE.convert_nodes_attr()
    edges_attr_tensor = GAFNE.normalize_edges_attr()

    ################################################以下为构建归一化时间的代码##############################################
    # GC = GraphConstruction(all_node_matrix_tensor, all_node_nor_time_matrix_tensor, edges_attr_tensor, dis_matrix_nor_tensor)
    # Graph_with_time = GC.construct_attr_graph_with_time()
    # #Graph_without_time = GC.construct_attr_graph_without_time()
    # print(Graph_with_time)
    # #draw_graph(Graph_with_time)
    # save_graph(Graph_with_time, dgl_file)

    ################################################以下为构建全部时间的代码，不光包括归一化时间还包括原始时间##############################################
    GC_alltime = GraphConstruction_all_time(all_node_matrix_tensor, all_node_time_matrix_tensor, all_node_nor_time_matrix_tensor, edges_attr_tensor, dis_matrix_nor_tensor, dis_matrix_org_tensor)
    Graph_with_all_time = GC_alltime.construct_attr_graph_with_time()
    print(Graph_with_all_time)
    save_graph(Graph_with_all_time, dgl_file_with_all_time)
