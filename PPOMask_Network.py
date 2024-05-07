import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
from dgl.dataloading import GraphDataLoader
from DataLoader import GraphDataset

# PNAConv
class PNAConv_Module(nn.Module):
    def __init__(self, in_size, out_size, edge_feat_size, aggregators, scalers, last = 0, delta = 1.0,
                 dropout = 0.0, num_towers = 1):
       super(PNAConv_Module, self).__init__()
       self.last = last
       self.PNA = dglnn.PNAConv(in_size, out_size, aggregators, scalers, delta, dropout, num_towers, edge_feat_size)

    def forward(self, g, n_feat, e_feat):
        h = self.PNA(g, n_feat, e_feat)
        if self.last == 0:
            h = F.relu(h)
        return h

class Masked_Actor_Net_PNAConv(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, num_nodes, out_tool_size, out_process_size):
        super(Masked_Actor_Net_PNAConv, self).__init__()
        aggregators = ['mean', 'max', 'sum', 'std']
        scalars = ['identity']
        self.in_node_feats = in_node_feats
        self.final_out_size = max(num_nodes, out_tool_size, out_process_size)
        self.num_nodes = num_nodes
        self.Linear_1 = nn.Linear(num_nodes, 64)
        self.Linear_2 = nn.Linear(64, 32)
        dropout = 0.0
        self.PNAConvModule_1 = PNAConv_Module(in_node_feats, 384, in_edge_feats, aggregators, scalars, dropout = dropout)
        self.PNAConvModule_2 = PNAConv_Module(384+32, 384, in_edge_feats, aggregators, scalars, dropout = dropout)
        self.PNAConvModule_3 = PNAConv_Module(384, out_tool_size*out_process_size, in_edge_feats, aggregators, scalars, dropout = dropout, last = 1)

        self.Max_pool = dglnn.MaxPooling()

        self.Linear_3 = nn.Linear(self.num_nodes, 64)
        self.Linear_4 = nn.Linear(64, self.num_nodes)

    def forward(self, obs, state = None, info = {}):
        GraphDatas = GraphDataset(obs)
        GraphLoader = GraphDataLoader(GraphDatas, batch_size=obs.shape[0])
        GraphLoader_iter = iter(GraphLoader)
        gs = next(GraphLoader_iter)[0]

        ns = th.tensor(gs.dstdata["NodeAttr"], dtype=th.float32)
        es = th.tensor(gs.edata["EdgeAttr"], dtype=th.float32)

        dm = th.tensor(gs.dstdata["Dis_matrix"], dtype=th.float32)
        mask_fv = th.tensor(gs.dstdata["Mask_vector"], dtype=th.float32)
        B = obs.shape[0]
        mask_fv_r = mask_fv.reshape((B, -1))

        d_feat_1 = self.Linear_1(dm) # B 64
        d_feat_2 = self.Linear_2(d_feat_1)# B 32

        n_feat_1 = self.PNAConvModule_1(gs, ns, es)
        n_feat_1_c = self.concatenate_attr(n_feat_1, d_feat_2.squeeze())
        n_feat_2 = self.PNAConvModule_2(gs, n_feat_1_c, es)
        n_feat_3 = self.PNAConvModule_3(gs, n_feat_2, es) # B*N, C

        # SENet
        n_feat_m,_ = th.max(n_feat_3, -1)
        n_feat_m = n_feat_m.reshape((B, -1))
        n_feat_m_1 = F.relu(self.Linear_3(n_feat_m))
        n_feat_m_2 = F.sigmoid(self.Linear_4(n_feat_m_1)) # B, N
        n_feat_m_2_r1 = n_feat_m_2.reshape((B*self.num_nodes, -1)) # B*N, 1

        n_feat_2_3 = (n_feat_m_2_r1*n_feat_3).reshape((B, -1)) # B, N*C
        g_feature = n_feat_2_3.masked_fill(mask_fv_r==0, value=-1e5)
        soft_feat_1 = F.softmax(g_feature, -1)

        return soft_feat_1, state

    def concatenate_attr(self, node_attr, info_matrix):
        return th.concat([node_attr, info_matrix], dim=-1)

class Critic_Net_PNAConv(Masked_Actor_Net_PNAConv):
    def __init__(self,in_node_feats, in_edge_feats, num_nodes):
        super(Critic_Net_PNAConv, self).__init__(in_node_feats, in_edge_feats, num_nodes, 0, 0)
        self.Linear_a_1 = nn.Linear(384, 64)
        self.Linear_a_2 = nn.Linear(64, 1)


    def forward(self, obs, state = None, info = {}):
        GraphDatas = GraphDataset(obs)
        GraphLoader = GraphDataLoader(GraphDatas, batch_size=obs.shape[0])
        GraphLoader_iter = iter(GraphLoader)
        gs = next(GraphLoader_iter)[0]
        ns = th.tensor(gs.dstdata["NodeAttr"], dtype=th.float32)
        es = th.tensor(gs.edata["EdgeAttr"], dtype=th.float32)
        dm = th.tensor(gs.dstdata["Dis_matrix"], dtype=th.float32)

        d_feat_1 = self.Linear_1(dm) # B 64
        d_feat_2 = self.Linear_2(d_feat_1)# B 32

        n_feat_1 = self.PNAConvModule_1(gs, ns, es)
        n_feat_1_c = self.concatenate_attr(n_feat_1, d_feat_2.squeeze())
        n_feat_2 = self.PNAConvModule_2(gs, n_feat_1_c, es)
        global_feat = self.Max_pool(gs, n_feat_2)

        v_a_feat = self.Linear_a_1(global_feat)
        v_a = self.Linear_a_2(v_a_feat)

        return v_a

class Actor_Net_PNAConv_without_SENet(Masked_Actor_Net_PNAConv):
    def __init__(self,in_node_feats, in_edge_feats, num_nodes, out_tool_size, out_process_size):
        super(Actor_Net_PNAConv_without_SENet, self).__init__(in_node_feats, in_edge_feats, num_nodes, out_tool_size, out_process_size)

    def forward(self, obs, state = None, info = {}):
        GraphDatas = GraphDataset(obs)
        GraphLoader = GraphDataLoader(GraphDatas, batch_size=obs.shape[0])
        GraphLoader_iter = iter(GraphLoader)
        gs = next(GraphLoader_iter)[0]

        ns = th.tensor(gs.dstdata["NodeAttr"], dtype=th.float32)
        es = th.tensor(gs.edata["EdgeAttr"], dtype=th.float32)

        dm = th.tensor(gs.dstdata["Dis_matrix"], dtype=th.float32)
        mask_fv = th.tensor(gs.dstdata["Mask_vector"], dtype=th.float32)
        B = obs.shape[0]
        mask_fv_r = mask_fv.reshape((B, -1))

        d_feat_1 = self.Linear_1(dm)  # B 64
        d_feat_2 = self.Linear_2(d_feat_1)  # B 32

        n_feat_1 = self.PNAConvModule_1(gs, ns, es)
        n_feat_1_c = self.concatenate_attr(n_feat_1, d_feat_2.squeeze())
        n_feat_2 = self.PNAConvModule_2(gs, n_feat_1_c, es)
        n_feat_3 = self.PNAConvModule_3(gs, n_feat_2, es)  # B*N, C

        n_feat_2_3 = n_feat_3.reshape((B, -1))  # B, N*C
        g_feature = n_feat_2_3.masked_fill(mask_fv_r == 0, value=-1e5)
        soft_feat_1 = F.softmax(g_feature, -1)

        return soft_feat_1, state

