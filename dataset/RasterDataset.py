# # 读取csv创建DGL数据集

# #### 1 引入DGL库

import pandas as pd
from dgl.data import DGLDataset
import torch
import dgl
import os
os.environ["DGLBACKEND"] = "pytorch"


def save_data(data, save_file):
    """保存数据"""
    import pickle
    # 保存到文件
    with open(save_file, 'wb') as file:
        pickle.dump(data, file)


def load_data(load_file):
    """读取数据"""
    import pickle
    # 打开文件
    with open(load_file, 'rb') as file:
        data = pickle.load(file)
    return data

# ### 2 创建数据集


class RasterDataset(DGLDataset):
    def __init__(self, path: str, train_rate: float = 0.8):
        self.path = path
        self.node_id_maps = load_data(self.path+"node_id_maps.pkl")
        self.graph_id_map = load_data(self.path+"graph_id_map.pkl")
        self.len = len(self.graph_id_map)
        self.train_rate = train_rate
        super(RasterDataset, self).__init__(name="Raster Timing Dataset")

    def process(self):
        self.graphs = [None] * self.len

        # 读取edges，nodes数据
        edges_df = pd.read_csv(self.path+"edges.csv")
        graph_properties = pd.read_csv(self.path + "graph_propertity.csv")
        nodes_df = pd.read_csv(self.path + "nodes.csv")

        # 先依据graph_id排序
        nodes_df = nodes_df.sort_values(by=['graph_id'], ascending=[True])
        edges_df = edges_df.sort_values(by=['graph_id'], ascending=[True])
        # 将edges，nodes依据graph_id分组
        graph_nodes_dfs = nodes_df.groupby(by="graph_id")
        graph_edges_dfs = edges_df.groupby(by="graph_id")

        # 先读取每个图的结点数量
        num_nodes_dict = {}
        for _, row in graph_properties.iterrows():
            num_nodes_dict[row["graph_id"]] = row["num_nodes"]

        # 处理每个图数据
        for (graph_id, nodes_df), (e_graph_id, edges_df) in zip(graph_nodes_dfs, graph_edges_dfs):
            # 找对应的图
            assert (graph_id == e_graph_id)
            num_nodes = num_nodes_dict[graph_id]
            # 获取图结构
            src = edges_df["src"].to_numpy()
            dst = edges_df["dst"].to_numpy()

            # edges的feats
            edges_feat = torch.from_numpy(edges_df[["F_1", "F_2"]].to_numpy())
            # nodes的feats
            nodes_feat = []
            nodes_label = []
            for i in range(num_nodes):
                t_row = nodes_df[nodes_df["node_id"] == i]
                if not t_row.empty:
                    feat = torch.from_numpy(
                        t_row.iloc[:, 2:-4].values).view(35)
                    label = torch.from_numpy(
                        t_row.iloc[:, -4:-2].values).view(2)
                else:
                    # 补全数据
                    feat = torch.zeros(35)
                    label = torch.zeros(2)
                nodes_feat.append(feat)
                nodes_label.append(label)
            nodes_feat = torch.stack(nodes_feat, dim=0)
            nodes_label = torch.stack(nodes_label, dim=0)
            # 创建图
            g = dgl.graph((src, dst), num_nodes=num_nodes)

            # 添加图的node和edge的feats
            g.edata["feat"] = edges_feat.float()
            g.ndata["feat"] = nodes_feat.float()
            g.ndata["label"] = nodes_label.float()
            # 避免一些结点入度为0
            g = dgl.add_self_loop(g)

            # 分割训练集,测试集
            n_train = int(num_nodes * self.train_rate)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[:n_train] = True
            test_mask[n_train:] = True
            g.ndata["train_mask"] = train_mask
            g.ndata["test_mask"] = test_mask

            self.graphs[graph_id] = g

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


if __name__ == "__main__":
    dataset = RasterDataset("./data/")
    g = dataset[0]
    print(g)
