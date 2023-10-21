from dgl.nn import GraphConv
import torch.nn.functional as F
import torch.nn as nn
from dataset.RasterDataset import RasterDataset
import torch
import dgl
from models.GAT import GAT
import os
os.environ["DGLBACKEND"] = "pytorch"


def train_single_graph(g, model, optimizer):

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    # Forward
    logits = model(g, features)
    # Compute prediction
    pred = logits

    # Compute loss
    # Note that you should only compute the losses of the nodes in the training set.
    train_loss = F.mse_loss(logits[train_mask], labels[train_mask])

    # Compute accuracy on training/validation/test
    test_loss = F.mse_loss(logits[test_mask], labels[test_mask])

    # Backward
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return train_loss, test_loss


def train_all_graphs(graphs, model, num_epochs):
    best_test_loss = float('inf')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for graph in graphs:
            train_loss, test_loss = train_single_graph(
                graph, model, optimizer)
            # Save the best validation accuracy and the corresponding test accuracy.
            if best_test_loss > test_loss:
                best_test_loss = test_loss

        if epoch % 5 == 0:
            print(
                f"[epoch {epoch:3}],train_loss: {train_loss:6},test_loss: {test_loss:6} (best test_loss: {best_test_loss:6})")


dataset = RasterDataset(path="./data/")
g = dataset[0]

# Create the model with given dimensions
model = GAT(g.ndata["feat"].shape[1], 16, g.ndata["label"].shape[1], heads=4)
train_all_graphs(dataset, model, 100)

checkpoint_path = "./checkpoints/GAT.pt"
torch.save(model.state_dict(), checkpoint_path)
