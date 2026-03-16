import torch
from hqa_gae.utils.datautil import DataUtil

# Simulate a small shapegraph edge index
edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
try:
    split = DataUtil.train_test_split_edges(edge_index, num_node=3, val_ratio=0.0, test_ratio=0.5)
    print("Success:", split)
except Exception as e:
    print("Exception:", str(e))
