import torch
from grinvi.config import GrinViConfig
from grinvi.model import GrinViModel
config = GrinViConfig.small()
model = GrinViModel(config)
input_ids = torch.tensor([[1, 2, 3]])
logits, kv = model(input_ids)
print("Initial kv cache shape:", kv[0][0].shape)
input_ids_next = torch.tensor([[4]])
logits, kv2 = model(input_ids_next, kv_caches=kv)
print("Next kv cache shape:", kv2[0][0].shape)
