from argparse import ArgumentParser
from torch import nn
import torch

parser = ArgumentParser()
parser.add_argument('--input_dim', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=10)

args = parser.parse_args()

# equivalent
# one_layer = nn.Linear(32, 10)
one_layer = torch.nn.Linear(args.input_dim, args.hidden_dim)

opt = torch.optim.Adam(one_layer.parameters(), lr=0.002)

for epoch in range(10):
  for batch_i in range(100):
    
    # fake optimization
    x = torch.rand(3, 32)
    
    loss = one_layer(x).sum()
    loss.backward()
    
    opt.step()
    opt.zero_grad()
    print('epoch ', epoch, 'batch_i', batch_i)
