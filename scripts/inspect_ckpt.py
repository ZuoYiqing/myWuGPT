import torch
import os, time, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import src.model as m
from torch.serialization import safe_globals
p='weights/pretrain.pt'
if not os.path.exists(p):
    print('checkpoint not found')
else:
    with safe_globals([m.GPTConfig]):
        ck = torch.load(p, map_location='cpu', weights_only=False)
    print('checkpoint step:', ck.get('step'))
    print('mtime:', time.ctime(os.path.getmtime(p)))
    print('train_args:', ck.get('train_args'))
