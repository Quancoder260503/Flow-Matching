import torch 
import torch.nn as nn 
from FlowMatchingRunner import *
from ConfigParser       import *


if __name__  == '__main__' :
    args = parse_args()
    runner = Runner(args = args)
    runner.sample()