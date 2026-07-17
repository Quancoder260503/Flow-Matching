import torch.nn as nn 

class EMAHelper(object):
  def __init__(self, mu = 0.999): 
   self.mu = mu 
   self.shadow = {}
   self.num_step = 0 

  def register(self, module): 
    if isinstance(module, nn.DataParallel):
      module = module.module 
    for name, param in module.named_parameters(): 
      if param.requires_grad : 
        self.shadow[name] = param.data.clone() 

  def update(self, module, start_ema_step = 2000):  
    if isinstance(module, nn.DataParallel):
      module = module.module 
    self.num_step += 1 
    for name, param in module.named_parameters(): 
      if param.requires_grad : 
        if self.num_step < start_ema_step : 
          self.shadow[name] = param.data 
        else : 
          self.shadow[name] = self.mu * self.shadow[name] + (1. - self.mu) * param.data 
    

  def ema(self, module):
   if isinstance(module, nn.DataParallel):
    module = module.module 
   for name, param in module.named_parameters(): 
    if param.requires_grad : 
      param.data.copy_(self.shadow[name].data)

  def ema_copy(self, module):
    if isinstance(module, nn.DataParallel):
      inner_module = module.module 
      module_copy  = type(inner_module)(inner_module.config).to(inner_module.config.device)
      module_copy.load_state_dict(inner_module.state_dict())
      module_copy = nn.DataParallel(module_copy)
    else :
      module_copy  = type(module)(module.config).to(module.config.device)
      module_copy.load_state_dict(module.state_dict())

    self.ema(module_copy)
    return module_copy 
    
  def state_dict(self):
    return self.shadow 
    
  def load_state_dict(self, state_dict): 
    self.shadow = state_dict 