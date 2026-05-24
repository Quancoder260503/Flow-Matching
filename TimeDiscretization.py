import torch 

'''
Read 
https://arxiv.org/abs/2206.00364
'''

def get_time_discretization(num_steps: int, rho = 7):
    step_indices = torch.arange(num_steps, dtype = torch.float32)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vector = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vector = torch.cat([sigma_vector, torch.zeros_like(sigma_vector[:1])])
    time_vector = (sigma_vector / (1 + sigma_vector)).squeeze()
    t_samples = 1.0 - torch.clip(time_vector, min = 0.0, max = 1.0)
    return t_samples
    
print(get_time_discretization(8))