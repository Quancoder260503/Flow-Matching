from abc import ABC, abstractmethod 
from dataclasses import dataclass, field 
from typing import Union 
import torch 
from torch import Tensor

"""
Scheduler for the mean and variance for the distribution \phi_{t}(x | x_1) = \sigma(x_1) x + \mean(x_1)
"""

@dataclass
class SchedulerOutput:
  alpha_t: Tensor = field(metadata = {"help": "alpha_t"})
  sigma_t: Tensor = field(metadata = {"help": "sigma_t"})
  d_alpha_t: Tensor = field(metadata = {"help": "Derivative of alpha_t."})
  d_sigma_t: Tensor = field(metadata = {"help": "Derivative of sigma_t."})


class Scheduler(ABC): 
  @abstractmethod
  def __call__(self, t : Tensor) -> SchedulerOutput:
    pass 

  @abstractmethod
  def snr_inverse(self, snr : Tensor) -> Tensor: 
    pass 


class ConvexScheduler(Scheduler):
  @abstractmethod 
  def __call__(self, t : Tensor) -> SchedulerOutput :
    pass 
    
  @abstractmethod
  def reparameterize_inverse(self, r_x_t : Tensor) -> Tensor: 
    pass 
    
  def snr_inverse(self, snr : Tensor) -> Tensor:
    r_x_t = snr / (1.0 + snr) 
    return self.reparameterize_inverse(r_x_t = r_x_t) 



class ConditionalOptimalTransportScheduler(ConvexScheduler):
  def __call__(self, t : Tensor) -> Scheduler:
    return SchedulerOutput(
      alpha_t = t, 
      sigma_t = 1 - t,
      d_alpha_t = torch.ones_like(t), 
      d_sigma_t = -torch.ones_like(t), 
    )
    
  def reparameterize_inverse(self, r_x_t : Tensor) -> Tensor:
    return r_x_t

class PolynomialConvexScheduler(ConvexScheduler):
  """Polynomial Scheduler."""

  def __init__(self, n: Union[float, int]) -> None:
    assert isinstance(n, (float, int)), f"`n` must be a float or int. Got {type(n)=}."
    assert n > 0, f"`n` must be positive. Got {n=}."
    self.n = n

  def __call__(self, t: Tensor) -> SchedulerOutput:
    return SchedulerOutput(
      alpha_t = t ** self.n,
      sigma_t = 1 - t ** self.n,
      d_alpha_t = self.n * (t ** (self.n - 1)),
      d_sigma_t = -self.n * (t ** (self.n - 1)),
    )

  def reparameterize_inverse(self, r_x_t : Tensor) -> Tensor:
    return torch.pow(r_x_t, 1.0 / self.n)


class VPScheduler(Scheduler):
  """Variance Preserving Scheduler."""
  def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
    self.beta_min = beta_min
    self.beta_max = beta_max
    super().__init__()

  def __call__(self, t: Tensor) -> SchedulerOutput:
    b = self.beta_min
    B = self.beta_max
    T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
    dT = -(1 - t) * (B - b) - b

    return SchedulerOutput(
      alpha_t = torch.exp(-0.5 * T),
      sigma_t = torch.sqrt(1 - torch.exp(-T)),
      d_alpha_t = -0.5 * dT * torch.exp(-0.5 * T),
      d_sigma_t = 0.5 * dT * torch.exp(-T) / torch.sqrt(1 - torch.exp(-T)),
    )

  def snr_inverse(self, snr: Tensor) -> Tensor:
    T = -torch.log(snr ** 2 / (snr ** 2 + 1))
    b = self.beta_min
    B = self.beta_max
    t = 1 - ((-b + torch.sqrt(b**2 + 2 * (B - b) * T)) / (B - b))
    return t


class LinearVPScheduler(Scheduler):
  """Linear Variance Preserving Scheduler."""
  def __call__(self, t: Tensor) -> SchedulerOutput:
    return SchedulerOutput(
      alpha_t = t,
      sigma_t = (1 - t**2) ** 0.5,
      d_alpha_t = torch.ones_like(t),
      d_sigma_t = -t / (1 - t**2) ** 0.5,
    )

  def snr_inverse(self, snr: Tensor) -> Tensor:
    return torch.sqrt(snr ** 2 / (1 + snr ** 2))


class CosineScheduler(Scheduler):
  """Cosine Scheduler."""
  def __call__(self, t: Tensor) -> SchedulerOutput:
    pi = torch.pi
    return SchedulerOutput(
      alpha_t = torch.sin(pi / 2 * t),
      sigma_t = torch.cos(pi / 2 * t),
      d_alpha_t = pi / 2 * torch.cos(pi / 2 * t),
      d_sigma_t = -pi / 2 * torch.sin(pi / 2 * t),
    )

  def snr_inverse(self, snr: Tensor) -> Tensor:
    return 2.0 * torch.atan(snr) / torch.pi