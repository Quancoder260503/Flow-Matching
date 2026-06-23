"""
Code re-implemented from 
https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/path_sample.py
"""

from abc import ABC, abstractmethod 
from dataclasses import field, dataclass 
from torch import Tensor 
from Scheduler import ConditionalOptimalTransportScheduler, Scheduler 

def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
  """
  `input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
   expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.
   Args:
    input_tensor (Tensor): (batch_size,).
    expand_to (Tensor): (batch_size, ...).

   Returns:
    Tensor: (batch_size, ...).
  """
  assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
  assert (
    input_tensor.shape[0] == expand_to.shape[0]
  ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

  dim_diff = expand_to.ndim - input_tensor.ndim

  t_expanded = input_tensor.clone()
  t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))
  return t_expanded.expand_as(expand_to)



@dataclass
class PathSample:
  r"""Represents a sample of a conditional-flow generated probability path.
  Attributes:
    x_1 (Tensor): the target sample :math:`X_1`.
    x_0 (Tensor): the source sample :math:`X_0`.
    t (Tensor): the time sample :math:`t`.
    x_t (Tensor): samples :math:`X_t \sim p_t(X_t)`, shape (batch_size, ...).
    dx_t (Tensor): conditional target :math:`\frac{\partial X}{\partial t}`, shape: (batch_size, ...).
  """

  x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
  x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
  t:   Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
  x_t: Tensor = field(metadata = {"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."})
  dx_t: Tensor = field(metadata = {"help": "conditional target dX_t, shape: (batch_size, ...)."})


@dataclass
class DiscretePathSample:
  r"""
  Represents a sample of a conditional-flow generated discrete probability path.

  Attributes:
   x_1 (Tensor): the target sample :math:`X_1`.
   x_0 (Tensor): the source sample :math:`X_0`.
   t (Tensor): the time sample  :math:`t`.
   x_t (Tensor): the sample along the path  :math:`X_t \sim p_t`.
  """

  x_1: Tensor = field(metadata = {"help": "target samples X_1 (batch_size, ...)."})
  x_0: Tensor = field(metadata = {"help": "source samples X_0 (batch_size, ...)."})
  t: Tensor = field(metadata = {"help": "time samples t (batch_size, ...)."})
  x_t: Tensor = field(metadata = {"help": "samples X_t ~ p_t(X_t), shape (batch_size, ...)."})

class ProbabilityPath(ABC):
  @abstractmethod 
  def sample(self, x_0 : Tensor, x_1 : Tensor, t : Tensor) -> PathSample : 
    pass 

  def assert_sample_shape(self, x_0 : Tensor, x_1 : Tensor, t : Tensor): 
    assert (t.ndim == 1), f" The time vector t must have the shape [batch_size]. Got {t.shape}"
    assert (t.shape[0] == x_0.shape[0] == x_1.shape[0]), \
        f"Time t dimension must match the batch size {x_1.shape[0]}. Got {t.shape}"


class AffineProbabilityPath(ProbabilityPath):
  def __init__(self, scheduler : Scheduler): 
    self.scheduler = scheduler 
    
  def sample(self, x_0 : Tensor, x_1 : Tensor, t : Tensor) -> PathSample:
    self.assert_sample_shape(x_0 = x_0, x_1 = x_1, t = t)

    scheduler_output = self.scheduler(t)
    alpha_t = expand_tensor_like(input_tensor = scheduler_output.alpha_t, expand_to = x_1)
    sigma_t = expand_tensor_like(input_tensor = scheduler_output.sigma_t, expand_to = x_1)
    d_alpha_t = expand_tensor_like(input_tensor = scheduler_output.d_alpha_t, expand_to = x_1)
    d_sigma_t = expand_tensor_like(input_tensor = scheduler_output.d_sigma_t, expand_to = x_1)

    # x_t \sim p_t(x | x_1)
    x_t = sigma_t * x_0 + alpha_t * x_1    
    dx_t = d_sigma_t * x_0 + d_alpha_t * x_1

    return PathSample(x_t = x_t, dx_t = dx_t, x_0 = x_0, x_1 = x_1, t = t)
    
  def target_to_velocity(self, x_1 : Tensor, x_t : Tensor, t : Tensor) -> Tensor:
    scheduler_ouput = self.scheduler(t)
    alpha_t   = scheduler_ouput.alpha_t 
    sigma_t   = scheduler_ouput.sigma_t
    d_alpha_t = scheduler_ouput.d_alpha_t
    d_sigma_t = scheduler_ouput.d_sigma_t

    a_t = d_sigma_t / sigma_t 
    b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t

    return a_t * x_t + b_t * x_1
    
  def epsilon_to_velocity(self, epsilon : Tensor, x_t : Tensor, t : Tensor) -> Tensor: 
    scheduler_ouput = self.scheduler(t)
    alpha_t   = scheduler_ouput.alpha_t 
    sigma_t   = scheduler_ouput.sigma_t
    d_alpha_t = scheduler_ouput.d_alpha_t
    d_sigma_t = scheduler_ouput.d_sigma_t

    a_t = d_alpha_t / alpha_t 
    b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t 
    
    return a_t * x_t + b_t * epsilon
    
  def velocity_to_target(self, velocity : Tensor, x_t : Tensor, t : Tensor) -> Tensor : 
    scheduler_ouput = self.scheduler(t)
    alpha_t   = scheduler_ouput.alpha_t 
    sigma_t   = scheduler_ouput.sigma_t
    d_alpha_t = scheduler_ouput.d_alpha_t
    d_sigma_t = scheduler_ouput.d_sigma_t

    a_t = sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)
    b_t = -sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)

    return velocity * a_t + b_t * x_t
    
  def epsilon_to_target(self, epsilon : Tensor, x_t : Tensor, t : Tensor) -> Tensor : 
    scheduler_ouput = self.scheduler(t)
    alpha_t   = scheduler_ouput.alpha_t 
    sigma_t   = scheduler_ouput.sigma_t

    a_t = -sigma_t / alpha_t 
    b_t = 1.0 / alpha_t

    return a_t * epsilon + b_t * x_t


  def velocity_to_epsilon(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
    scheduler_ouput = self.scheduler(t)
    alpha_t   = scheduler_ouput.alpha_t 
    sigma_t   = scheduler_ouput.sigma_t
    d_alpha_t = scheduler_ouput.d_alpha_t
    d_sigma_t = scheduler_ouput.d_sigma_t

    a_t = alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)
    b_t = -d_alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)

    return a_t * velocity + b_t * x_t


  def target_to_epsilon(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
    scheduler_ouput = self.scheduler(t)
    alpha_t   = scheduler_ouput.alpha_t 
    sigma_t   = scheduler_ouput.sigma_t

    a_t = -alpha_t / sigma_t
    b_t = 1.0 / sigma_t

    return x_1 * a_t + b_t * x_t

