import abc
import collections 
import torch 
from dataclasses import field, dataclass 
import bisect
import math 
'''
====================================================================
Fixed Step Size ODE Solvers with interpolation support.
====================================================================
'''

class FixedStepSizeODESolver(object):
  def __init__(
    self, 
    velo_func, 
    y_0, 
    step_size = None, 
    grid_constructor = None, 
    interpolation = 'linear', 
    perturb = False
  ):
    self.velo_func = velo_func
    self.y_0 = y_0 
    self.dtype = y_0.dtype 
    self.device = y_0.device 
    self.step_size = step_size
    self.interpolation = interpolation
    self.perturb = perturb 

    if step_size is None : 
      if grid_constructor is None : 
        self.grid_constructor = lambda f, y_0, t : t 
      else : 
        self.grid_constructor = grid_constructor 
    else : 
      if grid_constructor is None : 
        self.grid_constructor = self.grid_constructor_from_step_size(step_size)
      else : 
        raise ValueError("Cannot specify both `step_size` and `grid_constructor`")  
    

  def grid_constructor_from_step_size(self, step_size):
    def grid_constructor(t):
      start_time = t[0]
      end_time   = t[-1]
      num_iters = torch.ceil((end_time - start_time) / step_size + 1).item()
      t_infer = torch.arange(0, num_iters, dtype = t.dtype, device = t.device) * step_size + start_time
      t_infer[-1] = t[-1]

      return t_infer
    return grid_constructor
  
  @abc.abstractmethod
  def step_func(self, func, t0, dt, t1, y0):
    pass
  
  def integrate(self, t):
    time_grid = self.grid_constructor(self.velo_func, self.y_0, t)

    results = torch.empty(len(t), *self.y_0.shape, dtype = self.dtype, device = self.device)
    results[0] = self.y_0 

    j = 1 
    y_0 = self.y_0

    for t_0, t_1 in zip(time_grid[:-1], time_grid[1:]):
      dt = (t_1 - t_0)
      ones = torch.ones((self.y_0.shape[0], self.y_0.shape[1] - 1)) 


      dy, f_0 = self.step_func(
        func = self.velo_func, 
        t_0 = ones * t_0, 
        dt  = ones * dt, 
        t_1 = ones * t_1, 
        y_0 = y_0, 
      )
      y_1 = y_0 + dy

      while j < len(t) and t[j] <= t_1 : 
        if self.interpolation == 'linear':
          results[j] = self.linear_interpolation(t_0, t_1, y_0, y_1, t[j])
        elif self.interpolation == 'cubic': 
          f_1 = self.velo_func(t_1, y_1)
          results[j] = self.cubic_interpolation(t_0, y_0, f_0, t_1, y_1, f_1, t[j])
        else:
            raise ValueError(f"Unknown interpolation method {self.interp}")
        j += 1

      y_0 = y_1
    return results

  def linear_interpolation(self, t_0, t_1, y_0, y_1, t):
    if t == t_0 : 
      return y_0 
    if t == t_1 : 
      return y_1 

    slope = (t - t_0) / (t_1 - t_0) 
    return y_0 + slope * (y_1 - y_0)

  def cubic_interpolation(self, t_0, y_0, f_0, t_1, y_1, f_1, t):
    h = (t - t_0) / (t_1 - t_0)
    h00 = (1 + 2 * h) * (1 - h) * (1 - h)
    h10 = h * (1 - h) * (1 - h)
    h01 = h * h * (3 - 2 * h)
    h11 = h * h * (h - 1)
    dt = (t_1 - t_0)
    return h00 * y_0 + h10 * dt * f_0 + h01 * y_1 + h11 * dt * f_1


class EulerSolver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    v_0 = func(t_0, y_0)
    return dt * v_0, v_0
  
class MidPointSolver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    half_dt = dt / 2
    v_0 = func(t_0, y_0) 
    mid_point = y_0 + half_dt * v_0 
    return dt * func(t_0 + half_dt, mid_point), v_0
  
class Heun2Solver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    v_0 = func(t_0, y_0)
    # from butcher books
    butcher_tableu = [
      [0.0, 0.0, 0.0],
      [1.0, 1.0, 0.0],
      [0.0, 1/2, 1/2],
    ]

    k_1 = v_0
    k_2 = func(t_0 + dt * butcher_tableu[1][0], y_0 + dt * butcher_tableu[1][1] * k_1)
    return dt * (butcher_tableu[2][1] * k_1 + butcher_tableu[2][2] * k_2), v_0

class Heun3Solver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    v_0 = func(t_0, y_0)
    # from butcher books
    butcher_tableu = [
      [0.0, 0.0, 0.0, 0.0],
      [1/3, 1/3, 0.0, 0.0],
      [2/3, 0.0, 2/3, 0.0],
      [0.0, 1/4, 0.0, 3/4],
    ]


    k_1 = v_0
    k_2 = func(t_0 + dt * butcher_tableu[1][0], y_0 + dt * butcher_tableu[1][1] * k_1)
    k_3 = func(t_0 + dt * butcher_tableu[2][0], y_0 + dt * (butcher_tableu[2][1] * k_1 + butcher_tableu[2][2] * k_2))
    return dt * (butcher_tableu[3][1] * k_1 + butcher_tableu[3][2] * k_2 + butcher_tableu[3][3] * k_3), v_0


class RungeKutta4Solver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    v_0 = func(t_0, y_0)
    # sampling the trajectory field
    k_1 = v_0 
    k_2 = func(t_0 + dt / 3, y_0 + dt / 3 * k_1)
    k_3 = func(t_0 + dt * 2 / 3, y_0 + dt * (k_2 - k_1 / 3))
    k_4 = func(t_1, y_0 + dt * (k_1 - k_2 + k_3))
    return (k_1 + 3 * (k_2 + k_3) + k_4) * dt * 0.125, v_0


'''
====================================================================
Adaptive Step Size ODE Solvers with interpolation support.
====================================================================
'''
class AdaptiveStepsizeODESolver(abc.ABCMeta):
  def __init__(self, dtype, y_0, norm):
    self.y_0 = y_0
    self.dtype = dtype
    self.norm = norm

  def prepare_integrate(self, t):
      pass

  @abc.abstractmethod
  def advance(self, next_t):
    raise NotImplementedError

  @classmethod
  def valid_callbacks(cls):
    return set()

  def integrate(self, t):
    solution = torch.empty(len(t), *self.y0.shape, dtype = self.y0.dtype, device = self.y0.device)
    solution[0] = self.y_0
    t = t.to(self.dtype)
    self.prepare_integrate(t)
    for i in range(1, len(t)):
      solution[i] = self.advance(t[i])
    return solution


class AdaptiveStepsizeEventODESolver(AdaptiveStepsizeODESolver, abc.ABCMeta):
  @abc.abstractmethod
  def advance_until_event(self, event_fn):
    raise NotImplementedError

  def integrate_until_event(self, t_0, event_fn):
    t0 = t0.to(self.y0.device, self.dtype)
    self.prepare_integrate(t0.reshape(-1))
    event_time, y_1 = self.advance_until_event(event_fn)
    solution = torch.stack([self.y_0, y_1], dim=0)
    return event_time, solution

import abc 
import torch 

'''
====================================================================
Fixed Step Size ODE Solvers with interpolation support.
====================================================================
'''

class FixedStepSizeODESolver(object):
  def __init__(
    self, 
    velo_func, 
    y_0, 
    step_size = None, 
    grid_constructor = None, 
    interpolation = 'linear', 
    perturb = False
  ):
    self.velo_func = velo_func
    self.y_0 = y_0 
    self.dtype = y_0.dtype 
    self.device = y_0.device 
    self.step_size = step_size
    self.interpolation = interpolation
    self.perturb = perturb 

    if step_size is None : 
      if grid_constructor is None : 
        self.grid_constructor = lambda f, y_0, t : t 
      else : 
        self.grid_constructor = grid_constructor 
    else : 
      if grid_constructor is None : 
        self.grid_constructor = self.grid_constructor_from_step_size(step_size)
      else : 
        raise ValueError("Cannot specify both `step_size` and `grid_constructor`")  
    

  def grid_constructor_from_step_size(self, step_size):
    def grid_constructor(t):
      start_time = t[0]
      end_time   = t[-1]
      num_iters = torch.ceil((end_time - start_time) / step_size + 1).item()
      t_infer = torch.arange(0, num_iters, dtype = t.dtype, device = t.device) * step_size + start_time
      t_infer[-1] = t[-1]

      return t_infer
    return grid_constructor
  
  @abc.abstractmethod
  def step_func(self, func, t0, dt, t1, y0):
    pass
  
  def integrate(self, t):
    time_grid = self.grid_constructor(self.velo_func, self.y_0, t)

    results = torch.empty(len(t), *self.y_0.shape, dtype = self.dtype, device = self.device)
    results[0] = self.y_0 

    j = 1 
    y_0 = self.y_0

    for t_0, t_1 in zip(time_grid[:-1], time_grid[1:]):
      dt = (t_1 - t_0)
      ones = torch.ones((self.y_0.shape[0], self.y_0.shape[1] - 1)) 


      dy, f_0 = self.step_func(
        func = self.velo_func, 
        t_0 = ones * t_0, 
        dt  = ones * dt, 
        t_1 = ones * t_1, 
        y_0 = y_0, 
      )
      y_1 = y_0 + dy

      while j < len(t) and t[j] <= t_1 : 
        if self.interpolation == 'linear':
          results[j] = self.linear_interpolation(t_0, t_1, y_0, y_1, t[j])
        elif self.interpolation == 'cubic': 
          f_1 = self.velo_func(t_1, y_1)
          results[j] = self.cubic_interpolation(t_0, y_0, f_0, t_1, y_1, f_1, t[j])
        else:
            raise ValueError(f"Unknown interpolation method {self.interp}")
        j += 1

      y_0 = y_1
    return results

  def linear_interpolation(self, t_0, t_1, y_0, y_1, t):
    if t == t_0 : 
      return y_0 
    if t == t_1 : 
      return y_1 

    slope = (t - t_0) / (t_1 - t_0) 
    return y_0 + slope * (y_1 - y_0)

  def cubic_interpolation(self, t_0, y_0, f_0, t_1, y_1, f_1, t):
    h = (t - t_0) / (t_1 - t_0)
    h00 = (1 + 2 * h) * (1 - h) * (1 - h)
    h10 = h * (1 - h) * (1 - h)
    h01 = h * h * (3 - 2 * h)
    h11 = h * h * (h - 1)
    dt = (t_1 - t_0)
    return h00 * y_0 + h10 * dt * f_0 + h01 * y_1 + h11 * dt * f_1


class EulerSolver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    v_0 = func(t_0, y_0)
    return dt * v_0, v_0
  
class MidPointSolver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    half_dt = dt / 2
    v_0 = func(t_0, y_0) 
    mid_point = y_0 + half_dt * v_0 
    return dt * func(t_0 + half_dt, mid_point), v_0
  
class Heun2Solver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    v_0 = func(t_0, y_0)
    # from butcher books
    butcher_tableu = [
      [0.0, 0.0, 0.0],
      [1.0, 1.0, 0.0],
      [0.0, 1/2, 1/2],
    ]

    k_1 = v_0
    k_2 = func(t_0 + dt * butcher_tableu[1][0], y_0 + dt * butcher_tableu[1][1] * k_1)
    return dt * (butcher_tableu[2][1] * k_1 + butcher_tableu[2][2] * k_2), v_0

class Heun3Solver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    v_0 = func(t_0, y_0)
    # from butcher books
    butcher_tableu = [
      [0.0, 0.0, 0.0, 0.0],
      [1/3, 1/3, 0.0, 0.0],
      [2/3, 0.0, 2/3, 0.0],
      [0.0, 1/4, 0.0, 3/4],
    ]


    k_1 = v_0
    k_2 = func(t_0 + dt * butcher_tableu[1][0], y_0 + dt * butcher_tableu[1][1] * k_1)
    k_3 = func(t_0 + dt * butcher_tableu[2][0], y_0 + dt * (butcher_tableu[2][1] * k_1 + butcher_tableu[2][2] * k_2))
    return dt * (butcher_tableu[3][1] * k_1 + butcher_tableu[3][2] * k_2 + butcher_tableu[3][3] * k_3), v_0


class RungeKutta4Solver(FixedStepSizeODESolver):
  def step_func(self, func, t_0, dt, t_1, y_0):
    v_0 = func(t_0, y_0)
    # sampling the trajectory field
    k_1 = v_0 
    k_2 = func(t_0 + dt / 3, y_0 + dt / 3 * k_1)
    k_3 = func(t_0 + dt * 2 / 3, y_0 + dt * (k_2 - k_1 / 3))
    k_4 = func(t_1, y_0 + dt * (k_1 - k_2 + k_3))
    return (k_1 + 3 * (k_2 + k_3) + k_4) * dt * 0.125, v_0


'''
====================================================================
Adaptive Step Size ODE Solvers with interpolation support.
====================================================================
'''
class AdaptiveStepsizeODESolver(abc.ABCMeta):
  def __init__(self, dtype, y_0, norm):
    self.y_0 = y_0
    self.dtype = dtype
    self.norm = norm

  @abc.abstractmethod
  def prepare_integrate(self, t):
      pass

  @abc.abstractmethod
  def advance(self, next_t):
    raise NotImplementedError

  @classmethod
  def valid_callbacks(cls):
    return set()

  def integrate(self, t):
    solution = torch.empty(len(t), *self.y0.shape, dtype = self.y0.dtype, device = self.y0.device)
    solution[0] = self.y_0
    t = t.to(self.dtype)
    self.prepare_integrate(t)
    for i in range(1, len(t)):
      solution[i] = self.advance(t[i])
    return solution


class AdaptiveStepsizeEventODESolver(AdaptiveStepsizeODESolver, abc.ABCMeta):
  @abc.abstractmethod
  def advance_until_event(self, event_fn):
    raise NotImplementedError

  def integrate_until_event(self, t_0, event_fn):
    t0 = t0.to(self.y0.device, self.dtype)
    self.prepare_integrate(t0.reshape(-1))
    event_time, y_1 = self.advance_until_event(event_fn)
    solution = torch.stack([self.y_0, y_1], dim=0)
    return event_time, solution

@dataclass
class Tableau:
  alpha : torch.Tensor
  beta : list[torch.Tensor]
  c_sol : torch.Tensor
  c_error: torch.Tensor

  def to(self, device, dtype):
    return Tableau(
      alpha = self.alpha.to(device = device, dtype = dtype),
      beta = [b.to(device = device, dtype = dtype) for b in self.beta],
      c_sol = self.c_sol.to(device = device, dtype = dtype),
      c_error = self.c_error.to(device = device, dtype = dtype)
    )
  
@dataclass
class RungeKuttaState:
  y_1 : torch.Tensor
  f_1 : torch.Tensor
  t_0 : torch.Tensor
  t_1 : torch.Tensor
  dt : torch.Tensor
  interpolation_coef : torch.Tensor

  def to(self, device, dtype):
    return RungeKuttaState(
      y_1 = self.y_1.to(device = device, dtype = dtype),
      f_1 = self.f_1.to(device = device, dtype = dtype),
      t_0 = self.t_0.to(device = device, dtype = dtype),
      t_1 = self.t_1.to(device = device, dtype = dtype),
      dt = self.dt.to(device = device, dtype = dtype),
      interpolation_coef = self.interpolation_coef.to(device = device, dtype = dtype)
    )
  

def sort_tvals(tvals, t0):
  tvals = tvals[tvals >= t0]
  return torch.sort(tvals).values

def interpolation_fit(y_0, y_1, y_mid, f_0, f_1, dt):
  """Fit coefficients for 4th order polynomial interpolation.
    Args:
      y0: function value at the start of the interval.
      y1: function value at the end of the interval.
      y_mid: function value at the mid-point of the interval.
      f0: derivative value at the start of the interval.
      f1: derivative value at the end of the interval.
      dt: width of the interval.
    Returns:
      List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
      `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
      between 0 (start of interval) and 1 (end of interval).
  """
  a = 2 * dt * (f_1 - f_0) - 8 * (y_1 + y_0) + 16 * y_mid
  b = dt * (5 * f_0 - 3 * f_1) + 18 * y_0 + 14 * y_1 - 32 * y_mid
  c = dt * (f_1 - 4 * f_0) - 11 * y_0 - 5 * y_1 + 16 * y_mid 
  d = dt * f_0
  e = y_0
  return [e, d, c, b, a]

def interpolate_eval(coefficients, t_0, t_1, t):
  """Evaluate polynomial interpolation at the given time point.

    Args:
      coefficients: list of Tensor coefficients as created by `interp_fit`.
      t0: scalar float64 Tensor giving the start of the interval.
      t1: scalar float64 Tensor giving the end of the interval.
      t: scalar float64 Tensor giving the desired interpolation point.

    Returns:
      Polynomial interpolation of the coefficients at time `t`.
  """
  assert (t_0 <= t) & (t <= t_1), f'invalid interpolation, fails {t_0} <= {t} <= {t_1}'
  x = (t - t_0) / (t_1 - t_0)
  x = x.to(coefficients[0].dtype)

  total = coefficients[0] + x * coefficients[1]
  x_pow = x 
  for coefficient in coefficient[2:] : 
    x_pow = x_pow * x 
    total = total + x_pow * coefficient
  return total 


def select_initial_step(velo_func, t_0, y_0, order, rtol, atol, norm, v_0 = None):
  """Empirically select a good initial step.
    The algorithm is described in [1]_.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
        https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/misc.py#L36   
  """
  dtype = y_0.dtype 
  device = y_0.device 
  
  if v_0 is None : 
    v_0 = velo_func(t_0, y_0)

  scale = atol + torch.abs(y_0) * rtol

  d_0 = norm(y_0 / scale).abs()
  d_1 = norm(v_0 / scale).abs()  

  if d_0 < 1e-5 or d_1 < 1e-5:
    h_0 = torch.tensor(1e-6, dtype = dtype, device = device)
  else : 
    h_0 = 0.01 * d_0 / d_1

  y_1 = y_0 + h_0 * v_0
  v_1 = velo_func(t_0 + h_0, y_1)  

  d_2 = torch.abs(norm(v_1 - v_0) / scale / h_0)

  if d_1 <= 1e-15 and d_2 <= 1e-15:
    h_1 = torch.max(torch.tensor(1e-6, dtype = dtype, device = device), h_0 * 1e-3)
  else:
    h_1 = (0.01 / max(d_1, d_2)) ** (1. / float(order + 1))
  
  h_1 = h_1.abs()

  return torch.min(100 * h_0, h_1).to(t_0.dtype)


def compute_error_ratio(error_estimate, rtol, atol, y_0, y_1, norm):
  error_tol = atol + rtol * torch.max(y_0.abs(), y_1.abs())
  return norm(error_estimate / error_tol).abs()

@torch.no_grad()
def optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
  """
  Compute the optimal step size for the next step 
  """
  if error_ratio == 0:
    return last_step * ifactor
  if error_ratio < 1. : 
    dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)

  error_ratio = error_ratio.type_as(last_step)
  exponent = torch.tensor(order, dtype = last_step.dtype, device = last_step.device).reciprocal() # 1 / x
  factor = torch.min(ifactor, torch.max(safety / error_ratio ** exponent, dfactor))

  return last_step * factor 

class _UncheckedAssign(torch.autograd.Function):
  @staticmethod
  def forward(ctx, scratch, value, index): 
    ctx.index = index 
    scratch.data[index] = value 
    return scratch 
  
  @staticmethod 
  def backward(ctx, grad_scratch):
    return grad_scratch, grad_scratch[ctx.index], None 


def runge_kutta_step(velo_func, y_0, f_0, t_0, dt, t_1, tableau):
  t_dtype = y_0.abs().dtype
  t_0 = t_0.to(t_dtype)
  t_1 = t_1.to(t_dtype) 
  dt  = dt.to(t_dtype)

  k = torch.empty(*f_0.shape, len(tableau.alpha) + 1, dtype = y_0.dtype, device = y_0.device) 
  k = _UncheckedAssign.apply(k, f_0 (..., 0))

  for i, (alpha_i, beta_i) in enumerate(tableau.alpha, tableau.beta):
    if alpha_i == 1. : 
      t_i = t_1 
    else : 
      t_i = t_0 + alpha_i * dt 
  
    y_i = y_0 + torch.sum(k[..., : (i + 1)] * (beta_i * dt), dim = -1).view_as(f_0)
    f = velo_func(t_i, y_i)
    k = _UncheckedAssign.apply(k, f_0 (..., 0))
  
  if not (tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()):
    y_i = y_0 + torch.sum(k * (dt * tableau.c_sol), dim = -1).view_as(f_0)
  
  y_1 = y_i
  f_1 = k[..., -1]
  y_1_error = torch.sum(k * (dt * tableau.c_error), dim = -1)
  return y_1, f_1, y_1_error, k


def find_event(interpolation_func, sign_, t_0, t_1, event_func, tol):
  with torch.no_grad():
    num_iters = torch.ceil(torch.log((t_1 - t_0) / tol) / math.log(2.0))
    for _ in range(num_iters.long()):
      t_mid = (t_1 + t_0) / 2.0 
      y_mid = interpolation_func(t_mid)
      sign_mid = torch.sign(event_func(t_mid, y_mid))
      same_sign = (sign_mid == sign_)
      t_0 = torch.where(same_sign, t_mid, t_0)
      t_1 = torch.where(same_sign, t_1, t_mid)
    
    event_t = (t_0 + t_1) / 2

    return event_t, interpolation_func(event_t)



class RungeKuttaAdaptiveStepSizeOdeSolver(AdaptiveStepsizeEventODESolver):
  mid : torch.Tensor
  tableau : Tableau
  order : int 
  def __init__(
    self, 
    velo_func, 
    y_0, 
    rtol, 
    atol, 
    min_step = 0, 
    max_step = float('inf'), 
    first_step = None, 
    step_t = None, 
    jump_t = None, 
    safety = 0.9, 
    ifactor = 10.0, 
    dfactor = 0.2, 
    max_num_steps = 2 ** 31 - 1, 
    dtype = torch.float64, 
  ):
    if y_0.dtype != torch.float64:
     y_0 = y_0.to(torch.float64)

    self.device = y_0.device 
    self.velo_func = velo_func 
    self.rtol = torch.as_tensor(rtol, dtype = dtype, device = self.device)
    self.atol = torch.as_tensor(atol, dtype = dtype, device = self.device)
    self.min_step = torch.as_tensor(min_step, dtype = dtype, device = self.device)
    self.max_step = torch.as_tensor(max_step, dtype=dtype, device = self.device)
    self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype = dtype, device = self.device)
    self.safety = torch.as_tensor(safety, dtype = dtype, device = self.device)
    self.ifactor = torch.as_tensor(ifactor, dtype = dtype, device = self.device)
    self.dfactor = torch.as_tensor(dfactor, dtype = dtype, device = self.device)
    self.max_num_steps = torch.as_tensor(max_num_steps, dtype = torch.int32, device = self.device)
    self.dtype = dtype
    self.step_t = None if step_t is None else torch.as_tensor(step_t, dtype = dtype, device = self.device)
    self.jump_t = None if jump_t is None else torch.as_tensor(jump_t, dtype = dtype, device = self.device)

    self.tableau = self.tableau.to(device = self.device, dtype = y_0.dtype)
    self.mid = self.mid.to(device = self.device, dtype = y_0.dtype)

  def prepare_integrate(self, t):
    t_0 = t[0] 
    v_0 = self.velo_func(t[0], self.y_0)
    
    if self.first_step is None : 
      first_step = select_initial_step(
        self.velo_func, 
        self.y_0, 
        self.order - 1, 
        self.rtol, 
        self.atol, 
        self.norm, 
        v_0, 
      )
    else : 
      first_step = self.first_step
    
    self.rk_state = RungeKuttaState(
      y_0 = self.y_0, 
      f_1 = v_0, 
      t_0 = t[0], 
      t_1 = t[0], 
      dt = first_step, 
      interpolation_coef = [self.y_0] * 5, 
    )

    if self.step_t is None : 
      step_t = torch.tensor([], dtype = self.dtype, device = self.device)
    else : 
      step_t = sort_tvals(self.jump_t, t_0)
      step_t = step_t.to(self.dtype)

    if self.jump_t is None : 
      jump_t = torch.tensor([], dtype = self.dtype, device = self.device)
    else : 
      jump_t = sort_tvals(self.jump_t, t_0)
      jump_t = jump_t.to(self.dtype)
    
    counts = torch.cat([step_t, jump_t]).unique(return_counts = True)[1]
    if (counts > 1).any():
      raise ValueError("`step_t` and `jump_t` must not have any repeated elements between them.")

    self.step_t = step_t 
    self.jump_t = jump_t
    
    self.next_step_index = min(bisect.bisect(self.step_t.tolist(), t[0]), len(self.step_t) - 1) 
    self.next_jump_index = min(bisect.bisect(self.jump_t.tolist(), t[0]), len(self.jump_t) - 1)


  def advance(self, next_t):
    num_steps = 0
    while next_t > self.rk_state.t_1 : 
      assert num_steps < self.max_num_steps, f'max_num_steps exceed {num_steps} >= {self.max_num_steps}'
      self.rk_state = self.adaptive_step(self.rk_state)
      num_steps += 1 
    return interpolate_eval(
      self.rk_state.interpolation_coef, 
      self.rk_state.t_0, 
      self.rk_state.t_1, 
      next_t
    )

  def advance_until_event(self, event_fn):
    if event_fn(self.rk_state.t_1, self.rk_state.y_1) == 0: 
      return (self.rk_state.t_1, self.rk_state.y_1)
    
    num_steps = 0 
    sign = torch.sign(event_fn(self.rk_state.t_1, self.rk_state.y_1))
    
    while sign == torch.sign(event_fn(self.rk_state.t_1, self.rk_state.y_1)):
      assert num_steps < self.max_num_steps, f'max_num_steps exceed {num_steps} >= {self.max_num_steps}'
      self.rk_state = self.adaptive_step(self.rk_state)
      num_steps += 1
    interpoloate_fn = lambda t : interpolate_eval(
      self.rk_state.interpolation_coef, 
      self.rk_state.t_0, 
      self.rk_state.t_1, 
      t, 
    )

    return find_event(
      interpoloate_fn, 
      sign, 
      self.rk_state.t_0, 
      self.rk_state.t_1, 
      event_fn, 
      self.atol, 
    )

  def adaptive_step(self, runge_kutta_state):
    y_0, t_0, dt, interpolation_coef, f_0 = self.rk_state.y_1, self.rk_state.t_1, \
      self.rk_state.dt, self.rk_state.interpolation_coef, self.rk_state.f_1

    if not torch.isfinite(dt):
      dt = self.min_step

    dt = dt.clamp(self.min_step, self.max_step)
    self.velo_func.call

    t_1 = t_0 + dt 
    
    assert t_0 + dt > t_0, f'underflow in dt {dt.item()}'
    assert torch.isfinite(y_0).all(), f'non-finite values in state `y`: {y_0}'

    on_step_t = False 
    if len(self.step_t):
      next_step_t = self.step_t[self.next_step_index]
      on_step_t = t_0 < next_step_t < t_0 + dt
      if on_step_t : 
        t_1 = next_step_t 
        dt = t_1 - t_0

    on_jump_t = False
    if len(self.jump_t):
      next_jump_t = self.jump_t[self.next_jump_index]
      on_jump_t = t_0 < next_jump_t < t_0 + dt
      if on_jump_t : 
        t_1 = next_jump_t 
        dt = t_1 - t_0
    
    y_1, f_1, y_1_error, k = runge_kutta_step(
      self.velo_func, 
      y_0 = y_0, 
      f_0 = f_0, 
      t_0 = t_0,
      t_1 = t_1,  
      dt = dt, 
      tableau = self.tableau, 
    )

    error_ratio = compute_error_ratio(y_1_error, self.rtol, self.atol, y_0, y_1, self.norm)

    accept_state = error_ratio <= 1 

    if dt > self.max_step:
      accept_step = False
    if dt <= self.min_step:
      accept_step = True
    

    if accept_state:
      t_next = t_1 
      y_next = y_1
      interpolation_coef = self.interpolation_fit(
        y_0 = y_0, 
        y_1 = y_next, 
        k = k, 
        dt = dt
      )
      if on_step_t:
        if self.next_step_index < len(self.step_t):
          self.next_step_index += 1 
      
      if on_jump_t:
        if self.next_jump_index < len(self.jump_t):
          self.next_jump_index += 1 
        f_1 = self.velo_func(t_next, y_next)
      f_next = f_1
    else : 
      t_next, y_next, f_next = t_0, y_0, f_0
    
    dt_next = optimal_step_size(
      last_step = dt, 
      error_ratio = error_ratio, 
      safety = self.safety, 
      ifactor = self.ifactor, 
      dfactor = self.dfactor, 
      order = self.order, 
    )

    dt_next = dt_next.clamp(self.min_step, self.max_step)

    rk_state = RungeKuttaState(
      y_1 = y_next, 
      f_1 = f_next,
      t_0 = t_0, 
      t_1 = t_next, 
      dt = dt_next, 
      interpolation_coef = interpolation_coef 
    )

    return rk_state


  def interpolation_fit(self, y_0, y_1, k, dt):
    dt = dt.type_as(y_0)
    y_mid = y_0 + torch.sum(k * (dt * self.mid), dim = -1).view_as(y_0)
    f_0 = k[..., 0]
    f_1 = k[..., -1]
    return interpolation_fit(y_0, y_1, y_mid, f_0, f_1, dt)


# DOPRI5
#=======================================================================================
DORMAND_PRINCE_SHAMPINE_TABLEAU = Tableau(
  alpha = torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1.], dtype=torch.float64),
  beta = [
    torch.tensor([1 / 5], dtype=torch.float64),
    torch.tensor([3 / 40, 9 / 40], dtype=torch.float64),
    torch.tensor([44 / 45, -56 / 15, 32 / 9], dtype=torch.float64),
    torch.tensor([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], dtype=torch.float64),
    torch.tensor([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], dtype=torch.float64),
    torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84], dtype=torch.float64),
  ],
  c_sol = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=torch.float64),
  c_error = torch.tensor([
    35 / 384 - 1951 / 21600,
    0,
    500 / 1113 - 22642 / 50085,
    125 / 192 - 451 / 720,
    -2187 / 6784 - -12231 / 42400,
    11 / 84 - 649 / 6300,
    -1. / 60.,
  ], dtype=torch.float64), 
)

DPS_C_MID = torch.tensor([
    6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2, -2691868925 / 45128329728 / 2,
    187940372067 / 1594534317056 / 2, -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2
], dtype = torch.float64)


class Dopri5Solver(RungeKuttaAdaptiveStepSizeOdeSolver):
  order = 5
  tableau = DORMAND_PRINCE_SHAMPINE_TABLEAU
  mid = DPS_C_MID

# DOPRI8 
#=======================================================================================
A = [1 / 18, 1 / 12, 1 / 8, 5 / 16, 3 / 8, 59 / 400, 93 / 200, 5490023248 / 9719169821, 13 / 20, 1201146811 / 1299019798, 1, 1, 1]

B = [
    [1 / 18],

    [1 / 48, 1 / 16],

    [1 / 32, 0, 3 / 32],

    [5 / 16, 0, -75 / 64, 75 / 64],

    [3 / 80, 0, 0, 3 / 16, 3 / 20],

    [29443841 / 614563906, 0, 0, 77736538 / 692538347, -28693883 / 1125000000, 23124283 / 1800000000],

    [16016141 / 946692911, 0, 0, 61564180 / 158732637, 22789713 / 633445777, 545815736 / 2771057229, -180193667 / 1043307555],

    [39632708 / 573591083, 0, 0, -433636366 / 683701615, -421739975 / 2616292301, 100302831 / 723423059, 790204164 / 839813087, 800635310 / 3783071287],

    [246121993 / 1340847787, 0, 0, -37695042795 / 15268766246, -309121744 / 1061227803, -12992083 / 490766935, 6005943493 / 2108947869, 393006217 / 1396673457, 123872331 / 1001029789],

    [-1028468189 / 846180014, 0, 0, 8478235783 / 508512852, 1311729495 / 1432422823, -10304129995 / 1701304382, -48777925059 / 3047939560, 15336726248 / 1032824649, -45442868181 / 3398467696, 3065993473 / 597172653],

    [185892177 / 718116043, 0, 0, -3185094517 / 667107341, -477755414 / 1098053517, -703635378 / 230739211, 5731566787 / 1027545527, 5232866602 / 850066563, -4093664535 / 808688257, 3962137247 / 1805957418, 65686358 / 487910083],

    [403863854 / 491063109, 0, 0, -5068492393 / 434740067, -411421997 / 543043805, 652783627 / 914296604, 11173962825 / 925320556, -13158990841 / 6184727034, 3936647629 / 1978049680, -160528059 / 685178525, 248638103 / 1413531060, 0],

    [14005451 / 335480064, 0, 0, 0, 0, -59238493 / 1068277825, 181606767 / 758867731, 561292985 / 797845732, -1041891430 / 1371343529, 760417239 / 1151165299, 118820643 / 751138087, -528747749 / 2220607170, 1 / 4]
]

C_sol = [14005451 / 335480064, 0, 0, 0, 0, -59238493 / 1068277825, 181606767 / 758867731, 561292985 / 797845732, -1041891430 / 1371343529, 760417239 / 1151165299, 118820643 / 751138087, -528747749 / 2220607170, 1 / 4, 0]

C_err = [14005451 / 335480064 - 13451932 / 455176623, 0, 0, 0, 0, -59238493 / 1068277825 - -808719846 / 976000145, 181606767 / 758867731 - 1757004468 / 5645159321, 561292985 / 797845732 - 656045339 / 265891186, -1041891430 / 1371343529 - -3867574721 / 1518517206, 760417239 / 1151165299 - 465885868 / 322736535, 118820643 / 751138087 - 53011238 / 667516719, -528747749 / 2220607170 - 2 / 45, 1 / 4, 0]

h = 1 / 2

C_mid = [0.] * 14

C_mid[0] = (- 6.3448349392860401388 * (h**5) + 22.1396504998094068976 * (h**4) - 30.0610568289666450593 * (h**3) + 19.9990069333683970610 * (h**2) - 6.6910181737837595697 * h + 1.0) / (1 / h)

C_mid[5] = (- 39.6107919852202505218 * (h**5) + 116.4422149550342161651 * (h**4) - 121.4999627731334642623 * (h**3) + 52.2273532792945524050 * (h**2) - 7.6142658045872677172 * h) / (1 / h)

C_mid[6] = (20.3761213808791436958 * (h**5) - 67.1451318825957197185 * (h**4) + 83.1721004639847717481 * (h**3) - 46.8919164181093621583 * (h**2) + 10.7281392630428866124 * h) / (1 / h)

C_mid[7] = (7.3347098826795362023 * (h**5) - 16.5672243527496524646 * (h**4) + 9.5724507555993664382 * (h**3) - 0.1890893225010595467 * (h**2) + 0.5526637063753648783 * h) / (1 / h)

C_mid[8] = (32.8801774352459155182 * (h**5) - 89.9916014847245016028 * (h**4) + 87.8406057677205645007 * (h**3) - 35.7075975946222072821 * (h**2) + 4.2186562625665153803 * h) / (1 / h)

C_mid[9] = (- 10.1588990526426760954 * (h**5) + 22.6237489648532849093 * (h**4) - 17.4152107770762969005 * (h**3) + 6.2736448083240352160 * (h**2) - 0.6627209125361597559 * h) / (1 / h)

C_mid[10] = (- 12.5401268098782561200 * (h**5) + 32.2362340167355370113 * (h**4) - 28.5903289514790976966 * (h**3) + 10.3160881272450748458 * (h**2) - 1.2636789001135462218 * h) / (1 / h)

C_mid[11] = (29.5553001484516038033 * (h**5) - 82.1020315488359848644 * (h**4) + 81.6630950584341412934 * (h**3) - 34.7650769866611817349 * (h**2) + 5.4106037898590422230 * h) / (1 / h)

C_mid[12] = (- 41.7923486424390588923 * (h**5) + 116.2662185791119533462 * (h**4) - 114.9375291377009418170 * (h**3) + 47.7457971078225540396 * (h**2) - 7.0321379067945741781 * h) / (1 / h)

C_mid[13] = (20.3006925822100825485 * (h**5) - 53.9020777466385396792 * (h**4) + 50.2558364226176017553 * (h**3) - 19.0082099341608028453 * (h**2) + 2.3537586759714983486 * h) / (1 / h)


A = torch.tensor(A, dtype=torch.float64)
B = [torch.tensor(B_, dtype=torch.float64) for B_ in B]
C_sol = torch.tensor(C_sol, dtype=torch.float64)
C_err = torch.tensor(C_err, dtype=torch.float64)
_C_mid = torch.tensor(C_mid, dtype=torch.float64)

DOPRI8_TABLEAU = Tableau(alpha=A, beta=B, c_sol=C_sol, c_error=C_err)


class Dopri8Solver(RungeKuttaAdaptiveStepSizeOdeSolver):
  order = 8
  tableau = DOPRI8_TABLEAU
  mid = _C_mid


# Bosh3 
#=======================================================================================
BOGACKI_SHAMPINE_TABLEAU = Tableau(
  alpha = torch.tensor([1 / 2, 3 / 4, 1.], dtype = torch.float64),
  beta = [
    torch.tensor([1 / 2], dtype = torch.float64),
    torch.tensor([0., 3 / 4], dtype = torch.float64),
    torch.tensor([2 / 9, 1 / 3, 4 / 9], dtype = torch.float64)
  ],
  c_sol = torch.tensor([2 / 9, 1 / 3, 4 / 9, 0.], dtype = torch.float64),
  c_error = torch.tensor([2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8], dtype = torch.float64),
)

BS_C_MID = torch.tensor([0., 0.5, 0., 0.], dtype=torch.float64)


class Bosh3Solver(RungeKuttaAdaptiveStepSizeOdeSolver):
  order = 3
  tableau = BOGACKI_SHAMPINE_TABLEAU
  mid = BS_C_MID


# Tsit 5
#=======================================================================================
_TSITOURAS_TABLEAU = Tableau(
    alpha=torch.tensor([
        161 / 1000,
        327 / 1000,
        9 / 10,
        .9800255409045096857298102862870245954942137979563024768854764293221195950761080302604,
        1,
        1
    ], dtype=torch.float64),
    beta=[
        torch.tensor([161 / 1000], dtype=torch.float64),
        torch.tensor([
            -.8480655492356988544426874250230774675121177393430391537369234245294192976164141156943e-2,
            .3354806554923569885444268742502307746751211773934303915373692342452941929761641411569
        ], dtype=torch.float64),
        torch.tensor([
            2.897153057105493432130432594192938764924887287701866490314866693455023795137503079289,
            -6.359448489975074843148159912383825625952700647415626703305928850207288721235210244366,
            4.362295432869581411017727318190886861027813359713760212991062156752264926097707165077,
        ], dtype=torch.float64),
        torch.tensor([
            5.325864828439256604428877920840511317836476253097040101202360397727981648835607691791,
            -11.74888356406282787774717033978577296188744178259862899288666928009020615663593781589,
            7.495539342889836208304604784564358155658679161518186721010132816213648793440552049753,
            -.9249506636175524925650207933207191611349983406029535244034750452930469056411389539635e-1
        ], dtype=torch.float64),
        torch.tensor([
            5.861455442946420028659251486982647890394337666164814434818157239052507339770711679748,
            -12.92096931784710929170611868178335939541780751955743459166312250439928519268343184452,
            8.159367898576158643180400794539253485181918321135053305748355423955009222648673734986,
            -.7158497328140099722453054252582973869127213147363544882721139659546372402303777878835e-1,
            -.2826905039406838290900305721271224146717633626879770007617876201276764571291579142206e-1
        ], dtype=torch.float64),
        torch.tensor([
            .9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1,
            1 / 100,
            .4798896504144995747752495322905965199130404621990332488332634944254542060153074523509,
            1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331,
            -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677,
            2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841
        ], dtype=torch.float64),
    ],
    c_sol=torch.tensor([
        .9468075576583945807478876255758922856117527357724631226139574065785592789071067303271e-1,
        .9183565540343253096776363936645313759813746240984095238905939532922955247253608687270e-2,
        .4877705284247615707855642599631228241516691959761363774365216240304071651579571959813,
        1.234297566930478985655109673884237654035539930748192848315425833500484878378061439761,
        -2.707712349983525454881109975059321670689605166938197378763992255714444407154902012702,
        1.866628418170587035753719399566211498666255505244122593996591602841258328965767580089,
        1 / 66
    ], dtype=torch.float64),
    c_error=torch.tensor([
        -1.780011052225771443378550607539534775944678804333659557637450799792588061629796e-03,
        -8.164344596567469032236360633546862401862537590159047610940604670770447527463931e-04,
        7.880878010261996010314727672526304238628733777103128603258129604952959142646516e-03,
        -1.44711007173262907537165147972635116720922712343167677619514233896760819649515e-01,
        5.823571654525552250199376106520421794260781239567387797673045438803694038950012e-01,
        -4.580821059291869466616365188325542974428047279788398179474684434732070620889539e-01,
        1 / 66
    ], dtype=torch.float64),
)

x = 1 / 2
TSIT_C_MID = torch.tensor([
    -1.0530884977290216*x*(x-1.329989018975412)*(x*x-1.4364028541716351*x+0.7139816917074209),
    0.1017*x*x*(x*x-2.1966568338249754*x+1.2949852507374631),
    2.490627285651252793*x*x*(x*x-2.38535645472061657*x+1.57803468208092486),
    -16.54810288924490272*(x-1.21712927295533244)*(x-0.61620406037800089)*x*x,
    47.37952196281928122*(x-1.203071208372362603)*(x-0.658047292653547382)*x*x,
    -34.87065786149660974*(x-1.2)*(x-2/3)*x*x,
    2.5*(x-1)*(x-0.6)*x*x
], dtype=torch.float64)

class Tsit5Solver(RungeKuttaAdaptiveStepSizeOdeSolver):
  order = 5
  tableau = _TSITOURAS_TABLEAU
  mid = TSIT_C_MID


# Adaptive Heun 
ADAPTIVE_HEUN_TABLEAU = Tableau(
  alpha = torch.tensor([1.], dtype = torch.float64),
  beta = [torch.tensor([1.], dtype = torch.float64), ],
  c_sol = torch.tensor([0.5, 0.5], dtype = torch.float64),
  c_error=torch.tensor([0.5, -0.5,], dtype = torch.float64),
)

AH_C_MID = torch.tensor([0.5, 0.], dtype = torch.float64)


class AdaptiveHeunSolver(RungeKuttaAdaptiveStepSizeOdeSolver):
  order = 2
  tableau = ADAPTIVE_HEUN_TABLEAU
  mid = AH_C_MID