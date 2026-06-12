import abc
import collections 
import torch 
from dataclasses import field, dataclass 
import bisect
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
  pass 

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


def runge_kutta_step():
  pass 



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
    
    assert t_0 + dt > t_0, f'underflow in dt {dt.item()}')
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
  