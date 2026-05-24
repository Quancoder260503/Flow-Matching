"""
Adapted from 
https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/ode_solver.py
"""

from abc import ABC, abstractmethod 
import torch 
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint


def get_nearest_times(time_grid: Tensor, t_discretization : Tensor) -> Tensor:
    distances = torch.cdist(
        time_grid.unsqueeze(1), 
        t_discretization.unsqueeze(1), 
        compute_mode  = 'donot_use_mm_for_euclid_dist', 
    )
    nearest_distances = distances.argmin(dim = 1)
    return t_discretization[nearest_distances]

class Solver(ABC, nn.Module):
    @abstractmethod 
    def sample(self, x_0 : Tensor = None) -> Tensor :
        pass 

def gradient(output, x, grad_outputs, create_graph):
    """
    Compute the gradient of the inner product of output and grad_outputs w.r.t :math:`x`.

    Args:
        output (Tensor): [N, D] Output of the function.
        x (Tensor): [N, d_1, d_2, ... ] input
        grad_outputs (Optional[Tensor]): [N, D] Gradient of outputs, if `None`,
            then will use a tensor of ones
        create_graph (bool): If True, graph of the derivative will be constructed, allowing
            to compute higher order derivative products. Defaults to False.
    Returns:
        Tensor: [N, d_1, d_2, ... ]. the gradient w.r.t x.
    """
    if grad_outputs is None : 
        grad_outputs = torch.ones_like(output).detach()
    
    grad = torch.autograd.grad(
        output, x, grad_outputs = grad_outputs, create_graph = create_graph 
    )[0] 
    return grad 


class ODESolver(Solver):
    """
    This class is written to solve ordinary differential equations (ODEs) using a specified velocity model. 
    """
    def __init__(self, velocity_model): 
        super(ODESolver).__init__() 
        self.velocity_model = velocity_model 

    def sample(
        self, 
        x_init : Tensor, 
        method = "euler",
        step_size = None, 
        atol = 1e-5, 
        rtol = 1e-5, 
        time_grid = torch.tensor([0.0, 1.0]), 
        intermediate = False, 
        enable_grad = False, 
        **model_kwargs 
    ):
        """
           Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a 
            descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tensor, Sequence[Tensor]]: The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """

        time_grid = time_grid.to(x_init.device) 

        def ode_function(t, x): 
            return self.velocity_model(x = x, t = t, **model_kwargs)
        
        ode_opts = {"step_size" : step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad): 
            ode_solution = odeint(
                ode_function, 
                x_init, 
                time_grid, 
                method  = method, 
                options = ode_opts,
                atol    = atol, 
                rtol    = rtol,  
            )
            if intermediate : 
                return ode_solution
            else : 
                return ode_solution[-1]
        
    def compute_likelihood(
        self, 
        x_1, 
        log_p0, 
        step_size, 
        method = 'euler', 
        atol = 1e-5, 
        rtol = 1e-5, 
        time_grid = torch.Tensor([1.0, 0.0]),
        intermediate = False, 
        exact_divergence = False, 
        enable_grad = False, 
        **model_kwargs 
    ):
        """Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """
        
        assert(time_grid[0] == 1.0 and time_grid[-1] == 0.0), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"
        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_function(x, t): 
            return self.velocity_model(x = x, t = t, **model_kwargs)

        def dynamics_function(t, states): 
            xt = states[0] 
            with torch.enable_grad(True): 
                xt.requires_grad_()
                v_t = ode_function(xt, t) 

                if exact_divergence: 
                    div = 0 
                    for i in range(v_t.flatten(1).shape[1]):
                        g = gradient(v_t[:, i], xt, create_graph = True)[:, i] 
                        if not enable_grad : 
                            g = g.detach()
                        div += g 
                else : 
                    ut_dot_z = torch.einsum("ij,ij->i", v_t.flatten(start_dim = 1), z.flatten(start_dim = 1))   
                    grad_ut_dot_z = gradient(ut_dot_z, xt, create_graph = enable_grad)
                    div = torch.einsum("ij,ij->i", grad_ut_dot_z.flatten(start_dim = 1), z.flatten(start_dim = 1))
                
            if not enable_grad : 
                v_t = v_t.detach()
                div = div.detach()
            
            return v_t, div
        
        y_init = (x_1, torch.zeros(x_1.shape[0], device = x_1.device))
        ode_opts = {"step_size" : step_size} if step_size is not None else {}
        
        with torch.set_grad_enabled(enable_grad): 
            ode_solution, log_det = odeint(
                dynamics_function, 
                y_init, 
                time_grid, 
                method  = method, 
                options = ode_opts,
                atol    = atol, 
                rtol    = rtol,  
            )
        x_source = ode_solution[-1]
        source_log_p = log_p0(x_source) 

        if intermediate:
            return ode_solution, source_log_p + log_det[-1]
        else : 
            return ode_solution[-1], source_log_p + log_det[-1]
    




