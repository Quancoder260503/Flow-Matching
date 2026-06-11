import torch
from torch import nn, tensor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_moons
import Solver


class FlowModel(nn.Module):
  def __init__(self, dim=2, ffn_dim=64):
    super(FlowModel, self).__init__()
    self.dim = dim
    self.ffn_dim = ffn_dim
    self.model = nn.Sequential(
        nn.Linear(dim + 2, ffn_dim),  
        nn.ELU(),
        nn.Linear(ffn_dim, ffn_dim),
        nn.ELU(),
        nn.Linear(ffn_dim, ffn_dim),
        nn.ELU(),
        nn.Linear(ffn_dim, dim),
    )

  def forward(self, t, x, k):
    return self.model(torch.cat([x, t, k], dim=-1))

  def step(self, x_t: tensor, t_start: tensor, t_end: tensor, k: tensor): 
    t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
    t_end   = t_end.view(1, 1).expand(x_t.shape[0], 1)
    dt      = t_end - t_start
    t_mid   = t_start + dt / 2
    v_start = self(t_start, x_t, k)
    x_mid   = x_t + v_start * dt / 2
    v_mid   = self(t_mid, x_mid, k)
    return x_t + dt * v_mid


flow      = FlowModel()
optimizer = torch.optim.Adam(flow.parameters(), lr = 1e-3)
num_epochs = 10000
num_rectified_steps = 5
loss_fn   = nn.MSELoss()

flow.train()


def train(x_0, x_1, k):
  t        = torch.rand(len(x_1), 1)
  k_tensor = torch.full((len(x_1), 1), float(k))  # Fix 2: tuple size + float fill
  x_t      = x_1 * t + x_0 * (1 - t)
  dx_t     = x_1 - x_0
  optimizer.zero_grad()
  net_out  = flow(t, x_t, k_tensor)
  loss     = loss_fn(net_out, dx_t)
  loss.backward()
  optimizer.step()
  return loss.item()


def reflow(x_0, k): 
  with torch.no_grad():
    k_tensor = torch.full((len(x_0), 1), float(k))
    return flow.step(x_0, t_start = torch.tensor(0.0), t_end = torch.tensor(1.0), k = k_tensor)


for i in range(num_epochs):
    data, _ = make_moons(n_samples=2000, noise=0.05)
    data  = (data - data.mean(axis=0)) / data.std(axis=0)

    x_1 = torch.tensor(data, dtype = torch.float32)
    x_0 = torch.randn_like(x_1)

    loss = 0.0

    for k in range(1, num_rectified_steps + 1):
        loss += train(x_0 = x_0, x_1 = x_1, k = k)
        x_1 = reflow(x_0 = x_0, k = k) 

    if i % 100 == 0:
        print(f"step {i} | loss: {loss / num_rectified_steps:.4f}")


x = torch.randn(2000, 2)
n_steps    = 1
time_steps = torch.linspace(0, 1.0, n_steps + 1)

flow.eval()

with torch.no_grad():
    k_infer = torch.full((x.shape[0], 1), float(num_rectified_steps))



fig_anim, ax = plt.subplots(figsize=(4, 4))
ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-3.0, 3.0)
scat = ax.scatter([], [], s=10)
plt.axis("off")


def update(i):
    scat.set_offsets(frames[i].numpy())
    ax.set_title(f"t = {time_steps[i]:.2f}")
    return (scat,)


ani = animation.FuncAnimation(
    fig_anim, update, frames=len(frames), interval=500, blit=True
)
ani.save("flow.gif", writer="pillow", fps=2)
plt.show()
print("Saved to flow.gif")