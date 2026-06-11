import math 
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch 
import torch.nn.functional as F 


'''
=====================================
Diffusion Transformer 
=====================================
'''

def get_activation(activation, approximate = 'tanh'):
  if activation == 'GeLU':
    return nn.GELU(approximate = approximate)
  elif activation == 'LeakyReLU':
    return nn.LeakyReLU()
  elif activation == 'ELU':
    return nn.ELU()
  elif activation == 'ReLU':
    return nn.ReLU()
  elif activation == 'SiLU':
    return nn.SiLU()
  else : 
    raise NotImplementedError(f'activation : {activation} is currently not supported')
  
def timestep_embedding(timesteps, dim, max_period = 10000):
  """
  Create sinusoidal timestep embeddings.
  :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
  :param dim: the dimension of the output.
  :param max_period: controls the minimum frequency of the embeddings.
  :return: an [N x dim] Tensor of positional embeddings.
  """
  half = dim // 2
  freqs = torch.exp(-math.log(max_period) * torch.arange(start = 0, end = half, \
      dtype = torch.float32) / half).to(device = timesteps.device)
  args = timesteps[:, None].float() * freqs[None]
  embedding = torch.cat([torch.cos(args), torch.sin(args)], dim = -1)
  if dim % 2:
    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim = -1)
  return embedding

def get_1d_pos_embedding_grid(embedding_dim, position, temperature = 10000):
  assert embedding_dim % 2 == 0
  omega = np.arange(embedding_dim // 2, dtype = np.float32)
  omega /= embedding_dim / 2.
  omega = 1. / temperature ** omega  # (D / 2,)
  position = position.reshape(-1)  # (M,)
  out = np.einsum('m,d->md', position, omega)

  embedding_sin = np.sin(out)
  embedding_cos = np.cos(out) 

  embedding = np.concatenate([embedding_sin, embedding_cos], axis = 1)  # (M, D)
  return embedding


def get_2d_pos_embedding_grid(embedding_dim, grid, temperature = 10000):
  assert embedding_dim % 2 == 0
  embedding_h = get_1d_pos_embedding_grid(embedding_dim // 2, grid[0], temperature)  
  embedding_w = get_1d_pos_embedding_grid(embedding_dim // 2, grid[1], temperature)  
  embedding = np.concatenate([embedding_h, embedding_w], axis = 1) # (H * W, D)
  return embedding

def get_2d_pos_embedding(embedding_dim, grid_size, class_token = False, extra_tokens = 0):
    grid_h = np.arange(grid_size, dtype = np.float32)
    grid_w = np.arange(grid_size, dtype = np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis = 0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_pos_embedding_grid(embedding_dim, grid)
    if class_token and extra_tokens > 0:
      pos_embed = np.concatenate([np.zeros([extra_tokens, embedding_dim]), pos_embed], axis=0)
    return pos_embed


class MultiheadAttention(nn.Module): 
  def __init__(self, num_heads, hidden_size): 
    super(MultiheadAttention, self).__init__()
    self.num_heads = num_heads 
    self.hidden_size = hidden_size
    self.proj_qkv = nn.Linear(self.hidden_size, self.hidden_size * 3)

  def forward(self, x): 
    """
    :query, key, value: tensor of shape [batch_size, n_channels, hidden_size] 
    :return : 
      attn_out : tensor of shape [batch_size, n_channels, hidden_size]
    """

    batch_size, n_channels, hidden_size = x.shape
    assert n_channels % self.num_heads == 0 

    qkv = self.proj_qkv(x)
    query, key, value = qkv.chunk(3, dim = 2)

    channels_per_head = n_channels // self.num_heads 
    # (batch_size, num_heads, channels_per_head, hidden_size)
    query = query.contiguous().view(batch_size, self.num_heads, channels_per_head, -1)
    key   = key.contiguous().view(batch_size, self.num_heads,   channels_per_head, -1)
    value = value.contiguous().view(batch_size, self.num_heads, channels_per_head, -1)

    # (batch_size * num_heads, channels_per_head, hidden_size)

    query = query.contiguous().view(batch_size * self.num_heads, channels_per_head, -1)
    key   = key.contiguous().view(batch_size * self.num_heads, channels_per_head, -1)
    value = value.contiguous().view(batch_size * self.num_heads, channels_per_head, -1)

    matching_mat  = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(hidden_size)
    matching_mat  = torch.softmax(matching_mat, dim = -1) # (batch_size * num_heads, channels_per_head, hidden_size) 


    attn_out = torch.bmm(matching_mat, value) 
    attn_out = attn_out.contiguous().view(batch_size, self.num_heads, channels_per_head, -1) 
    attn_out = attn_out.contiguous().view(batch_size, self.num_heads * channels_per_head, -1)
    return attn_out 


class TransformerBlock(nn.Module):
 def __init__(self, hidden_size, num_heads, ffn_dim, dropout, activation = 'GeLU'):
   super(TransformerBlock, self).__init__()
   self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine = False, eps = 1e-6)
   self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine = False, eps = 1e-6)
   self.attn = MultiheadAttention(num_heads = num_heads, hidden_size = hidden_size) 
   self.mlp = nn.Sequential(
     nn.Linear(hidden_size, ffn_dim),
     get_activation(activation = activation),  
     nn.Dropout(p = dropout), 
     nn.Linear(ffn_dim, hidden_size), 
   )
   self.adaln_module = nn.Sequential(
    nn.SiLU(), 
    nn.Linear(hidden_size, 6 * hidden_size, bias = True)
   )
 def forward(self, x, cond):
   mean_mha, std_mha, gate_mha, mean_mlp, std_mlp, gate_mlp = self.adaln_module(cond).chunk(6, dim = 1)
   x = x + gate_mha.unsqueeze(1) * self.attn((std_mha.unsqueeze(1) + 1.) * self.norm1(x) + mean_mha.unsqueeze(1))
   x = x + gate_mlp.unsqueeze(1) * self.mlp( (std_mlp.unsqueeze(1) + 1.) * self.norm2(x) + mean_mlp.unsqueeze(1))
   return x 

class TransformerFinalBlock(nn.Module):
  def __init__(self, hidden_size, patch_size, out_channels):
    super(TransformerFinalBlock, self).__init__()
    self.hidden_size = hidden_size
    self.patch_size = patch_size 
    self.out_channels = out_channels

    self.norm = nn.LayerNorm(hidden_size, elementwise_affine = False, eps = 1e-6)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias = True)
    self.adaln_module = nn.Sequential(
      nn.SiLU(), 
      nn.Linear(hidden_size, 2 * hidden_size, bias = True) 
    )
  
  def forward(self, x, cond):
    mean, std = self.adaln_module(cond).chunk(2, dim = 1)
    x = self.norm(x) * (std.unsqueeze(1) + 1) + mean.unsqueeze(1) 
    x = self.linear(x)
    return x 
  
class PatchEmbedding(nn.Module):
  def __init__(
    self, 
    img_size = 224,
    patch_size = 16, 
    in_channels = 3, 
    hidden_size = 768, 
    norm_layer = False, 
    flatten = True, 
    bias = True,
    device = 'cuda', 
    dtype = torch.float32, 
  ):
    super(PatchEmbedding, self).__init__()
    self.patch_size = (patch_size, patch_size)
    self.img_size, self.grid_size, self.num_patches = self.clip_img_size(img_size = img_size) 
    self.in_channels = in_channels
    self.hidden_size = hidden_size 
    self.norm_layer = norm_layer
    self.flatten = flatten 
    self.bias = bias 
    self.device = device 
    self.dtype = dtype

    self.conv = nn.Conv2d(
      in_channels = self.in_channels, 
      out_channels = self.hidden_size, 
      kernel_size = self.patch_size, 
      stride = patch_size, 
      bias = self.bias, 
    )

    self.norm = nn.LayerNorm(hidden_size) if self.norm_layer else nn.Identity()

  def clip_img_size(self, img_size): 
    img_size = (img_size, img_size)
    grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
    num_patches = grid_size[0] * grid_size[1]
    return img_size, grid_size, num_patches
  
  def forward(self, x):
    B, C, H, W = x.shape 

    pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
    pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]

    x = F.pad(x, (0, pad_w, 0, pad_h))

    x = self.conv(x) # (B, hidden_size, h, w)

    x = x.flatten(2).transpose(1, 2) # (B, h * w, hidden_size)
    x = self.norm(x)
    return x 


class DiffusionTransformer(nn.Module):
  def __init__(
    self, 
    input_size = 32, 
    patch_size = 2, 
    in_channels = 4, 
    hidden_size = 512, 
    num_transformer_layers = 28, 
    num_attention_heads = 8,
    ffn_dim = 2048,
    activation = 'GeLU', 
    learn_sigma = True, 
    dropout_prob = 0.1, 
    frequency_embedding_size = 256, 
    num_classes = None, 
    device = 'cuda', 
  ):
    super(DiffusionTransformer, self).__init__()
    self.input_size = input_size
    self.patch_size = patch_size
    self.in_channels = in_channels
    self.hidden_size = hidden_size
    self.num_transformer_layers = num_transformer_layers
    self.num_attention_heads = num_attention_heads
    self.ffn_dim = ffn_dim
    self.dropout_prob = dropout_prob
    self.activation = activation 
    self.learn_sigma = learn_sigma
    self.frequency_embedding_size = frequency_embedding_size
    self.out_channels = in_channels * (2 if self.learn_sigma else 1)
    self.num_classes = num_classes
    self.device = device 

    self.patch_embedder = PatchEmbedding(
      img_size = self.input_size, 
      patch_size = self.patch_size, 
      in_channels = self.in_channels, 
      hidden_size = self.hidden_size, 
      bias = True, 
    )
    

    self.frequency_embedding = nn.Sequential(
      nn.Linear(self.frequency_embedding_size, self.hidden_size), 
      nn.SiLU(), 
      nn.Linear(self.hidden_size, self.hidden_size), 
    )

    self.blocks = nn.ModuleList([
      TransformerBlock(
        hidden_size = self.hidden_size, 
        num_heads = self.num_attention_heads, 
        ffn_dim = self.ffn_dim, 
        dropout = self.dropout_prob, 
        activation = self.activation
      )
      for _ in range(self.num_transformer_layers)
    ])

    self.out = TransformerFinalBlock(
      hidden_size = self.hidden_size, 
      patch_size = self.patch_size, 
      out_channels = self.out_channels, 
    )

    self.initialize_weights()

  def first_init(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.xavier_uniform_(module.weight)
      if module.bias is not None : 
        nn.init.constant_(module.bias, 0)
 
  def initialize_weights(self):
    self.apply(self.first_init)

    positional_embedding_2d = get_2d_pos_embedding(
      self.hidden_size, 
      int(self.patch_embedder.num_patches ** 0.5), 
    )
    
    self.patch_pos_embedding = torch.from_numpy(positional_embedding_2d).float().unsqueeze(0)

    self.patch_pos_embedding = nn.Parameter(self.patch_pos_embedding, requires_grad = False).to(self.device)


    w = self.patch_embedder.conv.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    nn.init.constant_(self.patch_embedder.conv.bias, 0)

    nn.init.normal_(self.frequency_embedding[0].weight, std = 0.02)
    nn.init.normal_(self.frequency_embedding[2].weight, std = 0.02)

    # zero out adaln module, helps avoid instability
    for block in self.blocks: 
      nn.init.constant_(block.adaln_module[-1].weight, 0)
      nn.init.constant_(block.adaln_module[-1].bias, 0)

    nn.init.constant_(self.out.adaln_module[-1].weight, 0)
    nn.init.constant_(self.out.adaln_module[-1].bias, 0)
    nn.init.constant_(self.out.linear.weight, 0)
    nn.init.constant_(self.out.linear.bias, 0)


  def unpatchify(self, x):
    """
    :params x : (B, T, patch_size * patch_size * C)
    """
    out_channels = self.out_channels 
    patch_size = self.patch_embedder.patch_size[0]
    h = w = int(x.shape[1] ** 0.5)

    x = x.contiguous().view(x.shape[0], h, w, patch_size, patch_size, out_channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    imgs = x.contiguous().view((x.shape[0], out_channels, h * patch_size, w * patch_size))
    return imgs

  def forward(self, imgs, timesteps, labels = None):
    """
    :params imgs : (B, C, H, W), tensor of spatial inputs 
    :params timesteps : (B, ), tensor of diffusion or flow matching time steps 
    :params labels : if is not None, then the class labels
    """
    x = self.patch_embedder(imgs) + self.patch_pos_embedding # (B, T, hidden_size) where T = (H * W) / (patch_size ** 2)
    conditional_embedding = self.frequency_embedding(\
      timestep_embedding(timesteps, self.frequency_embedding_size)) # (B, hidden_size)
    
    if self.num_classes is not None : 
      labels_embedding = self.class_embedding(labels)
      conditional_embedding = conditional_embedding + labels_embedding
    
    for block in self.blocks:
      x = block(x, conditional_embedding) # (B, T, hidden_size)
    
    x = self.out(x, conditional_embedding) # (B, T, patch_size ** 2 * out_channels)
    x = self.unpatchify(x) # (B, out_channels, H, W)
    return x  


# if __name__ == '__main__':
#   model = DiffusionTransformer().to('cuda')

#   inp = torch.randn((8, 4, 32, 32)).to('cuda')
#   timesteps = torch.zeros((8, ), dtype = torch.int32).to('cuda')
#   out = model(inp, timesteps)
#   print(out.shape)