import math 
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch 
import torch.nn.functional as F 

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
        
def get_conv_by_dim(dims, *args, **kwargs): 
    if  dims == 1: 
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2: 
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3: 
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'dim > 3 are not supported by PyTorch')


def get_avg_pool_by_dim(dims, *args, **kwargs): 
    if  dims == 1: 
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2: 
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3: 
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'dim > 3 are not supported by PyTorch')


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
    freqs = torch.exp(-math.log(max_period) * torch.arange(start = 0, end = half, dtype = torch.float32) / half).to(device = timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim = -1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim = -1)
    return embedding


class PoolMultiheadAttention(nn.Module):
    # Attend over the channels instead of spatial dimensions. 
    def __init__(self, num_heads):
        super(PoolMultiheadAttention, self).__init__()
        self.num_heads = num_heads 
    
    def forward(self, query, key, value):
        """
        :query, key, value: tensor of shape [batch_size, num_heads * channels, spatial_dim] 
        :return : 
           attention module, tensor of shape [batch_size, num_heads * channels, spatial_dim]
        """

        assert query.shape == key.shape == value.shape, "Mismatched shape for Attention Module" 
        batch_size, head_n_channels, spatial_dim = query.shape 
        assert head_n_channels % self.num_heads == 0 

        num_channels = head_n_channels // self.num_heads 
        query = query.contiguous().view(batch_size, self.num_heads, num_channels, -1)
        key   = key.contiguous().view(batch_size, self.num_heads, num_channels, -1)
        value = value.contiguous().view(batch_size, self.num_heads, num_channels, -1)

        query = query.contiguous().view(batch_size * self.num_heads, num_channels, -1)
        key   = key.contiguous().view(batch_size * self.num_heads, num_channels, -1)
        value = value.contiguous().view(batch_size * self.num_heads, num_channels, -1)

        matching_mat  = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(spatial_dim)
        matching_mat  = torch.softmax(matching_mat, dim = -1) # (batch_size * num_heads, num_channels, num_channels)

        attn_out = torch.bmm(matching_mat, value) #(batch_size * num_heads, num_channels, spatial_dim)
        attn_out = attn_out.contiguous().view(batch_size, self.num_heads, num_channels, -1) 
        attn_out = attn_out.contiguous().view(batch_size, self.num_heads * num_channels, -1)
        return attn_out 
    
class AttentionPool2D(nn.Module): 
    def __init__(self, spatial_dim, embedding_dim, num_head_channels, output_dim): 
        super(AttentionPool2D, self).__init__()
        self.spatial_dim        = spatial_dim 
        self.embedding_dim      = embedding_dim 
        self.num_head_channels  = num_head_channels 
        self.output_dim         = output_dim 

        # Learnable positional embedding like ViT
        self.positional_embedding = nn.Parameter(torch.randn(embedding_dim, spatial_dim ** 2 + 1) / embedding_dim ** 0.5)
        
        self.qkv_proj   = get_conv_by_dim(1, embedding_dim, 3 * embedding_dim, 1)
        self.proj_back  = get_conv_by_dim(1, embedding_dim, output_dim, 1)
        self.num_heads = embedding_dim // num_head_channels
        self.attention_module = PoolMultiheadAttention(num_heads = self.num_heads)

    def forward(self, inp): 
        b, c, *spatial = inp.shape
        inp = inp.reshape(b, c, -1)  # (B, C, H * W)
        inp = torch.cat([inp.mean(dim = -1, keepdim = True), inp], dim = -1) # (B, C, 1 + H * W)
        inp = inp + self.positional_embedding[None, :, :].to(inp.dtype)    

        inp = self.qkv_proj(inp) # (B, 3 * C, 1 + H * W)
        query, key, value = torch.chunk(inp, 3, dim = 1) # (B, C, 1 + H * W) each

        out = self.attention_module(query = query, key = key, value = value) # (B, C, 1 + H * W)
        out = self.proj_back(out) # (B, output_dim, 1 + H * W)

        return out[:, :, 0] # (batch_size, output_dim, 1)


class MultiheadAttention(nn.Module): 
    def __init__(self, num_heads): 
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads 

    def forward(self, query, key, value): 
        """
        :query, key, value: tensor of shape [batch_size, num_heads * channels, spatial_dim] 
        :return : 
           attention module, tensor of shape [batch_size, num_heads * channels, spatial_dim]
        """

        assert query.shape == key.shape == value.shape, "Mismatched shape for Attention Module" 
        batch_size, head_n_channels, spatial_dim = query.shape 
        assert head_n_channels % self.num_heads == 0 

        num_channels = head_n_channels // self.num_heads 
        query = query.contiguous().view(batch_size, self.num_heads, num_channels, -1)
        key   = key.contiguous().view(batch_size, self.num_heads, num_channels, -1)
        value = value.contiguous().view(batch_size, self.num_heads, num_channels, -1)

        query = query.contiguous().view(batch_size * self.num_heads, num_channels, -1)
        query = query.permute(0, 2, 1) # (batch_size * num_heads, spatial_dim, num_channels)
        key   = key.contiguous().view(batch_size * self.num_heads, num_channels, -1)
        key   = key.permute(0, 2, 1) # (batch_size * num_heads, spatial_dim, num_channels)
        value = value.contiguous().view(batch_size * self.num_heads, num_channels, -1)
        value = value.permute(0, 2, 1) # (batch_size * num_heads, spatial_dim, num_channels)

        matching_mat  = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(num_channels)
        matching_mat  = torch.softmax(matching_mat, dim = -1) # (batch_size * num_heads, spatial_dim, spatial_dim)

        attn_out = torch.bmm(matching_mat, value) #(batch_size * num_heads, spatial_dim, num_channels)
        attn_out = attn_out.permute(0, 2, 1) # (batch_size * num_heads, num_channels, spatial_dim)
        attn_out = attn_out.contiguous().view(batch_size, self.num_heads, num_channels, -1) 
        attn_out = attn_out.contiguous().view(batch_size, self.num_heads * num_channels, -1)
        return attn_out 
    
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_attention_heads):
        super(AttentionBlock, self).__init__() 
        self.channels = channels 
        self.num_attention_heads = num_attention_heads 

        self.norm = nn.GroupNorm(num_groups = 32, num_channels = channels)
        self.qkv_proj  = get_conv_by_dim(1, channels,  channels * 3, kernel_size = 1) 
        self.attention = MultiheadAttention(num_heads = self.num_attention_heads)
        self.proj_back = zero_module(get_conv_by_dim(1, channels, channels, kernel_size = 1)) # Reduce instability in the early stage of training 

    def forward(self, inp):
        b, c, *spatial = inp.shape
        inp = inp.reshape(b, c, -1) 
        qkv = self.qkv_proj(self.norm(inp)) # (b, 3 * c, spatial_dim) 
        query, key, value = torch.chunk(qkv, 3, dim = 1) # (b, c, spatial_dim)
        out = self.attention(query = query, key = key, value = value) 
        out = self.proj_back(out) 
        out = inp + out 
        out = out.contiguous().view(b, c, *spatial) 
        return out 



    
class Upsample(nn.Module): 
    def __init__(self, channels, use_conv, dims = 2, out_channels = None):
        super(Upsample, self).__init__()
        self.channels     = channels 
        self.out_channels = out_channels if out_channels is not None else channels 
        self.use_conv     = use_conv
        self.dims         = dims 

        if self.use_conv : 
            self.conv = get_conv_by_dim(dims, self.channels, self.out_channels, kernel_size = 3, padding = 1) 

    def forward(self, inp): 
        assert inp.shape[1] == self.channels 
        if self.dims == 3 : 
            inp = F.interpolate(inp, (inp.shape[2], inp.shape[3] * 2, inp.shape[4] * 2), mode = 'nearest')
        else : 
            inp = F.interpolate(inp, scale_factor = 2, mode = 'nearest')
        if self.use_conv : 
            inp = self.conv(inp) 
        return inp        

class Downsample(nn.Module): 
    def __init__(self, channels, use_conv, dims = 2, out_channels = None):
        super(Downsample, self).__init__()
        self.channels     = channels 
        self.out_channels = out_channels if out_channels is not None else channels 
        self.use_conv     = use_conv
        self.dims         = dims 

        stride = 2 if dims != 3 else (1, 2, 2)

        if self.use_conv : 
            self.conv = get_conv_by_dim(dims, self.channels, self.out_channels, kernel_size = 3, stride = stride, padding = 1) 
        else : 
            assert self.channels == self.out_channels 
            self.conv = get_avg_pool_by_dim(dims, kernel_size = stride, stride = stride)

    def forward(self, inp): 
        assert inp.shape[1] == self.channels
        inp = self.conv(inp) 
        return inp        


class ResidualBlock(nn.Module): 
    def __init__(self, 
        channels, 
        embedding_channels, 
        dropout,              
        out_channels         = None, 
        use_conv             = False, 
        use_scale_shift_norm = False, 
        dims                 = 2, 
        use_checkpoint       = False, 
        upsampling           = False, 
        downsampling         = False 
    ):
        super(ResidualBlock, self).__init__()
        self.channels = channels 
        self.embedding_channels = embedding_channels 
        self.dropout_p = dropout               
        self.out_channels = out_channels if out_channels is not None else channels 
        self.use_conv = use_conv 
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint 
        self.dims = dims 
        
        self.upsampling = upsampling 
        self.downsampling = downsampling 

        self.activation   = nn.SiLU()

        self.groupnorm1 = nn.GroupNorm(num_groups = 32, num_channels = channels)
        self.groupnorm2 = nn.GroupNorm(num_groups = 32, num_channels = self.out_channels)
    
        self.conv1 = get_conv_by_dim(dims, channels, self.out_channels, kernel_size = 3, padding = 1)
        self.conv2 = zero_module(get_conv_by_dim(dims, self.out_channels, self.out_channels, kernel_size = 3, padding = 1)) # To reduce the instabilities in the early stage of training 

        self.dropout = nn.Dropout(p = dropout)

        self.linear1 = nn.Linear(embedding_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)

        if self.upsampling : 
            self.residual_update = Upsample(channels = channels, use_conv = False, dims = dims) 
            self.input_update    = Upsample(channels = channels, use_conv = False, dims = dims)  
        elif self.downsampling : 
            self.residual_update = Downsample(channels = channels, use_conv = False, dims = dims) 
            self.input_update    = Downsample(channels = channels, use_conv = False, dims = dims)
        else :
            self.residual_update = nn.Identity()
            self.input_update    = nn.Identity() 

        if self.out_channels == channels : 
            self.skip_conn = nn.Identity() 
        else : 
            self.skip_conn = get_conv_by_dim(dims, channels, self.out_channels, 1)


    def forward(self, inp, embedding): 
        if self.upsampling or self.downsampling : 
            residual = self.groupnorm1(inp)
            residual = self.activation(residual)
            residual = self.residual_update(residual)
            residual = self.conv1(residual)
            inp      = self.input_update(inp) 
        else : 
            residual = self.groupnorm1(inp) 
            residual = self.activation(residual)
            residual = self.conv1(residual)
        
        embedding_out = self.activation(embedding)
        embedding_out = self.linear1(embedding_out) 

        while len(embedding_out.shape) < len(residual.shape): 
            embedding_out = embedding_out[..., None]
        
        if self.use_scale_shift_norm : 
            scale, shift = torch.chunk(embedding_out, chunks = 2, dim = 1) 
            residual = self.groupnorm2(residual)
            residual = residual * (1 + scale) + shift 
            residual = self.activation(residual)
            residual = self.dropout(residual)
            residual = self.conv2(residual)
        else : 
            residual = residual + embedding_out
            residual = self.groupnorm2(residual)
            residual = self.activation(residual)
            residual = self.dropout(residual)
            residual = self.conv2(residual)

        return self.skip_conn(inp) + residual


class Unet(nn.Module): 
    def __init__(
        self, 
        image_size, 
        in_channels, 
        model_channels, 
        out_channels, 
        num_residual_blocks,
        attention_resolutions, 
        dropout = 0.2, 
        channel_mult = (1, 2, 4, 8), 
        conv_resample = True, 
        dims = 2,
        num_classes = None,
        num_attention_heads = 1, 
        use_scale_shift_norm  = True, 
        residual_block_up_down = True, 
        embedding_to_model_dim_ratio = 4,  
        device = 'cuda', 
    ):
        super(Unet, self).__init__()
        self.image_size = image_size 
        self.in_channels = in_channels 
        self.model_channels = model_channels
        self.out_channels = out_channels 
        self.num_residual_blocks = num_residual_blocks 
        self.dropout = nn.Dropout(dropout)
        self.channel_mult = channel_mult 
        self.conv_resample = conv_resample 
        self.dims = dims 
        self.num_classes = num_classes
        self.num_attention_heads = num_attention_heads
        self.use_scale_shift_norm = use_scale_shift_norm 
        self.residual_block_up_down = residual_block_up_down
        self.embedding_to_model_dim_ratio = embedding_to_model_dim_ratio
        self.attention_resolutions = attention_resolutions
        self.device = device


        self.time_embedding_dim = model_channels * embedding_to_model_dim_ratio

        self.time_embedding = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embedding_dim), 
            nn.SiLU(), 
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim), 
        )
        
        if self.num_classes is not None : 
            self.label_embedding = nn.Embedding(self.num_classes, self.time_embedding_dim)
        
        current_channels = int(self.model_channels * self.channel_mult[0])
        
        self.encoder_blocks = nn.ModuleList([
            nn.ModuleList([get_conv_by_dim(dims, in_channels, current_channels, kernel_size = 3, padding = 1)])
        ])
        self.feature_size = current_channels
        encoder_block_channels = [current_channels]

        attn_coef = 1 

        for level, mult_coef in enumerate(channel_mult):
            for _ in range(self.num_residual_blocks):
                layers = nn.ModuleList([
                    ResidualBlock(
                        channels             = current_channels, 
                        embedding_channels   = self.time_embedding_dim,
                        dropout              = dropout, 
                        out_channels         = int(mult_coef * model_channels),
                        dims                 = dims,
                        use_conv             = True,
                        use_scale_shift_norm = use_scale_shift_norm,   
                    )
                ])
                current_channels = int(mult_coef * model_channels)
                if attn_coef in self.attention_resolutions: 
                    layers.append(
                        AttentionBlock(
                            channels = current_channels, 
                            num_attention_heads = self.num_attention_heads, 
                        )
                    )

                self.encoder_blocks.append(layers)
                self.feature_size += current_channels
                encoder_block_channels.append(current_channels)

            if level + 1 < len(channel_mult):
                output_channel = current_channels 
                self.encoder_blocks.append(nn.ModuleList([
                    ResidualBlock(
                        channels           = current_channels, 
                        embedding_channels = self.time_embedding_dim, 
                        dropout            = dropout,
                        out_channels       = output_channel, 
                        dims               = dims, 
                        use_conv           = True, 
                        use_scale_shift_norm = self.use_scale_shift_norm, 
                        downsampling         = True, 
                    ) if self.residual_block_up_down 
                    else Downsample(
                        channels     = current_channels, 
                        use_conv     = True, 
                        dims         = dims,
                        out_channels = output_channel,
                    )
                ]))
                encoder_block_channels.append(output_channel)
                attn_coef = attn_coef * 2 
                self.feature_size += output_channel
    
        self.mid_block = nn.ModuleList([
            ResidualBlock(
                channels = current_channels, 
                embedding_channels = self.time_embedding_dim, 
                dropout = dropout, 
                use_conv = True, 
                use_scale_shift_norm = self.use_scale_shift_norm, 
                dims = dims, 
                out_channels = current_channels, 
            ), 
            AttentionBlock(
                channels = current_channels,
                num_attention_heads = self.num_attention_heads,  
            ), 
            ResidualBlock(
                channels = current_channels, 
                embedding_channels = self.time_embedding_dim, 
                dropout = dropout, 
                use_conv = True, 
                use_scale_shift_norm = self.use_scale_shift_norm, 
                dims = dims, 
                out_channels = current_channels, 
            ), 
        ])

        self.feature_size += current_channels 

        self.decoder_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            if level + 1 < len(channel_mult):
                attn_coef = attn_coef // 2

            for ind in range(num_residual_blocks + 1): 
                enc_channels = encoder_block_channels.pop()
                layers = nn.ModuleList([ 
                    ResidualBlock(
                        channels = enc_channels + current_channels, 
                        embedding_channels = self.time_embedding_dim, 
                        dropout = dropout, 
                        out_channels = int(model_channels * mult), 
                        dims = dims, 
                        use_conv = True, 
                        use_scale_shift_norm = use_scale_shift_norm
                    )
                ])
                current_channels = int(model_channels * mult)
                if attn_coef in attention_resolutions : 
                    layers.append(
                        AttentionBlock(
                            channels = current_channels, 
                            num_attention_heads = self.num_attention_heads, 
                        )
                    )
                
                if level and ind == num_residual_blocks : 
                    output_channel = current_channels 
                    layers.append(
                        ResidualBlock(
                            channels           = current_channels, 
                            embedding_channels = self.time_embedding_dim, 
                            dropout            = dropout,
                            out_channels       = output_channel, 
                            dims               = dims, 
                            use_conv           = True, 
                            use_scale_shift_norm = self.use_scale_shift_norm, 
                            upsampling         = True, 
                        ) if self.residual_block_up_down 
                        else Upsample(
                            channels     = current_channels, 
                            use_conv     = True, 
                            dims         = dims,
                            out_channels = output_channel,
                        )
                    )

                self.feature_size += current_channels
                self.decoder_blocks.append(layers)

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups = 32, num_channels = current_channels), 
            nn.SiLU(), 
            zero_module(get_conv_by_dim(dims, current_channels, out_channels, kernel_size = 3, padding = 1)), # To reduce the instabilities in the early stage of training 
        )

    def forward(self, inp, timesteps, y = None): 
        if self.num_classes is not None : 
            assert y is not None, "The condition y cannot be None when num_classes is specified"
    
        embedding = self.time_embedding(timestep_embedding(timesteps, self.model_channels))
    
        if self.num_classes is not None : 
            embedding = embedding + self.label_embedding(y)

        output_stack = [] 

        output = inp.clone()
        for module in self.encoder_blocks :
            if isinstance(module, nn.ModuleList):
                for submodule in module:
                    if isinstance(submodule, ResidualBlock):
                        output = submodule(output, embedding)
                    else: 
                        output = submodule(output)
            else : 
                output = module(output)
            
            output_stack.append(output)
        
        for submodule in self.mid_block: 
            if isinstance(submodule, ResidualBlock):
                output = submodule(output, embedding)
            else: 
                output = submodule(output)

        for module in self.decoder_blocks : 
            output = torch.cat([output, output_stack.pop()], dim = 1)
            if isinstance(module, nn.ModuleList):
                for submodule in module:
                    if isinstance(submodule, ResidualBlock):
                        output = submodule(output, embedding)
                    else: 
                        output = submodule(output)
            else : 
                output = module(output)

        output = self.out(output)
        return output 

# if __name__ == '__main__':
#     model = ConditionalDistributionNetwork(
#         image_size = 128, 
#         in_channels = 1, 
#         model_channels = 64, 
#         output_dim     = 10,  
#         num_residual_blocks = 2, 
#         attention_resolutions = [1, 2, 4, 8], 
#         diffusion_num_time_steps = 300, 
#         num_attention_heads = 1, 
#         pool_num_attention_head_channel = 1, 
#         pool = 'attention', 
#     ).to('cuda')

#     inp = torch.randn((8, 1, 128, 128)).to('cuda')
#     timesteps = torch.zeros((8, ), dtype = torch.int32)
#     out = model(inp, timesteps)
#     print(out.shape)
