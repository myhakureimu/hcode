"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


from einops import repeat, rearrange, einsum
class SigmaReparam(nn.Module):
    """ "
    https://arxiv.org/pdf/2303.06296.pdf Appendix C
    """

    def __init__(self, d_in, d_out, bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(d_out), requires_grad=True) if bias else None
        u = torch.randn(d_out)
        self.u = nn.Parameter(u / u.norm(dim=0), requires_grad=False)
        v = torch.randn(d_in)
        self.v = nn.Parameter(v / v.norm(dim=0), requires_grad=False)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # same as nn.Linear
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                u = (self.W @ self.v).float()
                self.u.data = u / u.norm(dim=0)
                v = (self.W.T @ self.u).float()
                self.v.data = v / v.norm(dim=0)
        sigma = einsum(self.u, self.W, self.v, "d, d c , c->")
        W_hat = self.gamma / sigma * self.W
        out = F.linear(x, W_hat, self.b)
        return out
    
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads, but in a batch
        if config.SigmaRe in [0]:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        if config.SigmaRe in [1,2]:
            #print(1)
            self.c_attn = SigmaReparam(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        if config.SigmaRe in [0,1]:
            #print(2)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        if config.SigmaRe in [2]:
            self.c_proj = SigmaReparam(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = True#hasattr(torch.nn.functional, 'scaled_dot_product_attention') #TODO
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            #print('FLASH')
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            if self.config.NormAtt:
                pass
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                #print(att[0,0])
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
            
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        #x = x + self.attn(self.ln_1(x))
        #x = x + self.mlp(self.ln_2(x))
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x

@dataclass
class GPTConfig:
    input_dim: int = 3
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    SigmaRe: int = 0
    NormAtt: int = 0
    FirstLayerNorm: int = 0

def label2onehot(input_tensor, dim):
    """
    Converts a binary tensor of shape (batch_size, points, 1) to a one-hot tensor of shape (batch_size, points, 2),
    mapping 0 to [1, 0] and 1 to [0, 1].
    """
    # Remove the last dimension
    input_tensor = input_tensor.squeeze(-1)  # Shape: (batch_size, points)

    # Ensure the tensor is of integer type
    input_tensor = input_tensor.long()

    # One-hot encode
    one_hot = torch.nn.functional.one_hot(input_tensor, num_classes=dim)  # Shape: (batch_size, points, 2)

    return one_hot

class NanoGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        #self._read_in = nn.Linear(config.input_dim + 2, config.n_embd)
        self._read_in = nn.Linear(config.input_dim, config.n_embd)
        #self._read_in = nn.Embedding(config.input_dim, config.n_embd)#, max_norm=True)
        if config.FirstLayerNorm:
            self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.transformer = nn.ModuleDict(dict(
            #wte = nn.Embedding(config.vocab_size, config.n_embd), # I removed word embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            #drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        #self._read_out = nn.Linear(config.n_embd, 2)
        self._read_out = nn.Linear(config.n_embd, config.input_dim, bias=False)
        #self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        
        #self.transformer.wte.weight = self.lm_head.weight # I removed word embedding # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                #torch.nn.init.normal_(p, mean=0.0, std=0.04/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # def forward(self, idx, targets=None):
    #     device = idx.device
    #     b, t = idx.size()
    #     assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
    #     pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

    #     # forward the GPT model itself
    #     tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
    #     pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
    #     x = self.transformer.drop(tok_emb + pos_emb)
    #     for block in self.transformer.h:
    #         x = block(x)
    #     x = self.transformer.ln_f(x)

    #     if targets is not None:
    #         # if we are given some desired targets also calculate the loss
    #         logits = self.lm_head(x)
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    #     else:
    #         # inference-time mini-optimization: only forward the lm_head on the very last position
    #         logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
    #         loss = None

    #     return logits, loss

    #@staticmethod
    def _combine2(self, xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points = xs_b.shape
        '''
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        '''
        # zs = torch.stack((torch.cat([xs_b,                                               # x
        #                              torch.zeros([bsize, points, 1], device=ys_b.device) # y
        #                              ], dim=2), # x
        #                   torch.cat([torch.zeros_like(xs_b, device=ys_b.device),         # x
        #                              ys_b.view(bsize, points, 1)                         # y
        #                              ], dim=2)
        #                   ), dim=2)
        #print('forward')
        #print(xs_b.shape)
        #print(ys_b.shape)
        zs = torch.stack((label2onehot(xs_b, self.config.input_dim), # x
                          label2onehot(ys_b, self.config.input_dim), # y
                          ), dim=2)

        zs = zs.view(bsize, 2 * points, self.config.input_dim).float()
        #print(zs.shape)
        return zs
    
    def forward2(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
                
        zs = self._combine2(xs, ys)
        #print(zs.shape)
        input_embeds = self._read_in(zs)
        
        pos = torch.arange(0, self.config.block_size, dtype=torch.long, device=xs.device).unsqueeze(0) # shape (1, t)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        x = input_embeds + pos_emb[:,:zs.shape[1]]
        #output = self._backbone(inputs_embeds=embeds).last_hidden_state
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        prediction = self._read_out(x)
        return prediction[:, 0::2, :]#[:, inds]

    # def forward3(self, xs, ys, inds=None, attention_mask=None):
    #     #if inds is None:
    #     #    inds = torch.arange(ys.shape[1])
    #     #else:
    #     #    inds = torch.tensor(inds)
    #     #    if max(inds) >= ys.shape[1] or min(inds) < 0:
    #     #        raise ValueError("inds contain indices where xs and ys are not defined")
        
    #     input_embeds = self._read_in(xs)
        
    #     pos = torch.arange(0, self.config.block_size, dtype=torch.long, device=xs.device).unsqueeze(0) # shape (1, t)
    #     pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
    #     x = input_embeds + pos_emb[:, :xs.shape[1], :]
    #     if self.config.FirstLayerNorm:
    #          x = self.ln_f(x)
        
    #     # Transformer blocks
    #     hidden_states_list = [x]
    #     for layer in self.transformer.h:
    #         x = layer(x)
    #         hidden_states_list.append(x)
    #     # Final layer normalization
    #     #x = self.transformer.ln_f(x)
        
    #     prediction = self._read_out(x)
    #     return {'prediction': prediction, 'hiddens': hidden_states_list}
    
    # def forward4(self, xs, ys, insert, inds=None, attention_mask=None, L=None, injection=None):
    #     b1, b2 = insert
    #     #if inds is None:
    #     #    inds = torch.arange(ys.shape[1])
    #     #else:
    #     #    inds = torch.tensor(inds)
    #     #    if max(inds) >= ys.shape[1] or min(inds) < 0:
    #     #        raise ValueError("inds contain indices where xs and ys are not defined")
        
    #     input_embeds = self._read_in(xs)
        
    #     pos = torch.arange(0, self.config.block_size, dtype=torch.long, device=xs.device).unsqueeze(0) # shape (1, t)
    #     pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
    #     #print('pos_emb ', pos_emb.shape)
        
    #     x = input_embeds + pos_emb[:, :xs.shape[1], :]
    #     if self.config.FirstLayerNorm:
    #          x = self.ln_f(x)
             
    #     #print('x .shape ',x.shape)
    #     #print(hidden_states.shape)
    #     #print('L=',L)
    #     if (L != None) and (L==0):
    #         #print(L)
    #         x[[b1, b2]] = injection
    #         ### the second last is corresponding to the prediction
    #     # Transformer blocks
    #     hidden_states_list = [x]
    #     for l_index, layer in enumerate(self.transformer.h):
    #         #print(l_index)
    #         #print('x .shape ',x.shape)  
    #         x = layer(x)#, attention_mask=attention_mask)
    #         if (L != None) and (L==l_index+1):
    #             #print(L)
    #             x[[b1, b2]] = injection
    #         #print('x .shape ',x.shape)
    #         hidden_states_list.append(x)
    #     # Final layer normalization
    #     #x = self.transformer.ln_f(x)
        
    #     prediction = self._read_out(x)
    #     return {'prediction': prediction, 'hiddens': hidden_states_list}

