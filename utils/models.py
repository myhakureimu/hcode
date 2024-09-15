import torch
import torch.nn as nn
from utils.nano_gpt import NanoGPT, GPTConfig, Block, LayerNorm
from transformers import GPT2Model, GPT2Config


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, special_dimension=True):
        super(TransformerModel, self).__init__()
        if special_dimension == True:
            configuration = GPT2Config(
                n_positions=3 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
        else:
            configuration = GPT2Config(
                n_positions=2 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        if special_dimension == True:
            self._read_in = nn.Linear(n_dims + 2, n_embd)
        else:
            self._read_in = nn.Linear(n_dims + 0, n_embd)
        #self._backbone = GPT2Model(configuration)
        
        
        config = GPTConfig(
            input_dim = n_dims,
            block_size = 3 * n_positions,
            #vocab_size = 50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
            n_layer = n_layer,
            n_head = n_head,
            n_embd = n_embd,
            dropout = 0.0,
            bias = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        )
        self._backbone = nn.ModuleDict(dict(
            #wte = nn.Embedding(config.vocab_size, config.n_embd), # I removed word embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            #drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        
        self._read_out = nn.Linear(n_embd, 2)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs
    @staticmethod
    def _combine3(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        '''
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        '''
        zs = torch.stack((torch.cat([torch.ones([bsize, points, 1], device=ys_b.device)*0, # y
                                     torch.ones([bsize, points, 1], device=ys_b.device)*0, # >
                                     xs_b                                                  # x
                                     ], dim=2), # x
                          torch.cat([torch.ones([bsize, points, 1], device=ys_b.device)*0, # y
                                     torch.ones([bsize, points, 1], device=ys_b.device)*1, # >
                                     torch.zeros_like(xs_b, device=ys_b.device)            # x
                                     ], dim=2), # >
                          torch.cat([ys_b.view(bsize, points, 1),                          # y
                                     torch.ones([bsize, points, 1], device=ys_b.device)*0, # >
                                     torch.zeros_like(xs_b, device=ys_b.device)            # x
                                     ], dim=2)
                          ), dim=2)
        zs = zs.view(bsize, 3 * points, dim+2)
        return zs
    
    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        #print('xs', xs.shape)
        #print('ys', ys.shape)
        zs = self._combine(xs, ys)
        #print('zs', zs.shape)
        embeds = self._read_in(zs)
        #print('embeds', embeds.shape)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        #print('output', output.shape)
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]

    def forward2(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine3(xs, ys)
        #print('zs', zs.shape)
        embeds = self._read_in(zs)
        #print('embeds', embeds.shape)
        
        input_shape = embeds.size()
        input_embeds = embeds #self._backbone.wte(embeds)
        position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=embeds.device)
        position_embeds = self._backbone.wpe(position_ids)
        embeds = input_embeds + position_embeds
        
        #output = self._backbone(inputs_embeds=embeds).last_hidden_state
        for layer in self._backbone.h:
            embeds = layer(embeds)#[0]
        output = self._backbone.ln_f(embeds)
        
        
        prediction = self._read_out(output)
        return prediction[:, 1::3, 0][:, inds]


    def forward3(self, xs, ys, inds=None, attention_mask=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine3(xs, ys)
        embeds = self._read_in(zs)
        
        input_shape = embeds.size()
        input_embeds = embeds #self._backbone.wte(embeds)
        position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=embeds.device)
        position_embeds = self._backbone.wpe(position_ids)
        hidden_states = input_embeds + position_embeds
        
        # Transformer blocks
        hidden_states_list = [hidden_states]
        for layer in self._backbone.h:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)#[0]
            hidden_states_list.append(hidden_states)
        # Final layer normalization
        hidden_states = self._backbone.ln_f(hidden_states)
        
        prediction = self._read_out(hidden_states)
        return {'prediction': prediction[:, 1::3, 0][:, inds], 'hiddens': hidden_states_list}
    
        
    def forward4(self, xs, ys, inds=None, attention_mask=None, L=None, injection=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine3(xs, ys)
        embeds = self._read_in(zs)
        
        input_shape = embeds.size()
        input_embeds = embeds #self._backbone.wte(embeds)
        position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=embeds.device)
        position_embeds = self._backbone.wpe(position_ids)
        hidden_states = input_embeds + position_embeds
        #print(hidden_states.shape)
        if (L != None) and (L==0):
            #print(L)
            hidden_states[:,-2,:] = injection
            ### the second last is corresponding to the prediction
        # Transformer blocks
        hidden_states_list = [hidden_states]
        for l_index, layer in enumerate(self._backbone.h):
            hidden_states = layer(hidden_states, attention_mask=attention_mask)#[0]
            if (L != None) and (L==l_index+1):
                #print(L)
                hidden_states[:,-2,:] = injection
            hidden_states_list.append(hidden_states)
        # Final layer normalization
        hidden_states = self._backbone.ln_f(hidden_states)
        
        prediction = self._read_out(hidden_states)
        return {'prediction': prediction[:, 1::3, 0][:, inds], 'hiddens': hidden_states_list}
