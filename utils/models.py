import torch
import torch.nn as nn
from utils.nano_gpt import NanoGPT
from transformers import GPT2Model, GPT2Config

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

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions = n_positions,
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
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)

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
        zs = torch.stack((label2onehot(xs_b, self.n_dims), # x
                          label2onehot(ys_b, self.n_dims), # y
                          ), dim=2)

        zs = zs.view(bsize, 2 * points, self.n_dims).float()
        #print(zs.shape)
        return zs

    def forward2(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        #print('xs', xs.shape)
        #print('ys', ys.shape)
        zs = self._combine2(xs, ys)
        #print(zs.shape)
        #print('zs', zs.shape)
        embeds = self._read_in(zs)
        #print('embeds', embeds.shape)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        #print('output', output.shape)
        prediction = self._read_out(output)
        #print(prediction[:, ::2, :][:, inds].shape)
        #print(prediction[:, ::2, :][:, inds] == prediction[:, ::2, :])
        return prediction[:, ::2, :][:, inds]