# gen imports
from dataclasses import dataclass
import math
import inspect
import tiktoken
import numpy as np
import os
import time 

# torch imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# ROPE is only aviailable in linux
#from .kernel.rotary import apply_rotary_emb
from rms_norm import RMSNorm


# -------------------------------
# Funcs for Diff Transformer
# -------------------------------

def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12 # number of layers
    n_head: int = 6 # number of heads
    n_embd: int = 384 # embedding dimension
    model_parallel_size: int = 1
    decoder_kv_attention_heads: int = None

# -------------------------------
# Diff Transformer
# -------------------------------
class DiffAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.dk = config.n_embd // config.n_head
        self.dv = config.n_embd // config.n_head
        self.d = config.block_size
        self.d_model = config.n_embd
        self.w_q = nn.Linear(config.n_embd, self.dk)
        self.w_k = nn.Linear(config.n_embd, self.dk)
        self.w_v = nn.Linear(config.n_embd, self.dv)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, λ):
        # batch size, sequence length, embedding dimension
        B, N, D = x.size() # 4, 1024, 384

        assert self.w_q.weight.shape == (self.dk, self.d_model,), f"Q weight shape is {self.w_q.weight.shape}"
        assert self.w_k.weight.shape == (self.dk, self.d_model,), f"K weight shape is {self.w_k.weight.shape}"
        assert self.w_v.weight.shape == (self.dv, self.d_model,), f"V weight shape is {self.w_v.weight.shape}"

        # [Q1, Q2] = XW^Q - I want (1024, 2028)
        Q1, Q2 = torch.split(self.w_q(x), split_size_or_sections=32, dim=2)
        # [K1, K2] = XW^K
        K1, K2 = torch.split(self.w_k(x), split_size_or_sections=32, dim=2)
        # [V1, V2] = XW^V
        V = self.w_v(x)

        # assert Q, K, V
        assert Q1.shape == (B, N, self.dk/2), f"Q1 shape is {Q1.shape}"
        assert Q2.shape == (B, N, self.dk/2), f"Q2 shape is {Q2.shape}"
        assert K1.shape == (B, N, self.dk/2), f"K1 shape is {K1.shape}"
        assert K2.shape == (B, N, self.dk/2), f"K2 shape is {K2.shape}"
        assert V.shape == (B, N, self.dv), f"V shape is {V.shape}"

        # Qi, Ki: [b, n, d]; V; [b, n, 2d]
        s = 1 / math.sqrt(self.d)

        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s

        return (self.softmax(A1) - (λ*self.softmax(A2))) @ V
    
class MultiHead(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.d = config.block_size
        self.d_model = config.n_embd
        self.dv = config.n_embd // config.n_head
        self.w_o = nn.Linear(config.n_head*self.dv, self.d_model)
        self.λ = 0.8 - 0.6 * math.exp(-0.3 * config.n_layer)
        self.LN = RMSNorm(self.dv, eps=1e-5, elementwise_affine=True)
        self.GroupNorm = nn.GroupNorm(1, self.d*2, eps=1e-5)
        self.heads = nn.ModuleList([DiffAttn(config) for _ in range(config.n_head)])        
    
    def forward(self, x):
        O = [self.LN(head(x, self.λ))*(1-self.λ) for head in self.heads]
        return self.w_o(torch.concat(O, dim=2) )

# -------------------------------
# FC Layer
# -------------------------------
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        # histoical quirk bc paper used this, helps with the dead relu problem
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# -------------------------------
# Transformer Block
# -------------------------------
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHead(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # this is where they communicate - find what other words are important
        x = x + self.attn(self.ln_1(x))
        # no info being exchanged here, think individual about what they communicated about
        x = x + self.mlp(self.ln_2(x))
        return x

# -------------------------------
# GPT Model
# -------------------------------
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # token embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embeddings - we want it to know where it is
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # hidden - the transformer block
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # layer norm this is from gpt paper
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # classifier head from the embeding to vocab
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)      

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
# -------------------------------

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -------------------------------
# Data Loader
# -------------------------------
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    
class DataLoaderSMALL:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        if master_process:
            print(f"loaded {len(self.tokens)} tokens")

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y
# -------------------------------
# Trainning loop
# python train_gpt2.py or torchrun --standalone --nproc_per_node=8 train_gpt2.py
# -------------------------------

num_return_sequences = 5
max_length = 30

# -------------------------------
# DDP (Distributed Data Parallel)
# -------------------------------
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding('gpt2')


total_batch_size = 524288
B = 2 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# -------------------------------
# Options
# -------------------------------
dataset = 'small' # small runs the input.txt, if u want fineweb change
transfer_learning = False # if true loads weights from hugging weights (func not imp)
compile_model = False 

# -------------------------------
# data loader set up
# -------------------------------
if dataset == 'small':
    train_loader = DataLoaderSMALL(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
else:
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')


torch.set_float32_matmul_precision('high')

# -------------------------------
# init model
# -------------------------------
if transfer_learning:
    model = GPT.from_pretrained('gpt2')
else:
    model = GPT(GPTConfig(vocab_size=50304))
model.to(device)


if compile_model:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


# -------------------------------
# Set up for learning rate 
# -------------------------------
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)


# -------------------------------
# Training Loop
# -------------------------------
for i in range(max_steps):
    t0 = time.time()

    if dataset != 'small':
        # for the input.txt we dont have a validation set
        if i % 100 == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)

                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")

    # train
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss because we are accumulating gradients
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # pytorch has a lr scheduler
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    print(f"step {i}, loss: {loss_accum.item():.6f}, norm: {norm:.4f}, lr: {lr:.4e}")

if ddp:
    destroy_process_group()