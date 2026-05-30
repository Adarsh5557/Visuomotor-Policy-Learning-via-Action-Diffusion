import os, math, collections, urllib.request, zipfile, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

class PushTStateDataset(Dataset):
    def __init__(self, dataset_path, obs_horizon=2, pred_horizon=16):
        import zarr
        root = zarr.open(dataset_path, mode="r")
        state_data  = root["data"]["state"][:]   
        action_data = root["data"]["action"][:]  
        episode_ends = root["meta"]["episode_ends"][:]
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        
        self.obs_min, self.obs_max = state_data.min(axis=0), state_data.max(axis=0)
        self.act_min, self.act_max = action_data.min(axis=0), action_data.max(axis=0)
        
        self.state_norm  = 2.0 * (state_data  - self.obs_min) / (self.obs_max - self.obs_min + 1e-8) - 1.0
        self.action_norm = 2.0 * (action_data - self.act_min) / (self.act_max - self.act_min + 1e-8) - 1.0

        indices = []
        start = 0
        for end in episode_ends:
            for idx in range(start + obs_horizon - 1, end - pred_horizon + 1):
                indices.append(idx)
            start = end
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, k):
        idx = self.indices[k]
        obs_seq = self.state_norm[idx - self.obs_horizon + 1 : idx + 1]
        act_seq = self.action_norm[idx : idx + self.pred_horizon]
        return {"obs": torch.from_numpy(obs_seq).float(), "action": torch.from_numpy(act_seq).float()}

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        emb  = math.log(10000) / (half - 1)
        emb  = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb  = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class Downsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1, self.norm2 = nn.GroupNorm(8, out_ch), nn.GroupNorm(8, out_ch)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, cond):
        h = F.mish(self.norm1(self.conv1(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.mish(self.norm2(self.conv2(h)))
        return h + self.res(x)

class DiffusionUNet(nn.Module):
    def __init__(self, act_dim=2, obs_dim=5, obs_horizon=2):
        super().__init__()
        cond_dim = 256
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim * obs_horizon, 256), nn.Mish(), nn.Linear(256, cond_dim))
        self.time_emb = SinusoidalPosEmb(cond_dim)
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.Mish(), nn.Linear(cond_dim, cond_dim))
        self.d1 = ResBlock(act_dim, 256, cond_dim); self.down1 = Downsample1D(256)
        self.d2 = ResBlock(256, 512, cond_dim);     self.down2 = Downsample1D(512)
        self.mid = ResBlock(512, 512, cond_dim)
        self.up1 = Upsample1D(512);                 self.u1 = ResBlock(1024, 256, cond_dim)
        self.up2 = Upsample1D(256);                 self.u2 = ResBlock(512, 256, cond_dim)
        self.final = nn.Conv1d(256, act_dim, 1)
    def forward(self, x, t, obs):
        x = x.transpose(1, 2)
        cond = self.obs_enc(obs.flatten(1)) + self.time_mlp(self.time_emb(t))
        h1 = self.d1(x, cond); h2 = self.d2(self.down1(h1), cond); h3 = self.mid(self.down2(h2), cond)
        u = self.u1(torch.cat([self.up1(h3), h2], dim=1), cond)
        u = self.u2(torch.cat([self.up2(u),  h1], dim=1), cond)
        return self.final(u).transpose(1, 2)

class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1.0 - self.decay) * p.detach()
    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

DATASET_PATH, OBS_HORIZON, PRED_HORIZON = "data/pusht/pusht_cchi_v7_replay.zarr", 2, 16

if not os.path.exists(DATASET_PATH):
    zip_path = "pusht.zip"

    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(
            "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip",
            zip_path
        )

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/")

    if os.path.exists(zip_path):
        os.remove(zip_path)

dataset = PushTStateDataset(DATASET_PATH, OBS_HORIZON, PRED_HORIZON)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiffusionUNet(2, 5, OBS_HORIZON).to(device)
ema = EMA(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=True
)

NUM_EPOCHS = 1000

lr_scheduler = get_scheduler(
    "cosine",
    optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * NUM_EPOCHS
)

checkpoint_path = "checkpoint.pth"
best_model_path = "best_model.pth"

start_epoch = 0
best_loss = float("inf")

history = {"loss": [], "success": [], "reward": []}

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    ema.shadow = ckpt['ema_shadow']
    start_epoch = ckpt['epoch'] + 1
    history = ckpt.get('history', history)
    print(f"Resuming from Epoch {start_epoch}")

print(f"Starting training on {device}...")

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    epoch_losses = []

    with tqdm(dataloader, desc=f"Epoch {epoch}") as t_epoch:
        for batch in t_epoch:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)

            noise = torch.randn_like(action)
            t = torch.randint(0, 100, (action.shape[0],), device=device).long()

            noisy = noise_scheduler.add_noise(action, noise, t)

            pred = model(noisy, t, obs)
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.update(model)

            epoch_losses.append(loss.item())
            t_epoch.set_postfix(loss=np.mean(epoch_losses))

    epoch_loss = np.mean(epoch_losses)
    history["loss"].append(epoch_loss)

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "ema_shadow": ema.shadow,
        "history": history,
        "obs_min": dataset.obs_min, 
        "obs_max": dataset.obs_max,
        "act_min": dataset.act_min,
        "act_max": dataset.act_max
    }, checkpoint_path)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save({
            "epoch": epoch,
            "model": ema.shadow,
            "loss": best_loss,
            "obs_min": dataset.obs_min,
            "obs_max": dataset.obs_max, 
            "act_min": dataset.act_min,
            "act_max": dataset.act_max
        }, best_model_path)
        print(f"Best model saved at epoch {epoch} | loss: {best_loss:.6f}")

print("Training finished.")

if os.path.exists(checkpoint_path):
    shutil.copy(checkpoint_path, "checkpoint_backup.pth")
    print("Backup saved.")
else:
    print("No checkpoint found.")
