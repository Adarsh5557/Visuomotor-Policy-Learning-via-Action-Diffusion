import os
import torch
import numpy as np
import gymnasium as gym
from diffusion_policy.env.pusht import pusht_env
import collections
import pymunk
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math

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
        h1 = self.d1(x, cond)
        h2 = self.d2(self.down1(h1), cond)
        h3 = self.mid(self.down2(h2), cond)
        u = self.u1(torch.cat([self.up1(h3), h2], dim=1), cond)
        u = self.u2(torch.cat([self.up2(u), h1], dim=1), cond)
        return self.final(u).transpose(1, 2)


class DummyHandler:
    def __init__(self): self.post_solve = None

def bulletproof_patch(self, a, b=None): return DummyHandler()

pymunk.Space.add_collision_handler = bulletproof_patch
if hasattr(pymunk.Space, 'add_default_collision_handler'):
    pymunk.Space.add_default_collision_handler = bulletproof_patch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_HORIZON, PRED_HORIZON = 2, 16

checkpoint_path = "best_model.pth"  #use best model

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"{checkpoint_path} not found!")

ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

model = DiffusionUNet(2, 5, OBS_HORIZON).to(device)

model.load_state_dict(ckpt["model"])
model.eval()

obs_min = torch.tensor(ckpt['obs_min'], device=device)
obs_max = torch.tensor(ckpt['obs_max'], device=device)
act_min = torch.tensor(ckpt['act_min'], device=device)
act_max = torch.tensor(ckpt['act_max'], device=device)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2"
)


env = pusht_env.PushTEnv()

num_eval_episodes = 50
max_steps = 300
action_execution_horizon = 8

eval_rewards = []
eval_successes = []

print(f"Starting Evaluation on {device}...")

with torch.no_grad():
    for ep in tqdm(range(num_eval_episodes)):
        obs = env.reset()
        obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)

        max_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            obs_seq = np.stack(obs_deque)

            obs_tensor = 2.0 * (torch.from_numpy(obs_seq).to(device) - obs_min) / (obs_max - obs_min + 1e-8) - 1.0
            obs_tensor = obs_tensor.unsqueeze(0).float()

            noisy_action = torch.randn((1, PRED_HORIZON, 2), device=device)

            noise_scheduler.set_timesteps(20)

            for t in noise_scheduler.timesteps:
                t_batch = torch.tensor([t], device=device).long()
                noise_pred = model(noisy_action, t_batch, obs_tensor)
                noisy_action = noise_scheduler.step(noise_pred, t, noisy_action).prev_sample

            norm_actions = noisy_action.squeeze(0)

            pred_actions = (norm_actions + 1.0) / 2.0 * (act_max - act_min) + act_min
            pred_actions = pred_actions.cpu().numpy()

            pred_actions = np.clip(pred_actions, act_min.cpu().numpy(), act_max.cpu().numpy())

            for i in range(action_execution_horizon):
                obs, reward, done, info = env.step(pred_actions[i])
                obs_deque.append(obs)
                max_reward = max(max_reward, reward)
                step += 1
                if done:
                    break

        eval_rewards.append(max_reward)
        eval_successes.append(1 if max_reward > 0.9 else 0)

env.close()

import imageio.v2 as imageio

print("\nGenerating high-quality demo video...")

env = pusht_env.PushTEnv()
obs = env.reset()
obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)

frames = []
step = 0
done = False

with torch.no_grad():
    while not done and step < max_steps:
        obs_seq = np.stack(obs_deque)

        obs_tensor = 2.0 * (torch.from_numpy(obs_seq).to(device) - obs_min) / (obs_max - obs_min + 1e-8) - 1.0
        obs_tensor = obs_tensor.unsqueeze(0).float()

        noisy_action = torch.randn((1, PRED_HORIZON, 2), device=device)

        noise_scheduler.set_timesteps(20)

        for t in noise_scheduler.timesteps:
            t_batch = torch.tensor([t], device=device).long()
            noise_pred = model(noisy_action, t_batch, obs_tensor)
            noisy_action = noise_scheduler.step(noise_pred, t, noisy_action).prev_sample

        norm_actions = noisy_action.squeeze(0)
        pred_actions = (norm_actions + 1.0) / 2.0 * (act_max - act_min) + act_min
        pred_actions = pred_actions.cpu().numpy()
        pred_actions = np.clip(pred_actions, act_min.cpu().numpy(), act_max.cpu().numpy())

        for i in range(action_execution_horizon):
            obs, reward, done, info = env.step(pred_actions[i])
            obs_deque.append(obs)
            step += 1

            try:
                frame = env.render(mode='rgb_array')
            except TypeError:
                frame = env.render()

            if frame is not None:
                frame = np.array(frame)

                if frame.dtype != np.uint8:
                    frame = (255 * frame).astype(np.uint8)

            
                frame = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)

                frames.append(frame)

            if done:
                break

env.close()

video_path = "pusht_demo_hd.mp4"

with imageio.get_writer(
    video_path,
    format="FFMPEG",
    mode="I",
    fps=20,
    codec="libx264",
    quality=8,
    pixelformat="yuv420p"
) as writer:
    for frame in frames:
        writer.append_data(frame)

print(f"High-quality video saved as {video_path}")

success_rate = np.mean(eval_successes)

print("\nEvaluation Summary:")
print(f"Average Max Reward: {np.mean(eval_rewards):.3f}")
print(f"Success Rate: {success_rate * 100:.1f}%")
