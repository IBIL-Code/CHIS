import torch
import torch.nn.functional as F
from .guidance import latent_patchmatch


class CHISController:
    def __init__(
        self,
        scheduler,
        sigma,
        nucleus_bank,
        bg_bank,
        soft_mask,
        guide_latents,
        inject_step=40,
        seed=42,
        fg_low_lock=0.95,
        fg_high_lock=0.6,
        bg_lock=0.15,
    ):
        self.inject_step = inject_step
        self.device = guide_latents.device
        self.guide_latents = guide_latents.to(self.device)
        self.scheduler = scheduler
        self.seed = seed
        self.soft_mask = soft_mask.to(self.device).float()
        self.sigma = sigma
        self.nuc_bank = nucleus_bank
        self.bg_bank = bg_bank
        self.fg_low_lock = fg_low_lock
        self.fg_high_lock = fg_high_lock
        self.bg_lock = bg_lock

    def get_dynamic_noisy_guide(self, timestep):
        seed_step = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        generator = torch.Generator(device=self.device).manual_seed(
            self.seed + seed_step
        )
        noise = torch.randn(
            self.guide_latents.shape,
            device=self.guide_latents.device,
            dtype=self.guide_latents.dtype,
            generator=generator,
        )
        if not torch.is_tensor(timestep):
            t = torch.tensor([timestep], device=self.device, dtype=torch.long)
        else:
            t = timestep.reshape(-1)
        noisy_guide = self.scheduler.add_noise(self.guide_latents, noise, t)
        current_noise = noise
        return (noisy_guide, current_noise)

    def haar_wavelet_decompose(self, x):
        low = F.avg_pool2d(x, kernel_size=2, stride=2)
        low_upsampled = F.interpolate(low, size=x.shape[-2:], mode="nearest")
        high = x - low_upsampled
        return (low_upsampled, high)

    def apply_structure_lock(self, current_latents, guide_noisy):
        current_low, current_high = self.haar_wavelet_decompose(current_latents)
        guide_low, guide_high = self.haar_wavelet_decompose(guide_noisy)
        guide_mean = guide_low.mean(dim=(2, 3), keepdim=True)
        guide_std = guide_low.std(dim=(2, 3), keepdim=True) + 1e-06
        current_mean = current_low.mean(dim=(2, 3), keepdim=True)
        current_std = current_low.std(dim=(2, 3), keepdim=True) + 1e-06
        guide_low_aligned = (
            guide_low - guide_mean
        ) / guide_std * current_std + current_mean
        m = self.soft_mask
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(0)
        fg_low_lock = self.fg_low_lock
        fg_high_lock = self.fg_high_lock
        bg_low_lock = self.bg_lock
        bg_high_lock = self.bg_lock
        low_freq_alpha = m * fg_low_lock + (1 - m) * bg_low_lock
        high_freq_alpha = m * fg_high_lock + (1 - m) * bg_high_lock
        mixed_low = (
            low_freq_alpha * guide_low_aligned + (1 - low_freq_alpha) * current_low
        )
        mixed_high = high_freq_alpha * guide_high + (1 - high_freq_alpha) * current_high
        target_nucleus = mixed_low + mixed_high
        ref_nuc_mu = self.nuc_bank.mean(dim=0).reshape(1, -1, 1, 1)
        ref_nuc_std = self.nuc_bank.std(dim=0).reshape(1, -1, 1, 1) + 1e-06
        ref_bg_mu = self.bg_bank.mean(dim=0).reshape(1, -1, 1, 1)
        ref_bg_std = self.bg_bank.std(dim=0).reshape(1, -1, 1, 1) + 1e-06
        ref_bg_std = (ref_bg_std - ref_bg_std.min()) / (
            ref_bg_std.max() - ref_bg_std.min() + 1e-06
        )
        x = target_nucleus
        eps = 1e-06
        nuc_weight = m
        nuc_sum = nuc_weight.sum() + eps
        curr_nuc_mu = (x * nuc_weight).sum(dim=(2, 3), keepdim=True) / nuc_sum
        curr_nuc_std = (
            ((x - curr_nuc_mu) ** 2 * nuc_weight).sum(dim=(2, 3), keepdim=True)
            / nuc_sum
        ).sqrt() + eps
        bg_weight = 1 - m
        bg_sum = bg_weight.sum() + eps
        curr_bg_mu = (x * bg_weight).sum(dim=(2, 3), keepdim=True) / bg_sum
        curr_bg_std = (
            ((x - curr_bg_mu) ** 2 * bg_weight).sum(dim=(2, 3), keepdim=True) / bg_sum
        ).sqrt() + eps
        aligned_nuc = (x - curr_nuc_mu) / curr_nuc_std * ref_nuc_std + ref_nuc_mu
        damping_factor = 0.6
        target_bg_std = torch.max(curr_bg_std * damping_factor, ref_bg_std)
        aligned_bg = (x - curr_bg_mu) / (
            curr_bg_std + 1e-08
        ) * target_bg_std + ref_bg_mu
        target_nucleus_fixed = m * aligned_nuc + (1 - m) * aligned_bg
        return target_nucleus_fixed

    def apply_texture_patchmatch(self, latents_struct):
        latents_f32 = latents_struct.float()
        bank_f32 = self.nuc_bank.float()
        latents_fixed = latent_patchmatch(
            latents_struct=latents_f32, texture_bank=bank_f32, soft_mask=self.soft_mask
        )
        return latents_fixed.to(latents_struct.dtype)

    def __call__(self, step_idx, timestep, latents):
        if step_idx > self.inject_step:
            return
        with torch.no_grad():
            guide_noisy, current_noise = self.get_dynamic_noisy_guide(timestep)
            m = self.soft_mask
            latents_struct = self.apply_structure_lock(latents, guide_noisy)
            if step_idx < 15:
                latents_final = latents_struct
                latents[:] = latents_final
            else:
                latents_patched_clean = self.apply_texture_patchmatch(
                    self.guide_latents
                )
                t_tensor = (
                    timestep.reshape(-1)
                    if torch.is_tensor(timestep)
                    else torch.tensor([timestep], device=self.device, dtype=torch.long)
                )
                latents_patched_noisy = self.scheduler.add_noise(
                    latents_patched_clean, current_noise, t_tensor
                )
                alpha = 0.6
                latents_fused = (
                    alpha * latents_patched_noisy + (1 - alpha) * latents_struct
                )
                latents_final = m * latents_fused + (1 - m) * latents_struct
                latents[:] = latents_final
