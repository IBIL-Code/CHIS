import cv2
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter


def h_channel_deconvolution(rgb_image):
    if isinstance(rgb_image, Image.Image):
        rgb_image = np.array(rgb_image)
    if rgb_image.max() > 1.0:
        rgb_norm = rgb_image.astype(np.float32) / 255.0
    else:
        rgb_norm = rgb_image.astype(np.float32)
    rgb_norm = np.clip(rgb_norm, 1e-06, 1.0)
    od = -np.log10(rgb_norm + 1e-06)
    he_matrix = np.array(
        [[0.644, 0.717, 0.267], [0.093, 0.954, 0.283]], dtype=np.float32
    )
    he_inv = np.linalg.pinv(he_matrix.T)
    od_flat = od.reshape(-1, 3).T
    he_flat = he_inv @ od_flat
    H_flat = he_flat[0, :]
    H_channel = H_flat.reshape(od.shape[0], od.shape[1])
    p1, p99 = np.percentile(H_channel, (1, 99))
    if p99 - p1 > 1e-06:
        H_channel = np.clip((H_channel - p1) / (p99 - p1), 0, 1)
    else:
        H_channel = H_channel - H_channel.min()
        if H_channel.max() > 0:
            H_channel = H_channel / H_channel.max()
    return H_channel


def otsu_threshold_segmentation(image, fill_contours=True, min_area=50):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8 and image.dtype != np.uint16:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    threshold_value, binary_mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if fill_contours:
        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled_mask = np.zeros_like(binary_mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(filled_mask, [contour], -1, 255, thickness=-1)
        binary_mask = filled_mask
    return (binary_mask, int(threshold_value))


def segment_reference_mask(rgb_image, fill_contours=True, min_area=50):
    h_channel = h_channel_deconvolution(rgb_image)
    binary_mask, _ = otsu_threshold_segmentation(
        h_channel, fill_contours=fill_contours, min_area=min_area
    )
    reference_mask = (binary_mask / 255.0).astype(np.float32)
    return reference_mask


def process_masks(reference_mask, target_mask):
    reference_mask = torch.from_numpy(reference_mask).float()
    if reference_mask.dim() == 2:
        reference_mask = reference_mask.unsqueeze(0).unsqueeze(0)
    elif reference_mask.dim() == 3:
        reference_mask = reference_mask.unsqueeze(1)
    target_mask = torch.from_numpy(target_mask).float()
    if target_mask.dim() == 2:
        target_mask = target_mask.unsqueeze(0).unsqueeze(0)
    elif target_mask.dim() == 3:
        target_mask = target_mask.unsqueeze(1)
    dr_mask = F.max_pool2d(reference_mask, kernel_size=8, stride=8)
    dr_mask = (dr_mask > 0.5).float()
    dt_mask = F.avg_pool2d(target_mask, kernel_size=8, stride=8)
    hard_mask = (dt_mask > 0.5).float()
    hard_mask_np = hard_mask.squeeze().cpu().numpy()
    if hard_mask_np.sum() > 0:
        dist = distance_transform_edt(hard_mask_np)
        R = dist[hard_mask_np > 0].mean()
        sigma = R / 2.5
    else:
        sigma = 1.0
    dr_mask = dr_mask.to(torch.float16)
    dt_mask = dt_mask.to(torch.float16)
    return (dr_mask.half(), dt_mask.half(), sigma)


def encode_with_vae(vae, init_guide):
    if isinstance(init_guide, np.ndarray):
        image_tensor = torch.from_numpy(init_guide).float()
    else:
        image_tensor = init_guide.float()
    image_tensor = image_tensor / 127.5 - 1.0
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        latent_dist = vae.encode(image_tensor).latent_dist
        latents = latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents.half()


def generate_structure_guide(
    reference_image,
    reference_mask,
    target_mask,
    size=256,
    n_bins=100,
    background_blur_sigma=3,
    edge_feather_sigma=2,
    noise_scale=0.01,
    texture_smooth_sigma=1.2,
):
    ref_bg_mask = reference_mask == 0
    ref_bg_pixels = reference_image[ref_bg_mask]
    h, w = (size, size)
    if len(ref_bg_pixels) > 0:
        bg_mean = ref_bg_pixels.mean(axis=0)
    else:
        bg_mean = reference_image.reshape(-1, reference_image.shape[-1]).mean(axis=0)
    synthesized_bg = np.tile(bg_mean, (h, w, 1))
    clean_background = gaussian_filter(
        synthesized_bg.astype(float),
        sigma=(background_blur_sigma, background_blur_sigma, 0),
    )
    noise = np.random.randn(*clean_background.shape) * noise_scale * 255
    clean_background = np.clip(clean_background + noise, 0, 255).astype(
        reference_image.dtype
    )
    D_ref = distance_transform_edt(reference_mask)
    D_ref_max = D_ref.max()
    D_ref_norm = D_ref / D_ref_max if D_ref_max > 0 else D_ref
    target_mask_resized = cv2.resize(
        target_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
    )
    D_target = distance_transform_edt(target_mask_resized)
    D_target_max = D_target.max()
    D_target_norm = D_target / D_target_max if D_target_max > 0 else D_target
    bin_indices_ref = np.clip((D_ref_norm * (n_bins - 1)).astype(int), 0, n_bins - 1)
    texture_lut = [[] for _ in range(n_bins)]
    mask = reference_mask > 0
    bins = bin_indices_ref[mask]
    pixels = reference_image[mask]
    for b, p in zip(bins, pixels):
        texture_lut[int(b)].append(p)
    texture_lut = [np.asarray(v) if v else v for v in texture_lut]
    synthesized_cell = clean_background.copy()
    for y in range(size):
        for x in range(size):
            if target_mask_resized[y, x] > 0:
                d = D_target_norm[y, x]
                bin_idx = int(np.clip(d * (n_bins - 1), 0, n_bins - 1))
                if len(texture_lut[bin_idx]) > 0:
                    random_idx = np.random.randint(len(texture_lut[bin_idx]))
                    synthesized_cell[y, x] = texture_lut[bin_idx][random_idx]
                else:
                    found = False
                    for offset in range(1, n_bins):
                        for sign in [1, -1]:
                            nearby_bin = bin_idx + sign * offset
                            if (
                                0 <= nearby_bin < n_bins
                                and len(texture_lut[nearby_bin]) > 0
                            ):
                                random_idx = np.random.randint(
                                    len(texture_lut[nearby_bin])
                                )
                                synthesized_cell[y, x] = texture_lut[nearby_bin][
                                    random_idx
                                ]
                                found = True
                                break
                        if found:
                            break
    synthesized_cell_blurred = gaussian_filter(
        synthesized_cell, sigma=(texture_smooth_sigma, texture_smooth_sigma, 0)
    )
    mask_3ch = np.stack([target_mask_resized] * reference_image.shape[2], axis=-1)
    synthesized_cell = np.where(
        mask_3ch > 0, synthesized_cell_blurred, synthesized_cell
    )
    boundary_thickness = 3
    boundary_region = (D_target > 0) & (D_target <= boundary_thickness)
    if len(reference_image.shape) == 3:
        for c in range(reference_image.shape[2]):
            channel = synthesized_cell[:, :, c]
            boundary_pixels = channel[boundary_region]
            if len(boundary_pixels) > 0:
                channel[boundary_region] = np.clip(boundary_pixels * 0.85, 0, 255)
                synthesized_cell[:, :, c] = channel
    alpha_mask = gaussian_filter(
        target_mask_resized.astype(float), sigma=edge_feather_sigma
    )
    alpha_mask = np.clip(alpha_mask, 0, 1)
    if len(reference_image.shape) == 3:
        alpha_mask_3ch = np.stack([alpha_mask] * reference_image.shape[2], axis=-1)
    else:
        alpha_mask_3ch = alpha_mask
    result = (
        synthesized_cell * alpha_mask_3ch + clean_background * (1 - alpha_mask_3ch)
    ).astype(reference_image.dtype)
    nucleus_mask = target_mask_resized > 0.5
    if nucleus_mask.sum() > 0:
        for c in range(result.shape[2]):
            channel = result[:, :, c].astype(float)
            nucleus_pixels = channel[nucleus_mask]
            nucleus_mean = nucleus_pixels.mean()
            channel[nucleus_mask] = (
                nucleus_pixels - nucleus_mean
            ) * 1.3 + nucleus_mean * 0.8
            result[:, :, c] = np.clip(channel, 0, 255).astype(reference_image.dtype)
    return result


def build_texture_bank(vae, reference_image, dr_mask):
    dtype = vae.dtype
    device = vae.device
    if isinstance(reference_image, np.ndarray):
        reference_image = Image.fromarray(reference_image)
    img_tensor = torch.from_numpy(np.array(reference_image)).float().to(device)
    img_tensor = img_tensor / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(dtype)
    with torch.no_grad():
        ref_latents = (
            vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor
        )
    if dr_mask.dim() == 4:
        dr_mask = dr_mask.squeeze(0).squeeze(0)
    elif dr_mask.dim() == 3:
        dr_mask = dr_mask.squeeze(0)
    mask_bool = dr_mask > 0.5
    mask_flat = mask_bool.reshape(-1).to(device)
    if isinstance(reference_image, np.ndarray):
        ref_img_pil = Image.fromarray(reference_image)
    else:
        ref_img_pil = reference_image
    ref_small = ref_img_pil.resize((dr_mask.shape[1], dr_mask.shape[0]), Image.BICUBIC)
    ref_float = np.array(ref_small).astype(float)
    od_image = -np.log10((ref_float + 1.0) / 256.0)
    od_gray = np.max(od_image, axis=2) if od_image.ndim == 3 else od_image
    od_threshold = 0.03
    tissue_mask = od_gray > od_threshold
    tissue_mask_flat = torch.from_numpy(tissue_mask).reshape(-1).to(device)
    features = ref_latents.squeeze(0).reshape(16, -1).permute(1, 0)
    nucleus_bank = features[mask_flat]
    valid_bg_mask = ~mask_flat & tissue_mask_flat
    if valid_bg_mask.sum() > 10:
        bg_bank = features[valid_bg_mask]
    else:
        bg_bank = features[~mask_flat]
        print(
            "  [Texture Bank] Warning: Tissue filter removed too many pixels, fallback to full background."
        )
    if bg_bank.shape[0] == 0:
        bg_bank = features
        print(
            "  [Texture Bank] Warning: Background bank empty, fallback to all features."
        )
    return (nucleus_bank.to(dtype=dtype), bg_bank.to(dtype=dtype))


def latent_patchmatch(latents_struct, texture_bank, soft_mask):
    device = latents_struct.device
    dtype = latents_struct.dtype
    B, C, H, W = latents_struct.shape
    if isinstance(soft_mask, np.ndarray):
        soft_mask = torch.from_numpy(soft_mask).to(device)
    soft_mask = soft_mask.to(device).to(dtype)
    if soft_mask.dim() == 2:
        mask = soft_mask
    elif soft_mask.dim() == 3:
        mask = soft_mask.squeeze(0)
    elif soft_mask.dim() == 4:
        mask = soft_mask.squeeze(0).squeeze(0)
    else:
        raise ValueError(f"Invalid mask dimensions: {soft_mask.shape}")
    if isinstance(texture_bank, np.ndarray):
        texture_bank = torch.from_numpy(texture_bank).to(device).to(dtype)
    texture_bank = texture_bank.to(device).to(dtype)
    repair_threshold = 0.1
    repair_mask = mask > repair_threshold
    if not repair_mask.any():
        return latents_struct
    if texture_bank.numel() == 0 or texture_bank.shape[0] == 0:
        return latents_struct
    repair_coords = torch.nonzero(repair_mask, as_tuple=False)
    ys, xs = (repair_coords[:, 0], repair_coords[:, 1])
    current_features = latents_struct[0, :, ys, xs].T
    curr_norm = F.normalize(current_features.float(), p=2, dim=1)
    bank_norm = F.normalize(texture_bank.float(), p=2, dim=1)
    distances = torch.cdist(curr_norm, bank_norm)
    nearest_idx = distances.argmin(dim=1)
    matched_textures = texture_bank[nearest_idx]
    latents_repaired = latents_struct.clone()
    weights = mask[ys, xs].unsqueeze(1)
    blended_feats = (1 - weights) * current_features + weights * matched_textures
    latents_repaired[0, :, ys, xs] = blended_feats.T
    return latents_repaired


def create_frequency_mask(height, width, radius, transition_width=0.5, device=None):
    if device is None:
        device = torch.device("cpu")
    u = torch.arange(height, device=device)
    v = torch.arange(width, device=device)
    u, v = torch.meshgrid(u, v, indexing="ij")
    center_u, center_v = (height // 2, width // 2)
    dist = torch.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
    mask = torch.exp(-((dist - radius) ** 2) / (2 * transition_width**2))
    mask = torch.where(dist <= radius, torch.ones_like(mask), mask)
    return mask


def apply_fss_noise(
    latents: torch.Tensor,
    noise_std: float = 1.0,
    radius: float = 4.0,
    transition_width: float = 0.5,
    pad_factor: float = 2.0,
    generator: torch.Generator = None,
) -> torch.Tensor:
    latents = latents.float()
    device = latents.device
    dtype = latents.dtype
    B, C, H, W = latents.shape
    pad_h = int(H * (pad_factor - 1))
    pad_h = pad_h // 2 * 2
    pad_w = int(W * (pad_factor - 1))
    pad_w = pad_w // 2 * 2
    padded_latents = F.pad(
        latents.float(),
        (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2),
        mode="reflect",
    )
    pH, pW = padded_latents.shape[-2:]
    if generator is not None:
        noise = torch.randn(
            padded_latents.shape, generator=generator, device=device, dtype=dtype
        )
    else:
        noise = torch.randn_like(padded_latents, dtype=dtype)
    fft_guide = torch.fft.fft2(padded_latents, dim=(-2, -1))
    fft_noise = torch.fft.fft2(noise, dim=(-2, -1))
    fft_guide = torch.fft.fftshift(fft_guide, dim=(-2, -1))
    fft_noise = torch.fft.fftshift(fft_noise, dim=(-2, -1))
    phase_guide = torch.angle(fft_guide)
    phase_noise = torch.angle(fft_noise)
    mag_noise = torch.abs(fft_noise)
    phase_guide_clip_thresh = torch.quantile(torch.abs(phase_guide), 0.95)
    phase_guide = torch.clamp(
        phase_guide, -phase_guide_clip_thresh, phase_guide_clip_thresh
    )
    mag_clip_thresh = torch.quantile(mag_noise, 0.95)
    mag_noise = torch.clamp(mag_noise, max=mag_clip_thresh)
    mag_noise = mag_noise * noise_std
    mask = create_frequency_mask(pH, pW, radius, transition_width, device)
    mask = mask.view(1, 1, pH, pW)
    mixed_phase = mask * phase_guide + (1 - mask) * phase_noise
    fft_new = mag_noise * torch.exp(1j * mixed_phase)
    fft_new = torch.fft.ifftshift(fft_new, dim=(-2, -1))
    structured_noise = torch.fft.ifft2(fft_new, dim=(-2, -1)).real
    clamp_mask = (structured_noise < -5) | (structured_noise > 5)
    structured_noise = torch.where(clamp_mask, noise, structured_noise)
    output = structured_noise[
        :, :, pad_h // 2 : pad_h // 2 + H, pad_w // 2 : pad_w // 2 + W
    ]
    return output.to(torch.float16)
