"""Corrección cromática White-Patch: (R,G,B) -> (255/R_max*R, 255/G_max*G, 255/B_max*B)."""
import numpy as np


def apply_white_patch(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normaliza la respuesta cromática de la imagen (BGR uint8) mediante White-Patch."""
    if image is None or image.size == 0:
        raise ValueError("Imagen vacía o nula recibida en white_patch_normalizer.")

    img_float = image.astype(np.float32)
    b_max, g_max, r_max = (img_float[..., 0].max(), img_float[..., 1].max(), img_float[..., 2].max())

    b_scale = 255.0 / (b_max + eps)
    g_scale = 255.0 / (g_max + eps)
    r_scale = 255.0 / (r_max + eps)

    out = np.empty_like(img_float)
    out[..., 0] = img_float[..., 0] * b_scale
    out[..., 1] = img_float[..., 1] * g_scale
    out[..., 2] = img_float[..., 2] * r_scale

    return np.clip(out, 0, 255).astype(np.uint8)