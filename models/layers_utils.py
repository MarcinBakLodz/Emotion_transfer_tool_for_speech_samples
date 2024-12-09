import torch
import random

def crop_random_patch(image):
    _, _, H, W = image.shape  # Zamiana W z H
    
    scaling_factor = random.uniform(1/8, 1/4)
    
    patch_height = int(H * scaling_factor)
    patch_width = int(W * scaling_factor)  # Obliczamy szerokość na podstawie szerokości, a nie wysokości

    if patch_width > W or patch_height > H:
        scaling_factor = min(W / patch_width, H / patch_height)
        patch_height = int(H * scaling_factor)
        patch_width = int(W * scaling_factor)  # Poprawka również tutaj

    top = random.randint(0, H - patch_height)
    left = random.randint(0, W - patch_width)

    patch = image[:, :, top:top + patch_height, left:left + patch_width]  # Poprawiona kolejność indeksów

    target_size = H // 4  # Zakładam, że docelowy rozmiar powinien być proporcjonalny do wysokości

    patch_resized = torch.nn.functional.interpolate(patch, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return patch_resized