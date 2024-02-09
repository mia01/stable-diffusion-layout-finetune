from torchmetrics.image.kid import KernelInceptionDistance

import numpy as np
import torch
import PIL.Image as pil_image

from utils.helpers import convert_pil_to_tensor

# KID https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html
# is a metric similar to the FID. Also based on the Inception network, it measures the dissimilarity between two probability distributions Pr and Pg using samples drawn independently from each distribution

kid = KernelInceptionDistance(normalize=True)
def calculate_kid_score(real_images, generated_images):

    if type(real_images[0]) == pil_image.Image:
        real_images = [convert_pil_to_tensor(image) for image in real_images]
    if type(generated_images[0]) == pil_image.Image:
        generated_images = [convert_pil_to_tensor(image) for image in generated_images]

    real_images_tensor = torch.stack(real_images)
    generated_image_tensor = torch.stack(generated_images)

    kid.update(real_images_tensor, real=True)
    kid.update(generated_image_tensor, real=False)

    mean, std = kid.compute()
    return float(mean), float(std)