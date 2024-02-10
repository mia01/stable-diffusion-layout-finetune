
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import PIL.Image as pil_image

from utils.helpers import convert_pil_to_tensor, normalise_tensors

# FID results tend to be fragile as they depend on a lot of factors:
# The specific Inception model used during computation.
# The implementation accuracy of the computation.
# The image format (not the same if we start from PNGs vs JPGs).
# Keeping that in mind, FID is often most useful when comparing similar runs, but it is hard to reproduce paper results unless the authors carefully disclose the FID measurement code.
# These points apply to other related metrics too, such as KID and IS.
def calculate_fid_score(real_images, generated_images):

    fid = FrechetInceptionDistance(normalize=False)

    if type(real_images[0]) == pil_image.Image:
        real_images = [convert_pil_to_tensor(image) for image in real_images]
    if type(generated_images[0]) == pil_image.Image:
        generated_images = [convert_pil_to_tensor(image) for image in generated_images]

    real_images_tensor = torch.stack(real_images)
    generated_image_tensor = torch.stack(generated_images)

    real_images_tensor = normalise_tensors(real_images_tensor)
    generated_image_tensor = normalise_tensors(generated_image_tensor)

    fid.update(real_images_tensor, real=True)
    fid.update(generated_image_tensor, real=False)

    score = float(fid.compute())
    return score