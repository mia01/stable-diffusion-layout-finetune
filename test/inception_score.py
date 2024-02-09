import torch
from torchmetrics.image.inception import InceptionScore
import PIL.Image as pil_image

from utils.helpers import convert_pil_to_tensor

# Calculate the Inception Score (IS) which is used to access how realistic generated images are
inception = InceptionScore()
def calculate_inception_score(images):
    if type(images[0]) == pil_image.Image:
        images = [convert_pil_to_tensor(image) for image in images]

    images_tensor = torch.stack(images)
    inception.update(images_tensor)

    mean, std = inception.compute()

    return float(mean), float(std)
