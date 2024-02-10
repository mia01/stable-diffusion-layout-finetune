
import numpy as np
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial

from utils.helpers import convert_tensor_to_pil


# Both CLIP score and CLIP direction similarity rely on the CLIP model, which can make the evaluations biased.
def calculate_clip_score(generated_images, prompts):
    
   
    if type(generated_images[0]) == torch.Tensor or type(generated_images[0]) == np.ndarray:
        pil_array = [convert_tensor_to_pil(img) for img in generated_images]
        generated_images_np = np.asarray(pil_array)
    else: 
         # conveet list of images to tensor
        generated_images_np = np.asarray(generated_images)

        
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    
    images_int = (generated_images_np * 255).astype("uint8")
    clip_score_val = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score_val), 4)
