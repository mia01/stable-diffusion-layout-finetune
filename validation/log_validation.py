from typing import TypedDict
from accelerate.logging import get_logger
import numpy as np
from torch import nn
import torch
from inference.inference_with_pretrained_pipeline import run_inference_with_pipeline
from inference.run_inference import run_inference
from utils.save_progress import log_validation_image, save_captions_to_file, save_layouts
import wandb

class ModelComponents(TypedDict):
    vae: nn.Module
    text_encoder: nn.Module
    unet: nn.Module
    layout_embedder: nn.Module
    noise_scheduler: nn.Module
    tokenizer: nn.Module


def log_validation(accelerator, val_dataloader, model_components: ModelComponents,
                    epoch: int,
                    global_step: int,
                    seed: int,
                    num_inference_steps: int,
                    output_dir: str):
    logger = get_logger(__name__, log_level="INFO")

    logger.info("Running validation image inference... ")

    itervalloader = iter(val_dataloader)
    try:
        val_batch = next(itervalloader)

    except StopIteration:
        itervalloader = iter(itervalloader)
        val_batch = next(itervalloader)

    # Save text condition
    save_captions_to_file(val_batch["raw_data"], epoch, global_step, output_dir, "cond_text")
    data_set = val_dataloader.dataset
    save_layouts(data_set, val_batch, epoch, global_step, output_dir, "cond_layout")

    images = []
   
    with torch.autocast("cuda"):
        images = run_inference(accelerator, val_batch, model_components, seed, num_inference_steps)
        
        unconditioned_images = run_inference(accelerator, val_batch, model_components, seed, num_inference_steps)

        unconditioned_images_with_pipeline = run_inference_with_pipeline(accelerator, val_batch, "CompVis/stable-diffusion-v1-4", model_components["unet"], seed, num_inference_steps)
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            # np_images = np.stack([np.asarray(img) for img in images]) caused error: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            
            # np_images = np.stack([np.asarray(img.cpu().permute(0, 2, 3, 1)) for img in images])
            # Assuming 'images' is your PyTorch tensor with shape [4, 3, 512, 512]
            # Convert PyTorch tensor to NumPy array and permute dimensions to NHWC format
            np_images = images.permute(0, 2, 3, 1).cpu().numpy()

            # Normalize the images if they are in float format
            if np_images.dtype == np.float32:
                np_images = np.clip(np_images, 0, 1)

            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            captions = [vb["captions"] for vb in val_batch["raw_data"]]
            all_boxes = []
            # todo: plot each bounding box for this image
            # https://wandb.ai/stacey/yolo-drive/reports/Exploring-Bounding-Boxes-for-Object-Detection-With-Weights-Biases--Vmlldzo4Nzg4MQ

            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {captions[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    # Save layout conditions
    log_validation_image(val_batch["pixel_values"], epoch, global_step, output_dir, "val_image")
    log_validation_image(images, epoch, global_step, output_dir, "cond_image")
    log_validation_image(unconditioned_images, epoch, global_step, output_dir, "uncond_image")
    log_validation_image(unconditioned_images_with_pipeline, epoch, global_step, output_dir, "uncond_image_pipeline")


    torch.cuda.empty_cache()

    return images