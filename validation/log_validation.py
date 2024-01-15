from typing import TypedDict
from accelerate.logging import get_logger
import numpy as np
from torch import nn
import torch
from dataset.models import Annotation
from dataset.visual_genome_dataset import VisualGenomeValidation
from inference.inference_with_pretrained_pipeline import run_inference_with_pipeline
from inference.run_inference import run_inference
from utils.save_progress import log_validation_image, save_captions_to_file, save_layouts
import wandb
from utils.helpers import convert_pil_to_tensor

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
    # TODO fix the layout of the crop
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
                    "validation_images": [
                        wandb.Image(image, caption=f"{val_batch['raw_data'][i]['id']}: {captions[i]}", boxes = get_wandb_bounding_boxes(image, val_batch['raw_data'][i], val_dataloader.dataset))
                        for i, image in enumerate(val_batch["pixel_values"])
                    ]
                }
            )

            tracker.log(
                {
                    "validation_processed_images": [
                        wandb.Image(image, caption=f"{val_batch['raw_data'][i]['id']}: {captions[i]}")
                        for i, image in enumerate(val_batch["pixel_values"])
                    ]
                }
            )

            tracker.log(
                {
                    "conditioned_validation_images": [
                        wandb.Image(image, caption=f"{i}: {captions[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )

            tracker.log(
                {
                    "unconditioned_validation_images": [
                        wandb.Image(unconditioned_images, caption=f"{i}: {captions[i]}")
                        for i, image in enumerate(unconditioned_images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    # Save layout conditions
    log_validation_image(val_batch["pixel_values"], epoch, global_step, output_dir, "val_image")
    log_validation_image(images, epoch, global_step, output_dir, "cond_image")
    log_validation_image(unconditioned_images, epoch, global_step, output_dir, "uncond_image")
    log_validation_image(convert_pil_to_tensor(unconditioned_images_with_pipeline), epoch, global_step, output_dir, "uncond_image_pipeline")


    torch.cuda.empty_cache()

    return images

def get_wandb_bounding_boxes(image_pixels, image_data, dataset: VisualGenomeValidation):
    all_boxes = []
    annotations: list[Annotation] = image_data['annotations'] 
    bbox_builder = dataset.get_bounding_box_builders()[0]["bounding_boxes"]
    width, height = image_pixels.shape[1], image_pixels.shape[2]
    # width, height = image_pixels.height, image_pixels.width
    crop_bbox, bboxes = bbox_builder.get_bounding_boxes_from_condition(width, height, image_data["bounding_boxes"])
    # bbox =  # x0, y0, w, h
    # for ann in annotations:
    #     bbox = ann.bbox
    #     box_data = {"position" : {
    #         "minX" : bbox[0] * width,
    #         "maxX" : bbox[2] * width,
    #         "minY" : bbox[1] * height,
    #         "maxY" : bbox[3] * height},
    #         "class_id" : int(ann.category_id),
    #         # optionally caption each box with its class and score
    #         "box_caption" : dataset.category_id_to_name_dict()[int(ann.category_id)],
    #         "domain" : "pixel"
    #         }
    #     all_boxes.append(box_data)

    for cat_no, bbox in bboxes:
        box_data = {"position" : {
            "minX" : bbox[0],
            "maxX" : bbox[2],
            "minY" : bbox[1],
            "maxY" : bbox[3]},
            "class_id" : int(cat_no),
            # optionally caption each box with its class and score
            "box_caption" : dataset.category_no_to_name_dict()[int(cat_no)],
            "domain" : "pixel"
            }
        all_boxes.append(box_data)
    if crop_bbox is not None:
        box_data = {"position" : {
            "minX" : crop_bbox[0],
            "maxX" : crop_bbox[2],
            "minY" : crop_bbox[1],
            "maxY" : crop_bbox[3]},
            "class_id" : 0,
            # optionally caption each box with its class and score
            "box_caption" : "crop",
            "domain" : "pixel"
            }
        all_boxes.append(box_data)

    

    return {"predictions": {"box_data": all_boxes, "class_labels" : dataset.category_no_to_name_dict()}}

