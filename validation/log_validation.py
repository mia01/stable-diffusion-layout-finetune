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
import torch
from torchvision import transforms
class ModelComponents(TypedDict):
    vae: nn.Module
    text_encoder: nn.Module
    unet: nn.Module
    layout_embedder: nn.Module
    noise_scheduler: nn.Module
    tokenizer: nn.Module

def convert_pil_list_to_tensor(pil_list):

    # Define a transform to convert PIL images to tensors
    transform = transforms.ToTensor()

    # Apply the transform to each image and store the results in a list
    tensors = [transform(image) for image in pil_list]

    # Stack all tensors into a single tensor
    tensor_stack = torch.stack(tensors)

    return tensor_stack


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
        # logger.info(f"Epoch: {epoch} Running inference with text condition only and pretrained pipeline")
        # unconditioned_images_with_pipeline = run_inference_with_pipeline(accelerator, val_batch, "CompVis/stable-diffusion-v1-4", model_components["unet"], seed, num_inference_steps)

        logger.info(f"Epoch: {epoch} Running inference with layout and text condition")
        images = run_inference(accelerator, val_batch, model_components, seed, num_inference_steps, True)
        
        logger.info(f"Epoch: {epoch} Running inference with text condition only")
        unconditioned_images = run_inference(accelerator, val_batch, model_components, seed, num_inference_steps, False)

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            captions = [vb["captions"] for vb in val_batch["raw_data"]]

            # https://wandb.ai/stacey/yolo-drive/reports/Exploring-Bounding-Boxes-for-Object-Detection-With-Weights-Biases--Vmlldzo4Nzg4MQ
            tracker.log(
                {
                    "original_validation_images": [
                        wandb.Image(val_batch['raw_data'][i]["original_image"], caption=f"{val_batch['raw_data'][i]['id']}: {captions[i]}", boxes = get_wandb_bounding_boxes_for_original_images(val_batch['raw_data'][i], val_dataloader.dataset))
                        for i, image in enumerate(val_batch["pixel_values"])
                    ]
                }
            )

            tracker.log(
                {
                    "validation_processed_images": [
                        wandb.Image(image, caption=f"{val_batch['raw_data'][i]['id']}: {captions[i]}", boxes = get_wandb_bounding_boxes(image, val_batch['raw_data'][i], val_dataloader.dataset))
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
                        wandb.Image(image, caption=f"{i}: {captions[i]}")
                        for i, image in enumerate(unconditioned_images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    # Save layout conditions
    logger.info(f"Epoch: {epoch} Logging inference images")
    log_validation_image(val_batch["pixel_values"], epoch, global_step, output_dir, "val_image")
    log_validation_image(images, epoch, global_step, output_dir, "cond_image")
    log_validation_image(unconditioned_images, epoch, global_step, output_dir, "uncond_image")
    # log_validation_image(convert_pil_list_to_tensor(unconditioned_images_with_pipeline), epoch, global_step, output_dir, "uncond_image_pipeline")



    torch.cuda.empty_cache()

    return images

def get_wandb_bounding_boxes(image_pixels, image_data, dataset: VisualGenomeValidation):
    all_boxes = []
    bbox_builder = dataset.get_bounding_box_builders()[0]["bounding_boxes"]
    width, height = image_pixels.shape[1], image_pixels.shape[2]
    crop_bbox, bboxes = bbox_builder.get_bounding_boxes_from_condition(width, height, image_data["bounding_boxes"])

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
            }
        all_boxes.append(box_data)
        
    return {"predictions": {"box_data": all_boxes, "class_labels" : dataset.category_no_to_name_dict()}}

    
def get_wandb_bounding_boxes_for_original_images(image_data, dataset: VisualGenomeValidation):
    all_boxes = []
    annotations: list[Annotation] = image_data['annotations'] 
    width, height = image_data["original_image"].size[0], image_data["original_image"].size[1]

    # bounding box on original images
    for ann in annotations:
        box_data = {"position" : {
            "minX" : ann.bbox[0] * width,
            "maxX" : (ann.bbox[0] + ann.bbox[2]) * width,
            "minY" : ann.bbox[1] * height,
            "maxY" : (ann.bbox[1] + ann.bbox[3]) * height},
            "class_id" : int(ann.category_no),
            # optionally caption each box with its class and score
            "box_caption" : dataset.category_no_to_name_dict()[int(ann.category_no)],
            "domain" : "pixel"
            }
        all_boxes.append(box_data)

    return {"predictions": {"box_data": all_boxes, "class_labels" : dataset.category_no_to_name_dict()}}

