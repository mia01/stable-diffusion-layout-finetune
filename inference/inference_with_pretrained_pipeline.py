import torch
from diffusers import StableDiffusionPipeline
from accelerate.logging import get_logger

def run_inference_with_pipeline(accelerator, val_batch, pretrained_model_name_or_path: str,seed:int = None, num_inference_steps:int = 100):
    logger = get_logger(__name__, log_level="INFO")
    logger.info(f"Running inference pipeline with {num_inference_steps} steps... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path
    )

    if seed is None:
        generator = None
    else:
        generator = torch.manual_seed(seed)

    pipeline.set_progress_bar_config(disable=False)
    images = []
    for data in val_batch["raw_data"]:
        with torch.autocast("cuda"):
            image = pipeline(data["captions"][0], num_inference_steps=num_inference_steps, generator=generator).images[0]
        images.append(image)

    # with torch.autocast("cuda"):
    #     image = pipeline("a photograph of an astronaut riding a horse", num_inference_steps=num_inference_steps, generator=generator).images[0]
    #     images.append(image)
    return images