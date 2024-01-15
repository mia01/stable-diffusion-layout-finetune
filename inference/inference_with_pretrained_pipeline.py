import torch
from diffusers import StableDiffusionPipeline

def run_inference_with_pipeline(accelerator, val_batch, pretrained_model_name_or_path: str, unet,seed:int = None, num_inference_steps:int = 100):
    unet = accelerator.unwrap_model(unet)
    # if args.use_ema:
    #     ema_unet.copy_to(unet.parameters())

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

    with torch.autocast("cuda"):
        image = pipeline("a photograph of an astronaut riding a horse", num_inference_steps=num_inference_steps, generator=generator).images[0]
        images.append(image)
    return images