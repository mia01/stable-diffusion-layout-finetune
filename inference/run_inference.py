from typing import TypedDict
import torch
from tqdm.auto import tqdm
from torch import nn

from dataset.helpers import get_layout_conditioning

class PipelineComponents(TypedDict):
    vae: nn.Module
    text_encoder: nn.Module
    unet: nn.Module
    layout_embedder: nn.Module
    noise_scheduler: nn.Module
    tokenizer: nn.Module

class inputs(TypedDict):
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    bounding_boxes: torch.Tensor


def run_inference(accelerator, inputs, pipeline_components: PipelineComponents, seed: int | None = None, num_inference_steps = 100, exclude_layout_condition = False):

  with torch.no_grad():

    # Get text embedding (conditioning)
    encoder_hidden_states = pipeline_components["text_encoder"](inputs["input_ids"])[0]
    print("Got text condition")

    # Add layout embedding and concatenate with text embedding
    if exclude_layout_condition != False:
      inputs["bounding_boxes"].to(accelerator.device)
      layout_conditioning = get_layout_conditioning(pipeline_components["layout_embedder"], inputs["bounding_boxes"])
      layout_conditioning.to(accelerator.device)
      encoder_hidden_states.to(accelerator.device)
      print("Got bbox condition")

  if exclude_layout_condition != False:
    # Concatenate layout and text embeddings
    condition = torch.cat((encoder_hidden_states, layout_conditioning), 1)
  else:
    condition = encoder_hidden_states

  # Initialize the scheduler with our chosen num_inference_steps
  pipeline_components["noise_scheduler"].set_timesteps(num_inference_steps)

  if seed is not None:
    generator = torch.manual_seed(seed)
  else:
      generator = None

  # Generate random noise to start with
  batch_size = inputs["pixel_values"].shape[0]
  image_width = inputs["pixel_values"].shape[2]
  image_height = inputs["pixel_values"].shape[3]
  latents = torch.randn(
    (batch_size, pipeline_components["unet"].in_channels, image_width //8, image_height // 8),
    generator=generator,
  )
  latents = latents.to(accelerator.device)

  for t in tqdm(pipeline_components["noise_scheduler"].timesteps):

      latent_model_input = pipeline_components["noise_scheduler"].scale_model_input(latents, timestep=t)

      # predict the noise residual
      with torch.no_grad():
          noise_pred = pipeline_components["unet"](latent_model_input, t, condition).sample
          
      # compute the previous noisy sample x_t -> x_t-1
      latents = pipeline_components["noise_scheduler"].step(noise_pred, t, latents).prev_sample

      # scale and decode the image latents with vae
      latents = 1 / pipeline_components["vae"].config.scaling_factor * latents # see https://github.com/huggingface/diffusers/issues/726

  with torch.no_grad():
    image = pipeline_components["vae"].decode(latents).sample
    return image