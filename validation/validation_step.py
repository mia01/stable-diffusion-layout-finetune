from typing import TypedDict
import torch
from dataset.helpers import get_layout_conditioning

from utils.helpers import default
import torch.nn.functional as functional
from torch import nn

class PipelineComponents(TypedDict):
    vae: nn.Module
    text_encoder: nn.Module
    unet: nn.Module
    layout_embedder: nn.Module
    scheduler: nn.Module
    tokenizer: nn.Module

@torch.no_grad()
def validation_step(accelerator, val_dataloader, pipeline: PipelineComponents, num_timesteps, epoch, global_step):
    # setting the unet model within the pipeline object to evaluation mode
    pipeline["unet"].eval()
    pipeline["layout_embedder"].eval()
    val_loss = 0
    num_batches = 0
    for val_batch_idx, val_batch in enumerate(val_dataloader):
        # convert image to latent space
        val_batch_size = val_batch["pixel_values"].shape[0]
        latents = pipeline["vae"].encode(val_batch["pixel_values"]).latent_dist.sample()
        latents = latents * pipeline["vae"].config.scaling_factor
        
        bsz = latents.shape[0] # should be the size of the first stage of encoding an image
        timesteps = torch.randint(0, num_timesteps, (bsz,), device=latents.device) # get random timestep vector
        timesteps = timesteps.long()

        noise = torch.randn_like(latents) # sample noise (should be ths same size as the latent space)
        noisy_latents = pipeline["scheduler"].add_noise(latents, noise, timesteps) # add noise at t timestep

        # Get the text embedding for conditioning
        encoder_hidden_states = pipeline["text_encoder"](val_batch["input_ids"])[0]

        # Add layout embedding and concatenate with text embedding
        val_batch["bounding_boxes"].to(accelerator.device)
        layout_conditioning = get_layout_conditioning(pipeline["layout_embedder"], val_batch["bounding_boxes"])
        layout_conditioning.to(accelerator.device)
        encoder_hidden_states.to(accelerator.device)

        # Concatenate embeddings
        condition = torch.cat((encoder_hidden_states, layout_conditioning), 1)
        
        target = noise # what the unet should be able to predict
        # Predict the noise residual and compute loss
        model_pred = pipeline["unet"](noisy_latents, timesteps, condition).sample
        loss = functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
        avg_loss = accelerator.gather(loss.repeat(val_batch_size)).mean()
        val_loss += avg_loss.item() 
        num_batches += 1
        
    # Log validation loss
    accelerator.log({"val_loss": val_loss/ num_batches}, step=global_step)

        # TODO compute loss with layout conditioning only
        # TODO compute loss with text condition only
        # TODO compute loss with no condition

    pipeline["unet"].train()
    pipeline["layout_embedder"].train()