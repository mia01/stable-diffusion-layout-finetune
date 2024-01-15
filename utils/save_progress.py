import os
import numpy as np

import torch
from PIL import Image
import torchvision

from dataset.visual_genome_dataset import VisualGenomeValidation

def save_captions_to_file(batch, epoch:int, global_step:int, output_dir:str, file_name:str):
    """
    Saves the captions of a batch of images to a file.
    
    """
    idx = 0
    for b in batch:
      caption = "".join(b["captions"])
      text = f"Image ID: {b['id']} Caption: {caption}"

      print(f"Caption {idx}: {text}")

      save_path = os.path.join(f"{output_dir}", f"image_logs-{epoch}-{global_step}")
      filename = f"{file_name}_s-{global_step}_e-{epoch}_idx_{idx}.txt"
      path = os.path.join(save_path, filename)
      os.makedirs(os.path.split(path)[0], exist_ok=True)
      idx = idx + 1
      with open(path, 'w', encoding='utf-8') as file:
          file.write(text)


def save_tensor_as_image(tensor: torch.Tensor, filename: str):
    """
    Saves a PyTorch tensor as an image.

    Args:
    - tensor (torch.Tensor): A PyTorch tensor representing an image.
                             The tensor shape should be [C, H, W] and it should be in the range [0, 1].
    - filename (str): The path where the image will be saved.
    """
    # Ensure tensor is in CPU memory
    tensor = tensor.cpu()

    # Convert tensor to PIL Image
    if tensor.ndim == 3:  # Single image [C, H, W]
        # Normalize if the tensor is not already in [0, 1]
        if tensor.max() > 1:
            tensor = tensor / 255.0

        # Convert to [H, W, C] format for PIL and multiply by 255
        image = tensor.permute(1, 2, 0).numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
    elif tensor.ndim == 4:  # Batch of images [N, C, H, W]
        raise ValueError("Expected a single image tensor, got a batch. Please provide one image tensor at a time.")

    # Save the image
    pil_image.save(filename)


def save_layouts(data_set: VisualGenomeValidation, val_batch, epoch, global_step, output_dir, file_name):
    builders = data_set.get_bounding_box_builders()
    bbox_builder = builders[0]["bounding_boxes"]
    figure_size = (512,512) # val_batch["pixel_values"].shape[2], val_batch["pixel_values"].shape[3]

    plots = []

    for bbox in val_batch["bounding_boxes"]:
      plot = bbox_builder.plot(bbox.long(), data_set.get_category_no_to_id_dict(), figure_size)
      plots.append(plot)
    idx = 0
    for plot in plots:
      save_path = os.path.join(f"{output_dir}", f"image_logs-{epoch}-{global_step}")
      filename = f"{file_name}_s-{global_step}_e-{epoch}_idx-{idx}.png"
      idx = idx + 1
      path = os.path.join(save_path, filename)
      os.makedirs(os.path.split(path)[0], exist_ok=True)

      save_tensor_as_image(plot, path)

def log_validation_image(batch_tensors, epoch, global_step, output_dir, image_name):
  save_path = os.path.join(f"{output_dir}", f"image_logs-{epoch}-{global_step}")

  # Create a grid of images
  grid = torchvision.utils.make_grid(batch_tensors, nrow=4, normalize=True, pad_value=1)

  # Convert to PIL Image
  grid = grid.permute(1, 2, 0)  # Change from [C, H, W] to [H, W, C]
  grid = grid.cpu().numpy()     # Convert to NumPy array
  grid_image = (grid * 255).astype(np.uint8)
  pil_image = Image.fromarray(grid_image)
  filename = f"{image_name}_s-{global_step}_e-{epoch}.png"
  path = os.path.join(save_path, filename)
  os.makedirs(os.path.split(path)[0], exist_ok=True)
  pil_image.save(path)