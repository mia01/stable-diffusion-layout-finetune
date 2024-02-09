
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Test script.")
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="sd-model-layout2image-finetuned/checkpoint-0-0",
        required=False,
        help="Path to the model to be tested.",
    )
    parser.add_argument(
        "--test_annotation_file",
        type=str,
        default="/Users/miav1/Desktop/university/2023/disseration/datasets/visual genome/data/test_coco_style.json",
        help="Visual genome annotation file",
    )
    parser.add_argument(
        "--test_caption_file",
        type=str,
        default="/Users/miav1/Desktop/university/2023/disseration/datasets/visual genome/data/test_sg.json",
        help="Visual genome annotation file",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/Users/miav1/Desktop/university/2023/disseration/datasets/visual genome/data/images/VG_100K",
        help="Visual genome image folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_output",
        required=False,
        help="Path to save the output.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="test_logs",
        required=False,
        help="Path to save the test log output.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible testing.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=1, help="Batch size (per device) for the testing dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=5,
        help=(
            "The maximum number of objects to detect in the images."
        ),
    )
    parser.add_argument(
        "--max_test_steps",
        type=int,
        default=0,
        help=(
            "The maximum number of test steps."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help=(
            "The number of steps during inferece"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="test-layout2image-fine-tune-local",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args("") # pass empty - trick to force it to default inside notebook
  
    return args