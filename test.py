import os

import wandb
from dataset.helpers import get_layout_conditioning
from dataset.visual_genome_dataset import FilteredTestDataset, VisualGenomeTrain, VisualGenomeValidation
from embedders.layout_embedder import LayoutEmbedder, LayoutEmbedderConfig, LayoutEmbedderModel
from inference.inference_with_pretrained_pipeline import run_inference_with_pipeline
from inference.run_inference import run_inference
from test.kid_score import calculate_kid_score
from test.parse_args import parse_args
import logging
import math
import os
import shutil
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as functional
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from test.clip_score import calculate_clip_score
from test.fid_score import calculate_fid_score
from test.inception_score import calculate_inception_score
from utils.helpers import convert_pil_list_to_tensor, convert_pil_to_tensor
from utils.save_progress import get_layout_image_for_bbox, log_validation_images_separate, save_layouts
from validation.log_validation import get_wandb_bounding_boxes
from validation.validation_step import validation_step
from diffusers.utils.import_utils import is_xformers_available


# TODO - look into yoloscore
# https://github.com/ZGCTroy/LayoutDiffusion/blob/master/scripts/lama_yoloscore_test.py
# https://pjreddie.com/darknet/yolo/
# https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_prc.py - precision and recall


def test():
    args = parse_args()

    # set logging with accelerate
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger = get_logger(__name__, log_level="INFO")
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Args:")
    logger.info(args)
    
    # SE: Load all pre-trained components
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    layout_embedder_config = LayoutEmbedderConfig()
    layout_embedder = LayoutEmbedderModel(layout_embedder_config, device=accelerator.device)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        )
        
    # Added this to fix an issue on cpu only machines - may need removing for GPU
    
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        layout_embedder.requires_grad_(False)

        # Load the datasets
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
        test_dataset = FilteredTestDataset(
            args.test_annotation_file,
            args.test_caption_file,
            args.image_folder,
            args.resolution,
            args.max_objects
        )
        # Dataloader collate functions
        # We need to tokenize input captions and transform the images.
        def tokenize_captions(examples, is_train=True):
            captions = []

            # for caption in examples[caption_column]: from before when we used preprocess method
            for example in examples:
                caption = example["captions"]
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(','.join(caption))
                else:
                    raise ValueError(
                        f"Caption column captions should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids

        # Define the collate_fn for the dataloader
        def collate_fn(examples: list):
            pixel_values = torch.stack([example["normalised_image"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            
            bounding_boxes = torch.stack([example["bounding_boxes"] for example in examples])
            bounding_boxes = bounding_boxes.to(memory_format=torch.contiguous_format).long()

            tokens = tokenize_captions(examples)
            input_ids = torch.stack([token for token in tokens])
            input_ids = input_ids.to(memory_format=torch.contiguous_format).long()


            print(f"collating {len(examples)}")
            raw_data = [example for example in examples]
            return {"pixel_values": pixel_values,  "bounding_boxes": bounding_boxes, "input_ids": input_ids, "raw_data":raw_data }

       
        # DataLoaders creation:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=collate_fn,
            batch_size=args.test_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        check_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=collate_fn,
            batch_size=args.test_batch_size,
            num_workers=args.dataloader_num_workers,
        )
        # check all images load 
        logger.info("Checking all images load")
        for i, batch in enumerate(check_dataloader):
            logger.info(f"batch {i} {batch['pixel_values'].shape} - loaded successfully")
            

        # Prepare everything with our `accelerator`.
        # TODO - check is optimiser  and lr_scheduler needed
        logger.info("device")
        logger.info(accelerator.device)
        unet, test_dataloader, layout_embedder = accelerator.prepare(
            unet, test_dataloader, layout_embedder
        )

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.device.type == "cpu":
            weight_dtype = torch.float32
            args.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            args.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            args.mixed_precision = accelerator.mixed_precision

        # Move text_encode and vae to gpu and cast to weight_dtype
        text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae = vae.to(accelerator.device, dtype=weight_dtype)
        

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
            
         # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            

            def load_model_hook(models, input_dir):

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    
                    if model.__class__.__name__ == "LayoutEmbedderModel":
                        load_model = LayoutEmbedderModel.from_pretrained(input_dir, subfolder=model.__class__.__name__)
                        # model.register_to_config(**load_model.config)
                    elif model.__class__.__name__ == "UNet2DConditionModel":
                        load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder=model.__class__.__name__)
                        model.register_to_config(**load_model.config)
                    elif model.__class__.__name__ == "AutoencoderKL":
                        load_model = AutoencoderKL.from_pretrained(input_dir, subfolder=model.__class__.__name__)
                        model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_load_state_pre_hook(load_model_hook) # runs before load_state
        
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_config = dict(vars(args))
            accelerator.init_trackers(args.tracker_project_name, tracker_config)

        # Test
        num_test_samples = len(test_dataset)
        

        logger.info("***** Running Testing Loop *****")
        logger.info(f"  Num test samples = {len(test_dataset)}")
        logger.info(f"  Number of test samples with <= {args.max_objects}: {len(test_dataloader)}")
        logger.info(f"  Instantaneous batch size per device = {args.test_batch_size}")
        logger.info(f"  Max test steps = {args.max_test_steps}")
        global_step = 0

        accelerator.print(f"Loading from checkpoint {args.checkpoint_dir}")
        accelerator.load_state(args.checkpoint_dir)

        progress_bar = tqdm(
            range(0, num_test_samples),
            initial=0,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        # todo divide test sets by number of objects and compute scores
        # 2 - 10 
        # 10 - 20
        # 20 - 30

        all_text_onditioned_sd_pipeline_images = []
        all_text_layout_conditioned_images = []
        all_layout_only_conditioned_images = []
        all_text_only_conditioned_images = []
        original_validation_images = []
        prompts = []

        logger.info(f"Starting the testing")

        log_table_columns = ["Test Image Id", "Test Image", "No. Objects" ,"Layout","Caption", "Generated Image"]
        # Fix for RuntimeError: Input type (float) and bias type (c10::Half) should be the same
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
            for step, test_batch in enumerate(test_dataloader):
                progress_bar.update(args.test_batch_size)       
                logger.info(f"Step: {step} Running test inference")
                
                model_components = {
                    "vae":vae,
                    "text_encoder": text_encoder,
                    "unet": unet, 
                    "layout_embedder": layout_embedder,
                    "noise_scheduler": noise_scheduler,
                    "tokenizer": tokenizer
                    }
                # with torch.autocast("cuda"): Check with and without - Huggingface recommend not to use Don’t use torch.autocast in any of the pipelines as it can lead to black images and is always slower than pure float16 precision. see https://huggingface.co/docs/diffusers/optimization/fp16
                original_validation_images += test_batch["pixel_values"]
                captions = [",".join(vb["captions"]) for vb in test_batch["raw_data"]]
                prompts += captions

                logger.info(f"Step: {step} Running inference with text condition only and original pretrained pipeline")
                text_conditioned_sd_pipeline_images = run_inference_with_pipeline(accelerator, test_batch, "CompVis/stable-diffusion-v1-4", args.seed, args.num_inference_steps)
                all_text_onditioned_sd_pipeline_images += text_conditioned_sd_pipeline_images

                logger.info(f"Step: {step} Running inference with layout and text condition")
                text_layout_conditioned_images = run_inference(accelerator, test_batch, model_components, args.seed, args.num_inference_steps, True, True)
                all_text_layout_conditioned_images += text_layout_conditioned_images

                logger.info(f"Step: {step} Running inference with layout condition only")
                layout_only_conditioned_images = run_inference(accelerator, test_batch, model_components, args.seed, args.num_inference_steps, True, False)
                all_layout_only_conditioned_images += layout_only_conditioned_images

                logger.info(f"Step: {step} Running inference with text condition only")
                text_only_conditioned_images = run_inference(accelerator, test_batch, model_components, args.seed, args.num_inference_steps, False, True)
                all_text_only_conditioned_images += text_only_conditioned_images

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        
                        captions = [vb["captions"] for vb in test_batch["raw_data"]]
                        
                        # https://wandb.ai/stacey/yolo-drive/reports/Exploring-Bounding-Boxes-for-Object-Detection-With-Weights-Biases--Vmlldzo4Nzg4MQ
                        tracker.log(
                            {
                                "test_images": [
                                    wandb.Image(image, caption=f"{test_batch['raw_data'][i]['id']}: {captions[i]}", boxes = get_wandb_bounding_boxes(image, test_batch['raw_data'][i], test_dataloader.dataset))
                                    for i, image in enumerate(test_batch["pixel_values"])
                                ]
                            }
                        )

                        tracker.log(
                            {
                                "text_conditioned_sd_pipeline_images": [
                                    wandb.Image(image, caption=f"{i}: {captions[i]}")
                                    for i, image in enumerate(text_conditioned_sd_pipeline_images)
                                ]
                            }
                        )

                        tracker.log(
                            {
                                "text_layout_conditioned_images": [
                                    wandb.Image(image, caption=f"{i}: {captions[i]}")
                                    for i, image in enumerate(text_layout_conditioned_images)
                                ]
                            }
                        )

                        tracker.log(
                            {
                                "layout_only_conditioned_images": [
                                    wandb.Image(image, caption=f"{i}: {captions[i]}")
                                    for i, image in enumerate(layout_only_conditioned_images)
                                ]
                            }
                        )

                        tracker.log(
                            {
                                "text_only_conditioned_images": [
                                    wandb.Image(image, caption=f"{i}: {captions[i]}")
                                    for i, image in enumerate(text_only_conditioned_images)
                                ]
                            }
                        )
                    else:
                        logger.warn(f"image logging not implemented for {tracker.name}")
                
                # save images to file for future use in scoring
                save_layouts(test_dataset, test_batch, 0, global_step, "test_results", "cond_layout")
                # Log the original validation image
                log_validation_images_separate(test_batch["pixel_values"], 0, global_step, "test_results", "test_image")
                
                # Log image generated by original stable diffusion pipeline
                log_validation_images_separate(convert_pil_list_to_tensor(text_conditioned_sd_pipeline_images), 0, global_step, "test_results", "text_conditioned_sd_pipeline_images")
                
                # Log image generatedwith layout and text condition
                log_validation_images_separate(text_layout_conditioned_images, 0, global_step, "test_results", "text_layout_conditioned_images")

                # Log image generated with text condition only
                log_validation_images_separate(text_only_conditioned_images, 0, global_step, "test_results", "text_only_conditioned_images")

                # Log image generated with layout condition only
                log_validation_images_separate(layout_only_conditioned_images, 0, global_step, "test_results", "layout_only_conditioned_images")

                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        # columns = ["Test Image Id", "Test Image", "No. Objects" ,"Layout", "Caption", "Generated Image"]
                        
                        text_only_conditioned_images_log_table_data = []
                        for i in range(len(text_only_conditioned_images)):
                            text_only_conditioned_images_log_table_data.append([
                                    test_batch["raw_data"][i]["id"],
                                    wandb.Image(test_batch["pixel_values"][i]),
                                    test_batch["raw_data"][i]["num_objects"],
                                    wandb.Image(get_layout_image_for_bbox(test_dataset, test_batch["bounding_boxes"][i])),
                                    ','.join(test_batch["raw_data"][i]["captions"]),
                                    wandb.Image(text_only_conditioned_images[i])                
                            ])
                        text_only_conditioned_images_log_table = wandb.Table(columns=log_table_columns, data=text_only_conditioned_images_log_table_data)
                        tracker.log({"text_only_conditioned_images_log_table": text_only_conditioned_images_log_table})


                        text_layout_conditioned_images_log_table_data = []
                        for i in range(len(text_layout_conditioned_images)):
                            text_layout_conditioned_images_log_table_data.append([
                                    test_batch["raw_data"][i]["id"],
                                    wandb.Image(test_batch["pixel_values"][i]),
                                    test_batch["raw_data"][i]["num_objects"],
                                    wandb.Image(get_layout_image_for_bbox(test_dataset, test_batch["bounding_boxes"][i])),
                                    ','.join(test_batch["raw_data"][i]["captions"]),
                                    wandb.Image(text_layout_conditioned_images[i])
                            ])
                        text_layout_conditioned_images_log_table = wandb.Table(columns=log_table_columns, data=text_layout_conditioned_images_log_table_data)
                        tracker.log({"text_layout_conditioned_images_log_table": text_layout_conditioned_images_log_table})

                        layout_only_conditioned_images_log_table_data = []
                        for i in range(len(layout_only_conditioned_images)):
                            layout_only_conditioned_images_log_table_data.append([
                                    test_batch["raw_data"][i]["id"],
                                    wandb.Image(test_batch["pixel_values"][i]),
                                    test_batch["raw_data"][i]["num_objects"],
                                    wandb.Image(get_layout_image_for_bbox(test_dataset, test_batch["bounding_boxes"][i])),
                                    ','.join(test_batch["raw_data"][i]["captions"]),
                                    wandb.Image(layout_only_conditioned_images[i])
                            ])
                        layout_only_conditioned_images_log_table = wandb.Table(columns=log_table_columns, data=layout_only_conditioned_images_log_table_data)
                        tracker.log({"layout_only_conditioned_images_log_table": layout_only_conditioned_images_log_table})

                        
                        text_conditioned_sd_pipeline_log_table_data = [] 
                        for i in range(len(text_conditioned_sd_pipeline_images)):
                            text_conditioned_sd_pipeline_log_table_data.append([
                                test_batch["raw_data"][i]["id"],
                                wandb.Image(test_batch["pixel_values"][i]),
                                test_batch["raw_data"][i]["num_objects"],
                                wandb.Image(get_layout_image_for_bbox(test_dataset, test_batch["bounding_boxes"][i])),
                                ','.join(test_batch["raw_data"][i]["captions"]),
                                wandb.Image(text_conditioned_sd_pipeline_images[i])
                            ]) 
                        text_conditioned_sd_pipeline_log_table = wandb.Table(columns=log_table_columns, data = text_conditioned_sd_pipeline_log_table_data)
                        tracker.log({"text_conditioned_sd_pipeline_log_table": text_conditioned_sd_pipeline_log_table})
        
                if step >= args.max_test_steps:
                    break
        # KID score
        all_text_conditioned_sd_pipeline_kid_score_mean, all_text_conditioned_sd_pipeline_kid_score_std = calculate_kid_score(original_validation_images, all_text_onditioned_sd_pipeline_images)
        all_text_layout_conditioned_images_kid_score_mean, all_text_layout_conditioned_images_kid_score_std = calculate_kid_score(original_validation_images, all_text_layout_conditioned_images)
        all_layout_only_conditioned_images_kid_score_mean, all_layout_only_conditioned_images_kid_score_std = calculate_kid_score(original_validation_images, all_layout_only_conditioned_images)
        all_text_only_conditioned_images_kid_score_mean, all_text_only_conditioned_images_kid_score_std = calculate_kid_score(original_validation_images, all_text_only_conditioned_images)

        # FID Score           
        # calculate score across all generated images
        all_text_layout_conditioned_images_fid_score = calculate_fid_score(original_validation_images, all_text_layout_conditioned_images)
        all_text_conditioned_sd_pipeline_fid_score = calculate_fid_score(original_validation_images, all_text_onditioned_sd_pipeline_images)
        all_layout_only_conditioned_images_fid_score = calculate_fid_score(original_validation_images, all_layout_only_conditioned_images)
        all_text_only_conditioned_images_fid_score = calculate_fid_score(original_validation_images, all_text_only_conditioned_images)

        # Inception Score
        all_text_layout_conditioned_images_is_score_mean,  all_text_layout_conditioned_images_is_score_std = calculate_inception_score(all_text_layout_conditioned_images)
        all_text_conditioned_sd_pipeline_is_score_mean, all_text_conditioned_sd_pipeline_is_score_std = calculate_inception_score(all_text_onditioned_sd_pipeline_images)
        all_layout_only_conditioned_images_is_score_mean, all_layout_only_conditioned_images_is_std = calculate_inception_score(all_layout_only_conditioned_images)
        all_text_only_conditioned_images_is_score_mean, all_text_only_conditioned_images_is_score_std= calculate_inception_score(all_text_only_conditioned_images)

        # CLIP Score           
        # calculate score across all generated images
        all_text_conditioned_sd_pipeline_clip_score = calculate_clip_score(all_text_onditioned_sd_pipeline_images, prompts)
        all_text_layout_conditioned_images_clip_score = calculate_clip_score(all_text_layout_conditioned_images, prompts)
        all_layout_only_conditioned_images_clip_score = calculate_clip_score(all_layout_only_conditioned_images, prompts)
        all_text_only_conditioned_images_clip_score = calculate_clip_score(all_text_only_conditioned_images, prompts)
        
        # TODO - Precision and Recall

        # Wandb Table
        scores_columns = ["Name", "CLIP Score", "FID Score", "IS Score Mean", "IS Score Std", "KID Score Mean", "KID Score Std"]
        scores_table = wandb.Table(columns=scores_columns)
        scores_table.add_data("all_text_conditioned_sd_pipeline_clip_score", all_text_conditioned_sd_pipeline_clip_score, all_text_conditioned_sd_pipeline_fid_score, all_text_conditioned_sd_pipeline_is_score_mean, all_text_conditioned_sd_pipeline_is_score_std, all_text_conditioned_sd_pipeline_kid_score_mean, all_text_conditioned_sd_pipeline_kid_score_std)
        scores_table.add_data("all_text_layout_conditioned_images_clip_score",all_text_layout_conditioned_images_clip_score, all_text_layout_conditioned_images_fid_score, all_text_layout_conditioned_images_is_score_mean, all_text_layout_conditioned_images_is_score_std, all_text_layout_conditioned_images_kid_score_mean, all_text_layout_conditioned_images_kid_score_std)
        scores_table.add_data("all_layout_only_conditioned_images_clip_score",all_layout_only_conditioned_images_clip_score, all_layout_only_conditioned_images_fid_score, all_layout_only_conditioned_images_is_score_mean, all_layout_only_conditioned_images_is_std, all_layout_only_conditioned_images_kid_score_mean, all_layout_only_conditioned_images_kid_score_std)
        scores_table.add_data("all_text_only_conditioned_images_clip_score",all_text_only_conditioned_images_clip_score, all_text_only_conditioned_images_fid_score, all_text_only_conditioned_images_is_score_mean, all_text_only_conditioned_images_is_score_std, all_text_only_conditioned_images_kid_score_mean, all_text_only_conditioned_images_kid_score_std)
        
        for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log({"scores_table": scores_table})


        # Output the scores
        logger.info(f"Text conditioned SD pipeline CLIP score: {all_text_conditioned_sd_pipeline_clip_score}")
        logger.info(f"Text & layout conditioned CLIP score: {all_text_layout_conditioned_images_clip_score}")
        logger.info(f"Layout only conditioned CLIP score: {all_layout_only_conditioned_images_clip_score}")
        logger.info(f"Text only conditioned CLIP score: {all_text_only_conditioned_images_clip_score}")

        logger.info(f"Text conditioned SD pipeline FID score: {all_text_conditioned_sd_pipeline_fid_score}")
        logger.info(f"Text & layout conditioned FID score: {all_text_layout_conditioned_images_fid_score}")
        logger.info(f"Layout only conditioned FID score: {all_layout_only_conditioned_images_fid_score}")
        logger.info(f"Text only conditioned FID score: {all_text_only_conditioned_images_fid_score}")

        logger.info(f"Text conditioned SD pipeline IS score mean: {all_text_conditioned_sd_pipeline_is_score_mean}")
        logger.info(f"Text & layout conditioned IS score mean: {all_text_layout_conditioned_images_is_score_mean}")
        logger.info(f"Layout only conditioned IS score mean: {all_layout_only_conditioned_images_is_score_mean}")
        logger.info(f"Text only conditioned IS score mean: {all_text_only_conditioned_images_is_score_mean}")

        logger.info(f"Text conditioned SD pipeline IS score std: {all_text_conditioned_sd_pipeline_is_score_std}")
        logger.info(f"Text & layout conditioned IS score std: {all_text_layout_conditioned_images_is_score_std}")
        logger.info(f"Layout only conditioned IS score std: {all_layout_only_conditioned_images_is_std}")
        logger.info(f"Text only conditioned IS score std: {all_text_only_conditioned_images_is_score_std}")

        logger.info(f"Text conditioned SD pipeline KID score mean: {all_text_conditioned_sd_pipeline_kid_score_mean}")
        logger.info(f"Text & layout conditioned KID score mean: {all_text_layout_conditioned_images_kid_score_mean}")
        logger.info(f"Layout only conditioned KID score mean: {all_layout_only_conditioned_images_kid_score_mean}")
        logger.info(f"Text only conditioned KID score mean: {all_text_only_conditioned_images_kid_score_mean}")

        logger.info(f"Text conditioned SD pipeline KID score std: {all_text_conditioned_sd_pipeline_kid_score_std}")
        logger.info(f"Text & layout conditioned KID score std: {all_text_layout_conditioned_images_kid_score_std}")
        logger.info(f"Layout only conditioned KID score std: {all_layout_only_conditioned_images_kid_score_std}")
        logger.info(f"Text only conditioned KID score std: {all_text_only_conditioned_images_kid_score_std}")

        # Save the scores
        



#https://github.com/toshas/torch-fidelity
# https://github.com/justinpinkney/stable-diffusion/blob/main/scripts/autoencoder-eval.py
if __name__ == "__main__":
    test()
    # Find some captions to test with
    # or create a set of captions which demonstrate and compare with and without layout conditioning
    sample_captions = [ # taken from file:///Users/miav1/Downloads/s11042-021-11038-0.pdf
        "A person above a playingfield and left of another person left of grass, with a car left of a car above the grass.",
        "One broccoli left of another, which is inside vegetables and has a carrot below it.",
        "A person above the trees inside the sky, with a skateboard surrounded by sky.",
        "Three people with the first two inside a fence and the first left of the third."
    ]