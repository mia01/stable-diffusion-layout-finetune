import os
from dataset.helpers import get_layout_conditioning
from dataset.visual_genome_dataset import VisualGenomeTrain, VisualGenomeValidation
from embedders.layout_embedder import LayoutEmbedder, LayoutEmbedderConfig, LayoutEmbedderModel
from parse_args import parse_args
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
from validation.log_validation import log_validation
from validation.validation_step import validation_step

def main():
    args = parse_args()

    # set logging with accelerate
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
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
    # layout_embedder = LayoutEmbedder(device=accelerator.device)

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
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
        # Freeze vae and text_encoder and set unet to trainable
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.train()
        layout_embedder.train()

        # TODO - check if enable_xformers_memory_efficient_attention is needed

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    
                    for i, model in enumerate(models):
                        file_name = f"{model.__class__.__name__}"
                        model.save_pretrained(os.path.join(output_dir, file_name))

                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

            def load_model_hook(models, input_dir):

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    
                    if model.__class__.__name__ == "LayoutEmbedderModel":
                        load_model = LayoutEmbedderModel.from_pretrained(input_dir, subfolder=model.__class__.__name__)
                        # model.register_to_config(**load_model.config)
                    else:
                        load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder=model.__class__.__name__)
                        model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook) # runs before load_state

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Load the datasets
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
        train_dataset = VisualGenomeTrain(
            args.train_annotation_file,
            args.train_caption_file,
            args.image_folder,
            args.resolution
        )

        validation_dataset = VisualGenomeValidation(
            args.val_annotation_file,
            args.val_caption_file,
            args.image_folder,
            args.resolution
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

        # TODO - this section doesnt work - needs fixing
        if args.max_train_samples is not None:
            from torch.utils.data import DataLoader, SubsetRandomSampler

            K = args.max_train_samples
            subsample_train_indices = torch.randperm(len(train_dataset))[:K]
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=SubsetRandomSampler(subsample_train_indices)) 
        
        else: 
            # DataLoaders creation:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,
            )

        val_dataloader = torch.utils.data.DataLoader(
                validation_dataset,
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,
            )
        
        # Dataloader for logging image inference every interval
        val_image_dataloader = torch.utils.data.DataLoader(
                validation_dataset,
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=4,
                num_workers=args.dataloader_num_workers,
            )
        

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        logger.info("device")
        logger.info(accelerator.device)
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler, layout_embedder = accelerator.prepare(
            unet, optimizer, train_dataloader, val_dataloader,lr_scheduler, layout_embedder
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

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            tracker_config = dict(vars(args))
            tracker_config.pop("validation_prompts")
            accelerator.init_trackers(args.tracker_project_name, tracker_config)

        # Train!
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                # TODO - load unet and layout embedder correctly and also set epoch and step
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[2])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch

        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, args.num_train_epochs):
            train_loss = 0.0
            
            # TODO: Log Validation Images if first step in epoch (as baseline)
            if epoch == 0:
                logger.info(f"Epoch: {epoch} Logging Validation images")
                # log_validation(accelerator, val_image_dataloader, {"vae":vae,
                #         "text_encoder": text_encoder,
                #         "unet": unet, 
                #         "layout_embedder": layout_embedder,
                #         "noise_scheduler": noise_scheduler,
                #         "tokenizer": tokenizer
                #     }, epoch, global_step,args.seed, args.num_inference_steps, args.output_dir) # should I use args.num_inference_steps here?
            logger.info(f"Epoch: {epoch} starting the training steps")
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    logger.info(f"Epoch: {epoch} step: {step} Encoded image latents")
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if args.input_perturbation:
                        new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    # TODO - check what the number of noise_scheduler.config.num_train_timesteps is
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if args.input_perturbation:
                        noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    logger.info(f"Epoch: {epoch} step: {step} Added noise to latents")
                    
                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    logger.info(f"Epoch: {epoch} step: {step} Got text condition")

                    # Add layout embedding and concatenate with text embedding
                    batch["bounding_boxes"] = batch["bounding_boxes"].to(accelerator.device)
                    layout_conditioning = get_layout_conditioning(layout_embedder, batch["bounding_boxes"])
                    layout_conditioning.to(accelerator.device)
                    encoder_hidden_states.to(accelerator.device)
                    logger.info(f"Epoch: {epoch} step: {step} Got bbox condition")

                    # TODO experiment with and without concatenating with text
                    # Concatenate embeddings
                    condition = torch.cat((encoder_hidden_states, layout_conditioning), 1)

                    # Get the target for loss depending on the prediction type
                    if args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    logger.info(f"Epoch: {epoch} step: {step} Created noise schedule with prediction_type {noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = unet(noisy_latents, timesteps, condition).sample
                    
                    # if args.snr_gamma is None: # TODO: this is None by default check if should be removed
                    loss = functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    # else:
                        # by default snr_gamma is None - see if I should remove this
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        # snr = compute_snr(noise_scheduler, timesteps)
                        # if noise_scheduler.config.prediction_type == "v_prediction":
                        #     # Velocity objective requires that we add one to SNR values before we divide by them.
                        #     snr = snr + 1
                        # mse_loss_weights = (
                        #     torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        # )

                        # loss = functional.mse_loss(model_pred.float(), target.float(), reduction="none")
                        # loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        # loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    logger.info(f"Epoch: {epoch} step: {step} Backward propogation")

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % args.checkpointing_steps == 0:
                        logger.info(f"Epoch: {epoch} step: {step} Saving checkpoint")
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")

                            # TODO save the transformer for the layout data
                            # accelerator.register_for_checkpointing(layout_embedder)
                            accelerator.save_state(save_path)

                            logger.info(f"Epoch: {epoch} step: {step} Saved checkpoint state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

            if accelerator.is_main_process:
                if epoch % args.validation_epochs == 0 or step == args.validation_steps:
                        
                    logger.info(f"Epoch: {epoch} step: {step} Running validation loss")
                    validation_step(accelerator, val_dataloader, pipeline, epoch, global_step)

                    logger.info(f"Epoch: {epoch} step: {step} Running validation inference")
                    log_validation(accelerator, val_image_dataloader, {"vae":vae,
                        "text_encoder": text_encoder,
                        "unet": unet, 
                        "layout_embedder": layout_embedder,
                        "noise_scheduler": noise_scheduler,
                        "tokenizer": tokenizer
                    }, epoch, global_step,args.seed, args.num_inference_steps, args.output_dir) 
                    

        # Create the pipeline using the trained modules and save it.
        # TODO - amend to include the layout embedder - needs testing
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(unet)
            layout_embedder = accelerator.unwrap_model(layout_embedder)
            unet.save_pretrained(args.output_dir)
            layout_embedder.save_pretrained(args.output_dir)

            # Run a final round of inference.
            if args.validation_prompts is not None:
                logger.info("Running final inference for collecting generated images...")
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    revision=args.revision,
                    variant=args.variant,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.torch_dtype = weight_dtype
                pipeline.set_progress_bar_config(disable=False)

                log_validation(accelerator, val_image_dataloader, {"vae":vae,
                        "text_encoder": text_encoder,
                        "unet": unet, 
                        "layout_embedder": layout_embedder,
                        "noise_scheduler": noise_scheduler,
                        "tokenizer": tokenizer
                    }, epoch, global_step,args.seed, args.num_inference_steps, args.output_dir) 
                    

        accelerator.end_training()



if __name__ == "__main__":
    main()