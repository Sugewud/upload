import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from typing import  List, Optional, Union

class DiffEdit_v1(StableDiffusionImg2ImgPipeline):
    @torch.no_grad()
    def get_esitmate(
            self,
            prompt: Union[str, List[str]] = None,
            image: Union[torch.FloatTensor, Image.Image] = None,
            strength: float = 0.8,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            eta: Optional[float] = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        # Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )
        # Preprocess image
        image = self.image_processor.preprocess(image)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(1)

        # Prepare latent variables
        latents = self.prepare_latents(image, latent_timestep, 1, 1, prompt_embeds.dtype, device, generator)

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        # size: 1x3xhxw
        return image 

    @torch.no_grad()
    def get_mask(
        self,
        latents_num: int = 10,
        refer_prompt: Union[str, List[str]] = None,
        query_prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, Image.Image] = None,
        strength: float = 0.5,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        seed: int = 2625
    ):
        diff_list = []
        # cycle for n(latents_num) times
        for index in range(latents_num):
            # get reference noise latents
            refer_image = self.get_esitmate(
                prompt=refer_prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed * index)
            )
            # get query noise latents 
            query_image = self.get_esitmate(
                prompt=query_prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed * index)
            )
            diff_list.append(refer_image - query_image)
        # Creating a mask placeholder
        tensor_mask = torch.zeros_like(diff_list[0])

        # Taking an average of 10 iterations
        for index in range(latents_num):
            tensor_mask += torch.abs(diff_list[index])
        tensor_mask = tensor_mask/latents_num

        # Averaging multiple channels
        tensor_mask = tensor_mask.squeeze(0).mean(0)

        # Normalize
        tensor_mask = (tensor_mask - tensor_mask.min()) / (tensor_mask.max() - tensor_mask.min())

        # Binarizing and returning the mask object
        pil_mask = (tensor_mask.cpu().numpy() > 0.5).astype(np.uint8)*255
        
        return pil_mask

