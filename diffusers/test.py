from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')
# Initialize a prompt
prompt = "a dog wearing hat"
# Pass the prompt in the pipeline

pipe(prompt).images[0]