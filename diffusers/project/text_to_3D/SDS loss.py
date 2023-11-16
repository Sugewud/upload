import torch
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler
from PIL import Image
from tqdm.auto import tqdm

#设置参数
prompt = ["a photograph of an astronaut riding a horse"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 1000          # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance
generator = torch.manual_seed(1)    # Seed generator to create the inital latent noise
batch_size = 1

#载入stable diffusion的各个组件 并放到gpu上
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

vae = vae.to("cuda")
text_encoder = text_encoder.to("cuda")
unet = unet.to("cuda") 

#将文本进行编码 包括有文本的和无文本的 
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
  text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
  uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0]   
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

#生成输入的随机噪声
latents = torch.randn(
  (batch_size, unet.in_channels, height // 8, width // 8),
  generator=generator,
)
latents = latents.to("cuda")

#设置scheduler
scheduler.set_timesteps(num_inference_steps)
latents = latents * scheduler.init_noise_sigma

#使用SDS loss生成图片
iterations = 500
for nstep in tqdm(range(iterations)):
    t = torch.randint(0, num_inference_steps, (1,), device="cuda").long()
    alpha_t= scheduler.alphas[t].to("cuda")
    x = latents                            #生成器为恒等映射 生成一张图片
    eps = torch.randn_like(x).to("cuda")   #采样一个和图片维度相同的噪声
    z_t = scheduler.add_noise(x, eps, t)   #扩散模型的前向过程

    # 这一部分实现noise_pred = unet(z_t, y, t) 预测噪声 利用当前隐变量，文本条件信息，时间步t
    latent_model_input = torch.cat([z_t] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    noise_pred = noise_pred.to("cuda")

    grad = alpha_t * (noise_pred - eps) #计算梯度
    latents = latents - grad     #更新

#使用vae的解码器将去噪好的latent解码为图片
latents = 1 / 0.18215 * latents
with torch.no_grad():
  image = vae.decode(latents).sample

#显示图片
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype("uint8")
pil_images = Image.fromarray(image[0])
pil_images.save("/root/project/diffusers/result/result.jpg")



