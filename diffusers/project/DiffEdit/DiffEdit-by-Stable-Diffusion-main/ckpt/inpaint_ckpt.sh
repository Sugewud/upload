mkdir inpainting-v1-2
cd inpainting-v1-2
mkdir feature_extractor
wget -P ./feature_extractor https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/feature_extractor/preprocessor_config.json

mkdir scheduler
wget -P ./scheduler https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/scheduler/scheduler_config.json

mkdir text_encoder
wget -P ./text_encoder https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/text_encoder/config.json
wget -P ./text_encoder https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/text_encoder/pytorch_model.bin

mkdir tokenizer
wget -P ./tokenizer https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/tokenizer/merges.txt
wget -P ./tokenizer https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/tokenizer/special_tokens_map.json
wget -P ./tokenizer https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/tokenizer/tokenizer_config.json
wget -P ./tokenizer https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/tokenizer/vocab.json

mkdir unet
wget -P ./unet https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/unet/config.json
wget -P ./unet https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/unet/diffusion_pytorch_model.bin

mkdir vae
wget -P ./vae https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/vae/config.json
wget -P ./vae https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/vae/diffusion_pytorch_model.bin

wget https://hf-mirror.com/stabilityai/stable-diffusion-2-inpainting/resolve/main/model_index.json
