from transformers import ViTForImageClassification
import torch
import os
from PIL import Image
from diffusers import StableDiffusionPipeline
import argparse
import pandas as pd
import functools
import lpips

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--rarity_model_path", type=str)
    parser.add_argument("--sft_path", type=str)
    parser.add_argument("--dpok_paths", type=list)
    parser.add_argument("--dataset_dir", type=str, help="CSV dataset path to take prompts")
    

    return parser.parse_args()

def rank_imgs(rarity_model, perp_loss, gen_imgs, real_img):
    
    for g in gen_imgs:
        model_name = g.key
        g_image = g.value
        p_loss = perp_loss(real_img, g_image)
        r_reward = rarity_model(g_image)
        print(f"p_loss: {p_loss}")
        print(f"r_reward: {r_reward}")
            

def generate_images(pipes:dict, prompt:str, rank_imgs, args):
    gen_imgs = []
        
    for name, pipe in pipes.items():
        save_dir = os.path.join(args.output_dir, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_count = len(os.listdir(save_dir))
        img = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        img.save(os.path.join(save_dir, f"image_{img_count}.png"))
        gen_imgs.append({name:img})
    
    return gen_imgs


import torch.nn as nn

def _load_models(args):
    rarity_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    rarity_model.classifier = nn.Linear(rarity_model.config.hidden_size, 1)
    rarity_model.load_state_dict(torch.load(args['rarity_model_pth']))
    rarity_model.to("cuda")
    rarity_model.eval()
    perp_loss = lpips.LPIPS(net="alex")
    pipes = {}
    sft_pipe = StableDiffusionPipeline(path=args.sft_path)
    pipes['sft'] = sft_pipe
    for i, dpok in enumerate(args.dpok_paths): # probably 3 dpok path
        dpok_model = sft_pipe.load_lora_weights(dpok)
        pipes[f'dpok_{i}'] = dpok_model
    
    return rarity_model, perp_loss, pipes
    
if __name__ == "__main__":
    args = parse_args()
    rarity_model, perp_model, pipes = _load_models(args)
    rank_imgs_f = functools.partial(
        rank_imgs,
        rarity_model,
        perp_model
    )
    dataset = pd.read_csv(args.dataset_dir)
    for idx in range(len(dataset)):
        prompt = dataset['text'][idx]
        real_img_path = dataset['file_name'][idx]
        real_img = Image.open(real_img_path)
        generated_imgs = generate_images(pipes=pipes, prompt=prompt, rank_imgs=rank_imgs_f, args=args)
        rank_imgs_f(generated_imgs, real_img)