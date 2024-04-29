from transformers import ViTForImageClassification
import torch
import os
from PIL import Image
from diffusers import StableDiffusionPipeline
import argparse
import pandas as pd
import functools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_sd", type=str)
    parser.add_argument("--path_vit", type=str)

    return parser.parse_args()

def rank_imgs(rarity_model, fidelity_model, gen_imgs, real_img):
    

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
    
    rank_imgs(gen_imgs)

def _load_models(args):
    pass

if __name__ == "__main__":
    args = parse_args()
    rarity_model, fid_model, pipes = _load_models(args)
    rank_imgs_f = functools.partial(
        rank_imgs,
        rarity_model,
        fid_model
    )
    dataset = pd.read_csv(args.dataset_csv)
    for idx in range(len(dataset)):
        prompt = dataset['text'][idx]
        real_img_path = dataset['file_name'][idx]
        real_img = Image.open(real_img_path)
        generate_images(pipes=pipes, prompt=prompt, rank_imgs=rank_imgs_f, args=args)