from transformers import ViTForImageClassification
import torch
from diffusers import StableDiffusionPipeline
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_sd", type=str)
    parser.add_argument("--path_vit", type=str)

pipe = StableDiffusionPipeline.from_pretrained("/home/emir/Desktop/dev/datasets/weights/SFT").to("cuda")
df = pd.read_csv("/home/emir/Desktop/dev/datasets/ImageReward/metadata_for_rankings.csv")

pipe(prompt=df['text'][1000], num_inference_steps=50, guidance_scale=7.5).images[0].save("./image.png")