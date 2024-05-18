from ImageReward import utils
import argparse
import numpy as np
import pandas as pd
import os
from natsort import natsorted

def parse_arguments():
    
    args = argparse.ArgumentParser()
    args.add_argument("--csv_dir", type=str)
    args.add_argument("--dataset_path", type=str)
    args.add_argument("--indexes_path", type=str)
    
    return args.parse_args()

def _eval(aes, cls, bls, args):
    indexes = np.load(args.indexes_path)
    df = pd.read_csv(args.csv_dir)
    imgs = natsorted(os.listdir(args.dataset_path))
    aes_list = []
    bls_list = []
    cls_list = []
    for i, idx in enumerate(indexes):
        prompt = df.iloc[idx]['text']
        image_path = os.path.join(args.dataset_path, imgs[i])
        # print(prompt)
        # print(image_path)
        aes_val = aes.score(prompt, image_path)
        bls_val = bls.score(prompt, image_path)
        cls_val = cls.score(prompt, image_path)
        aes_list.append(aes_val)
        bls_list.append(bls_val)
        cls_list.append(cls_val)
        print(f"Aesthetic Score: {aes_val}, \n BLIP Score: {bls_val}, \n CLIP Score: {cls_val}")
    print(f"Aesthetic Score: {np.array(aes_list).mean()}, \n BLIP Score: {np.array(bls_val).mean()}, \n CLIP Score: {np.array(cls_val).mean()}")

if __name__ == "__main__":
    aesthetic_score = utils.load_score(name="Aesthetic")
    clip_score = utils.load_score()
    blip_score = utils.load_score(name="BLIP")
    args = parse_arguments()
    _eval(aesthetic_score, clip_score, blip_score, args)


