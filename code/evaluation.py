import os
import ssl
import csv
import torch
import lpips
import argparse

import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
from pytorch_msssim import ms_ssim

ssl._create_default_https_context = ssl._create_unverified_context

def load_image(clean_dir, poison_dir):
    clean_images, poison_images = [], [] 
    for image_name in os.listdir(clean_dir):
        if not image_name.endswith(".png"):
            continue
            
        clean_image_path = f"{clean_dir}/{image_name}"
        poison_image_path = f"{poison_dir}/{image_name}"
        if not os.path.exists(poison_image_path):
            continue
        clean_images.append(Image.open(clean_image_path).convert("RGB"))
        poison_images.append(Image.open(poison_image_path).convert("RGB"))

    return clean_images, poison_images


def calculate_fid(clean_images, poison_images):
    transform_fid = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    clean_tensors = torch.stack([transform_fid(image) for image in clean_images]).to("cuda")
    poison_tensors = torch.stack([transform_fid(image) for image in poison_images]).to("cuda")
    
    fid = FrechetInceptionDistance(normalize=True).to("cuda")
    fid.update(clean_tensors, real=True)
    fid.update(poison_tensors, real=False)
    return round(fid.compute().item(), 3)


def calculate_lpips(clean_images, poison_images):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    model = lpips.LPIPS(net='vgg').to("cuda")
    model.eval()

    scores = []
    for clean, poison in zip(clean_images, poison_images):
        img1 = transform(clean).unsqueeze(0).to("cuda")
        img2 = transform(poison).unsqueeze(0).to("cuda")

        with torch.no_grad():
            score = model(img1, img2).item()
            scores.append(score)

    return round(sum(scores) / len(scores), 3)

def calculate_ms_ssim(clean_images, poison_images, data_range=2.0):
    transform_ms_ssim = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    clean_tensors = torch.stack([transform_ms_ssim(image) for image in clean_images]).to("cuda")
    poison_tensors = torch.stack([transform_ms_ssim(image) for image in poison_images]).to("cuda")
    
    ms_ssim_value = ms_ssim(clean_tensors, poison_tensors, data_range=data_range, size_average=True)
    return round(ms_ssim_value.item(), 3)


def calculate_clip_sim(clip_dir, clean_images, poison_images):
    model = CLIPModel.from_pretrained(clip_dir).to("cuda")
    processor = CLIPProcessor.from_pretrained(clip_dir)
    similarities = []
    
    for i, image in enumerate(clean_images):
        inputs = processor(images=[clean_images[i], poison_images[i]], return_tensors="pt").to("cuda")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            clean_features = image_features[0].unsqueeze(0)
            poison_features = image_features[1].unsqueeze(0)
            sim = F.cosine_similarity(clean_features, poison_features)
            similarities.append(sim)

    sim_average = torch.stack(similarities).mean()
    return round(sim_average.item(), 3)

def main(args):
    csv_list = [["Artist\Metrics", "LPIPS", "FID", "MS-SSIM", "CLIP-SIM"],]
    lpips_sum, fid_sum, ms_ssim_sum, clip_sim_sum, artist_num = 0, 0, 0, 0, 0
    generated_dir = f"{args.generated_dir}/{args.finetuning_method}"
    for artist in os.listdir(generated_dir):
        
        if artist.startswith('.'):
            continue  
        
        clean_dir = f"{generated_dir}/{artist}/clean"
        poison_dir = f"{generated_dir}/{artist}/{args.experments_name}"
        
        if not os.path.exists(clean_dir) or not os.path.exists(poison_dir):
            continue  
    
        clean_images, poison_images = load_image(clean_dir, poison_dir)  
        
        if len(clean_images) == 0 or len(poison_images) == 0:
            continue

        artist_num += 1
        lpips_score = calculate_lpips(clean_images, poison_images)
        lpips_sum += lpips_score
        fid_score = calculate_fid(clean_images, poison_images)
        fid_sum += fid_score
        ms_ssim_score = calculate_ms_ssim(clean_images, poison_images)
        ms_ssim_sum += ms_ssim_score
        clip_sim = calculate_clip_sim(args.clip_dir, clean_images, poison_images)
        clip_sim_sum += clip_sim
        csv_list.append([artist, lpips_score, fid_score, ms_ssim_score, clip_sim])

    lpips_average = round(lpips_sum / artist_num, 3)
    fid_average = round(fid_sum / artist_num, 3)
    ms_ssim_average = round(ms_ssim_sum / artist_num, 3)
    clip_sim_average = round(clip_sim_sum / artist_num, 3)
    csv_list.append(["average", lpips_average, fid_average, ms_ssim_average, clip_sim_average])
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/{args.experments_name}.csv", mode="w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_list) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--clip_dir', type=str, default='')
    parser.add_argument('--generated_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--experments_name', type=str, default='') 
    parser.add_argument('--finetuning_method', type=str, default='') 
    args = parser.parse_args()
    main(args)

