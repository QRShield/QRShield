import os
import json
import torch
import argparse
from diffusers import StableDiffusionPipeline

def main(args):
    # print(f"model dir:{args.model_dir}")
    test_prompts = []
    with open(args.test_metadata_path, 'r', encoding='utf-8') as file:
        for line in file:
            dictionary = json.loads(line)
            test_prompts.append(dictionary["text"])

    if args.model_type == 'full':
        pipe = StableDiffusionPipeline.from_pretrained(args.model_dir, torch_dtype=torch.float16)
    elif args.model_type == 'dreambooth':
        pipe = StableDiffusionPipeline.from_pretrained(args.pre_model_dir, torch_dtype=torch.float16)
        pipe.load_lora_weights(args.model_dir)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.pre_model_dir, torch_dtype=torch.float16)
        pipe.load_textual_inversion(args.model_dir)
        
    pipe = pipe.to(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    for _, text in enumerate(test_prompts):
        for i, seed in enumerate(args.seeds):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            if args.model_type not in ['full', 'dreambooth']:
                words = text.split()
                # words[-1] = f"<{words[-1]}>"
                # prompt = " ".join(words)
                prompt = " ".join(words[:-2]) + f" in the style of <{words[-1]}>"
            else:
                prompt = text
            
            generated_image = pipe(
                prompt=prompt, 
                num_inference_steps=args.diff_steps, 
                width=args.resolution, height=args.resolution
            ).images[0]
            
            generated_image.save(f"{args.output_dir}/{text}_{i+1}.png")
            if i == 0 and prompt == text: print("the same")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image generation')
    parser.add_argument('--pre_model_dir', type=str, default='')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--test_metadata_path', type=str, default='')
    parser.add_argument('--model_type', type=str, default='full')
    
    parser.add_argument('--seeds', nargs='+', type=int, default=[123456, 111111, 222222, 333333, 444444, 555555])
    parser.add_argument('--diff_steps', default=50, type=int, help='')
    parser.add_argument('--resolution', default=512, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, help='device used for training')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)


