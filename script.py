import os
import glob
import json
import argparse
import zipfile
import torch
import open_clip
import concurrent.futures
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 1. Helper functions

def load_arguments(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return [{'id': arg.find('id').text, 'claim': arg.find('claim').text} for arg in root.findall('argument')]

def load_caption_file(caption_path):
    image_id = os.path.basename(os.path.dirname(caption_path))
    with open(caption_path, 'r', encoding='utf-8') as f:
        caption = f.read().strip()
    return image_id, caption

def load_captions_parallel(images_folder):
    captions = {}
    caption_paths = glob.glob(os.path.join(images_folder, 'I*', 'I*', 'image-caption.txt'))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for image_id, caption in executor.map(load_caption_file, caption_paths):
            captions[image_id] = caption
    return captions

def embed_texts(model, tokenizer, texts, device, batch_size=512):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            embeddings = model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)

# 2. Main function

def main(input_dir, output_dir):
    arguments_xml_path = os.path.join(input_dir, 'arguments.xml')
    images_folder = os.path.join(input_dir, 'images')

    print("Loading arguments...")
    arguments = load_arguments(arguments_xml_path)
    print(f"Loaded {len(arguments)} arguments.")

    print("Loading captions...")
    captions = load_captions_parallel(images_folder)
    print(f"Loaded {len(captions)} captions.")

    print("Loading CLIP model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)

    print("Generating embeddings...")
    claim_texts = [arg['claim'] for arg in arguments]
    caption_texts = list(captions.values())

    claim_embeddings = embed_texts(model, tokenizer, claim_texts, device)
    caption_embeddings = embed_texts(model, tokenizer, caption_texts, device)

    caption_ids = list(captions.keys())

    print("Computing similarities and retrieving Top-10...")
    TOP_K = 10
    results = []
    for idx, arg in enumerate(arguments):
        claim_emb = claim_embeddings[idx].unsqueeze(0)
        sims = F.cosine_similarity(claim_emb, caption_embeddings)
        topk = torch.topk(sims, k=TOP_K)
        for rank, index in enumerate(topk.indices.tolist(), start=1):
            results.append({
                "argument_id": arg['id'],
                "method": "retrieval",
                "image_id": caption_ids[index],
                "rank": rank,
                "tag": "CEDNAV-UTB; CLIP_Baseline"
            })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'submission.jsonl')

    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    print(f"Process completed. {len(results)} predictions generated.")

# 3. Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Path to input directory")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
