import os
import random
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
import re
import base64
from io import BytesIO

from transformers import ( 
    CLIPProcessor, 
    CLIPModel,
    AutoModel,
    AutoModelForVision2Seq,
    AutoProcessor
) 

# FT_CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# FT_CLIP_MODEL = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
FT_CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
BASE_CLIP_MODEL = "openai/clip-vit-base-patch32"
SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32 

logger = logging.getLogger(__name__)

_MODELS = {
    "ft_clip": None,
    "base_clip": None,
    "smolvlm": None
}

_TEXT_EMBED_CACHE = {}
_FT_IMAGE_EMBED_CACHE = {}
_BASE_IMAGE_EMBED_CACHE = {}


def ensure_pil_image(img):
    """Converts numpy arrays, PIL images, or base64 strings into an RGB PIL Image."""
    if isinstance(img, str):
        try:
            img_bytes = base64.b64decode(img)
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise ValueError("Provided string could not be decoded as a base64 image.")

    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
        
    return img.convert("RGB")

def get_image_hash(pil_img):
    """Creates a fast, unique hash for an image to use as a cache key."""
    return hashlib.md5(pil_img.tobytes()).hexdigest()

def extract_core_goal(question: str) -> str:
    pattern_complex = r"exactly described as the '([^']+)'"
    pattern_simple = r"(?:[Cc]an you find|[Cc]ould you find) the ([^?]+)"

    match_complex = re.search(pattern_complex, question)
    if match_complex:
        return match_complex.group(1).strip().rstrip('.')

    match_simple = re.search(pattern_simple, question)
    if match_simple:
        return match_simple.group(1).strip().rstrip('?')

    return question.strip()


def get_clip_instance(model_type="ft"):
    global _MODELS
    key = f"{model_type}_clip"
    if _MODELS[key] is not None:
        return _MODELS[key]

    model_id = FT_CLIP_MODEL if model_type == "ft" else BASE_CLIP_MODEL
    logger.info(f"Loading {model_id} on {DEVICE}...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=DTYPE, use_safetensors=True).to(DEVICE)
    model.eval()
    
    _MODELS[key] = (processor, model)
    return _MODELS[key]


def get_smolvlm_instance():
    global _MODELS
    if _MODELS["smolvlm"] is not None:
        return _MODELS["smolvlm"]

    logger.info(f"Loading SmolVLM model on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL)
    model = AutoModelForVision2Seq.from_pretrained(
        SMOLVLM_MODEL, 
        torch_dtype=DTYPE, 
        use_safetensors=True
    ).to(DEVICE)
    model.eval()
    
    _MODELS["smolvlm"] = (processor, model)
    return _MODELS["smolvlm"]


@torch.no_grad()
def get_cached_text_embedding(text, model_type="ft"):
    cache_key = f"{model_type}_{text}"
    if cache_key in _TEXT_EMBED_CACHE:
        return _TEXT_EMBED_CACHE[cache_key]

    processor, model = get_clip_instance(model_type)
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    embed = model.get_text_features(**inputs)
    embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
    
    _TEXT_EMBED_CACHE[cache_key] = embed
    return embed

@torch.no_grad()
def get_cached_image_embedding(pil_img, model_type="ft"):
    """Helper to cache and retrieve image embeddings universally."""
    img_hash = get_image_hash(pil_img)
    cache = _FT_IMAGE_EMBED_CACHE if model_type == "ft" else _BASE_IMAGE_EMBED_CACHE
    
    if img_hash in cache:
        return cache[img_hash]
        
    processor, model = get_clip_instance(model_type)
    inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE, DTYPE)
    embed = model.get_image_features(**inputs)
    embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
    
    cache[img_hash] = embed
    return embed


@torch.no_grad()
def get_top_k_similar_objects(target_desc, candidate_labels, k=5, temperature=1.0, image_goal=None):
    if not candidate_labels:
        return []

    if image_goal is not None:
        target_embed = get_cached_image_embedding(ensure_pil_image(image_goal), model_type="base")
    else:
        goal = extract_core_goal(target_desc)
        prompt = f"An indoor scene containing a {goal}."
        target_embed = get_cached_text_embedding(prompt, model_type="base") 
    
    candidate_embeds = [get_cached_text_embedding(label, model_type="base") for label in candidate_labels]
    candidate_embeds = torch.cat(candidate_embeds, dim=0) 
    
    logits = torch.matmul(target_embed, candidate_embeds.T).squeeze(0) 
    probs = F.softmax(logits / temperature, dim=0)
    
    k = min(k, len(candidate_labels))
    scores, indices = torch.topk(probs, k=k)
    
    results = []
    for score, idx in zip(scores, indices):
        results.append({
            "label": candidate_labels[idx.item()],
            "score": score.item(),
            "index": idx.item()
        })
        
    return results

@torch.no_grad()
def rank_images(goal, images: list, model_type="ft") -> list:
    if not images:
        return []

    processor, model = get_clip_instance(model_type)
    
    if isinstance(goal, Image.Image):
        target_embed = get_cached_image_embedding(goal, model_type)
    else:
        prompt = goal
        if model_type == "ft":
            prompt = f"An indoor scene containing a {goal}."
        elif model_type == "base":
            prompt = f"Image of a {goal}."
        target_embed = get_cached_text_embedding(prompt, model_type)
    
    image_embeds = [get_cached_image_embedding(img, model_type) for img in images]
    image_embeds = torch.cat(image_embeds, dim=0)
    
    scores = (target_embed @ image_embeds.T).squeeze(0)
    
    logit_scale = model.logit_scale.exp()
    scores = scores * logit_scale
    
    if scores.dim() == 0:
        scores = scores.unsqueeze(0)
        
    results = [{"index": i, "score": score.item()} for i, score in enumerate(scores)]
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results

@torch.no_grad()
def ask_smolvlm_if_present(goal, image) -> bool:
    """SmolVLM VQA capable of handling both Text goals and Image goals."""
    processor, model = get_smolvlm_instance()
    
    if isinstance(goal, Image.Image):
        question = "Is the target object shown in the first image present in the second image? Answer yes or no."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Target object:"},
                    {"type": "image"},
                    {"type": "text", "text": "Scene:"},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        input_images = [goal, image]
        log_q = "Image-to-Image Match"
    else:
        question = f"Is there an object described as '{goal}' in this image? Answer yes or no."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        input_images = [image]
        log_q = question
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=input_images, return_tensors="pt")
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

    generated_ids = model.generate(**inputs, max_new_tokens=10)
    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip().lower()
    
    logger.info(f"SmolVLM Question: '{log_q}' | Answer: '{answer}'")
    return "yes" in answer


def get_frontier(
    question, 
    frontier_images, 
    snapshot_images, 
    snapshot_crops, 
    vqa_override_margin=6.0,
    image_goal=None
):
    if image_goal is not None:
        active_goal = ensure_pil_image(image_goal)
        logger_goal_str = "Image Goal Provided"
    else:
        active_goal = extract_core_goal(question)
        logger_goal_str = active_goal

    logger.info(f"Goal: {logger_goal_str} | Frontiers: {len(frontier_images)} | Snapshots: {len(snapshot_images)}")

    frontiers_pil = [ensure_pil_image(img) for img in frontier_images]
    snapshots_pil_tuples = [(idx, ensure_pil_image(img)) for idx, img in snapshot_images]

    best_frontier_idx = None
    best_frontier_score = None
    if frontiers_pil:
        frontier_rankings = rank_images(active_goal, frontiers_pil, model_type="ft")
        best_frontier_idx = frontier_rankings[0]["index"]
        best_frontier_score = frontier_rankings[0]["score"]
        logger.info(f"Top Frontier Score: {best_frontier_score:.4f}")

    best_snap_list_idx = None
    best_snap_img = None
    best_snap_score = None
    if snapshots_pil_tuples:
        just_snap_imgs = [img for _, img in snapshots_pil_tuples]
        snapshot_rankings = rank_images(active_goal, just_snap_imgs, model_type="ft")
        
        best_local_idx = snapshot_rankings[0]["index"]
        best_snap_list_idx = best_local_idx
        best_snap_img = snapshots_pil_tuples[best_local_idx][1]
        best_snap_score = snapshot_rankings[0]["score"]
        logger.info(f"Top Snapshot Score: {best_snap_score:.4f}")

    choose_snapshot = False
    if best_snap_img is not None:
        choose_snapshot = ask_smolvlm_if_present(active_goal, best_snap_img)

    if not choose_snapshot and best_snap_score is not None and best_frontier_score is not None:
        score_diff = best_snap_score - best_frontier_score
        if score_diff >= vqa_override_margin:
            logger.info(f"SMOLVLM OVERRIDE: Snapshot score is higher than Frontier by {score_diff:.4f}. Trusting FT-CLIP.")
            choose_snapshot = True

    if choose_snapshot:
        original_snap_idx = snapshots_pil_tuples[best_snap_list_idx][0]
        crops = snapshot_crops.get(original_snap_idx, [])
        
        if not crops:
            return "snapshot", original_snap_idx, -1
            
        crops_pil = [ensure_pil_image(c) for c in crops]
        crop_rankings = rank_images(active_goal, crops_pil, model_type="base")
        
        return "snapshot", original_snap_idx, crop_rankings[0]["index"]

    else:
        logger.info("Evaluating fallbacks (No valid snapshot selected).")
        
        if best_frontier_idx is not None:
            return "frontier", best_frontier_idx, None
            
        elif best_snap_list_idx is not None:
            logger.warning("No frontiers available! Forcing fallback to highest-ranked snapshot.")
            original_snap_idx = snapshots_pil_tuples[best_snap_list_idx][0]

            crops = snapshot_crops.get(original_snap_idx, [])
        
            if not crops:
                return "snapshot", original_snap_idx, -1
                
            crops_pil = [ensure_pil_image(c) for c in crops]
            crop_rankings = rank_images(active_goal, crops_pil, model_type="base")
            
            return "snapshot", original_snap_idx, crop_rankings[0]["index"]
            
        else:
            logger.error("Dead end: No valid frontiers AND no snapshots available.")
            return "empty", None, None