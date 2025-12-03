import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM

# Load CLIP for perception (image+text embedding).
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Small reasoning LM as cognition component (causal LM).
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
reasoner = AutoModelForCausalLM.from_pretrained("distilgpt2")

def perceive_and_act(image_path, instruction):
    # Perception: encode image and instruction.
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=instruction, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_emb = clip_model.get_image_features(inputs["pixel_values"])
        text_emb = clip_model.get_text_features(inputs["input_ids"])
    # Fusion: simple concatenation and projection.
    fused = torch.cat([image_emb, text_emb], dim=-1)
    fused = fused / fused.norm(dim=-1, keepdim=True)  # normalize
    
    # Cognition: condition a small LM on the fused embedding (prepended tokens).
    # Project fused embedding to token embeddings size (simple linear layer emulated).
    proj = torch.nn.Linear(fused.size(-1), reasoner.config.n_embd).to(fused.device)
    prefix = proj(fused)
    # Convert prefix to tokens via greedy decode of a prompt constructed from instruction.
    prompt = instruction + " -> action:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # Prepend noised prefix via appended text; here we concatenate tokens only.
    outputs = reasoner.generate(input_ids, max_length=input_ids.size(1)+10, do_sample=False)
    action = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return action

# Example call (replace with real image path).
# print(perceive_and_act("scene.png", "pick up the red cube"))