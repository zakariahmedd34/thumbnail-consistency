# Stage 1 — Thumbnail Consistency Pipeline
## Step-by-step with code, discussion points, and fast testing strategy

---

## What This Stage Does

**Input:** a video + its thumbnail  
**Output:** a score P ∈ [0, 1] — how representative the thumbnail is of the video content  

**Core idea:** Train encoders to align video frames with transcript text.
Then score the thumbnail as an outsider against that learned video representation.
The thumbnail never touches training. It is only evaluated.

---

## The Three Phases

```
Phase 0 — Fast proof of concept    (use CLIP frozen, no training, 1 day)
Phase 1 — Self-supervised training (train your own encoders, 1-2 weeks)
Phase 2 — Evaluation               (compare to manual labels)
```

Start with Phase 0. Only move to Phase 1 if Phase 0 shows a bimodal score distribution.

---

---

# PHASE 0 — Fast Proof of Concept Using Frozen CLIP

## Goal
Show that the consistency score idea works before training anything.
Use frozen CLIP as a proxy encoder. Compute scores on 50 videos.
Plot the distribution. Show qualitative examples to your professor.

## Why this is valid
CLIP is trained on image-text pairs and already understands semantic similarity.
It is not trained on frame-transcript pairs specifically, so it is not a perfect proxy —
but it is good enough to test whether the score distribution is meaningful.

## Data — fastest option

Use the **YouCook2** dataset. It has:
- 2000 YouTube cooking videos
- Temporal annotations with text descriptions per segment
- Public video IDs — you download only what you need

```bash
# Download YouCook2 annotations (no videos needed yet)
wget http://youcook2.eecs.umich.edu/static/YouCookII/youcookii_annotations_trainval.zip
unzip youcookii_annotations_trainval.zip
```

Or even faster — collect 50 YouTube videos manually using your existing notebook.
For Phase 0 you only need: thumbnail + transcript. No frame extraction needed.
Whisper on 50 short videos (3-5 min each) on Colab GPU takes ~30 minutes total.

## Install

```python
# In Colab
!pip install openai-clip torch torchvision
!pip install openai-whisper yt-dlp
```

## Step 0.1 — Compute consistency score with frozen CLIP

```python
import torch
import clip
from PIL import Image
import json
import os
import numpy as np

# Load frozen CLIP — no training
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def embed_image(image_path):
    """Embed a single image with CLIP."""
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize
    return embedding.squeeze().cpu().numpy()

def embed_text(text):
    """Embed a text string with CLIP."""
    tokens = clip.tokenize([text[:77]], truncate=True).to(device)  # CLIP max 77 tokens
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze().cpu().numpy()

def embed_transcript(transcript_path, max_segments=20):
    """
    Embed transcript by averaging embeddings of up to max_segments segments.
    This gives a video-level text representation.
    """
    with open(transcript_path) as f:
        segments = json.load(f)
    
    # Take evenly spaced segments to cover the full video
    if len(segments) > max_segments:
        indices = np.linspace(0, len(segments)-1, max_segments, dtype=int)
        segments = [segments[i] for i in indices]
    
    embeddings = []
    for seg in segments:
        text = seg["text"].strip()
        if len(text) > 3:  # skip empty or very short segments
            embeddings.append(embed_text(text))
    
    if not embeddings:
        return None
    
    # Average across segments → one video-level text vector
    return np.mean(embeddings, axis=0)

def consistency_score(thumbnail_path, transcript_path):
    """
    Compute how consistent the thumbnail is with the video transcript.
    Returns cosine similarity in [-1, 1]. Higher = more representative.
    """
    thumb_emb = embed_image(thumbnail_path)
    trans_emb = embed_transcript(transcript_path)
    
    if trans_emb is None:
        return None
    
    # Cosine similarity (both vectors are already normalized)
    score = float(np.dot(thumb_emb, trans_emb))
    return score
```

## Step 0.2 — Run on your collected videos

```python
RAW_DIR = "raw"  # or your Drive path
results = []

for video_id in os.listdir(RAW_DIR):
    folder = os.path.join(RAW_DIR, video_id)
    if not os.path.isdir(folder):
        continue
    
    thumb_path = os.path.join(folder, "thumbnail.jpg")
    transcript_path = os.path.join(folder, "transcript.json")
    meta_path = os.path.join(folder, "meta.json")
    
    if not (os.path.exists(thumb_path) and os.path.exists(transcript_path)):
        print(f"Skipping {video_id} — missing files")
        continue
    
    score = consistency_score(thumb_path, transcript_path)
    
    title = ""
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        title = meta.get("title", "")
    
    results.append({
        "video_id": video_id,
        "title": title,
        "score": score
    })
    print(f"{video_id}: {score:.3f}  |  {title[:60]}")

# Save results
with open("phase0_scores.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Step 0.3 — Plot the distribution

```python
import matplotlib.pyplot as plt

scores = [r["score"] for r in results if r["score"] is not None]

plt.figure(figsize=(10, 5))
plt.hist(scores, bins=20, edgecolor='black', color='steelblue', alpha=0.8)
plt.xlabel("Consistency Score (CLIP cosine similarity)", fontsize=13)
plt.ylabel("Number of Videos", fontsize=13)
plt.title("Thumbnail-Video Consistency Score Distribution", fontsize=14)
plt.axvline(x=np.mean(scores), color='red', linestyle='--', label=f"Mean: {np.mean(scores):.3f}")
plt.legend()
plt.tight_layout()
plt.savefig("score_distribution.png", dpi=150)
plt.show()

print(f"\nTotal videos: {len(scores)}")
print(f"Mean score: {np.mean(scores):.3f}")
print(f"Std: {np.std(scores):.3f}")
print(f"Min: {min(scores):.3f}  Max: {max(scores):.3f}")
```

## Step 0.4 — Qualitative inspection (most important for professor discussion)

```python
from IPython.display import display, Image as IPImage
import IPython.display as ipd

# Sort by score
sorted_results = sorted(results, key=lambda x: x["score"] if x["score"] else -1)

# Show bottom 5 (likely clickbait)
print("=" * 60)
print("LOWEST SCORES — likely clickbait or unrelated thumbnails")
print("=" * 60)
for r in sorted_results[:5]:
    print(f"\nScore: {r['score']:.3f}")
    print(f"Title: {r['title']}")
    thumb = os.path.join(RAW_DIR, r["video_id"], "thumbnail.jpg")
    if os.path.exists(thumb):
        display(IPImage(thumb, width=300))

# Show top 5 (likely representative)
print("\n" + "=" * 60)
print("HIGHEST SCORES — likely representative thumbnails")
print("=" * 60)
for r in sorted_results[-5:]:
    print(f"\nScore: {r['score']:.3f}")
    print(f"Title: {r['title']}")
    thumb = os.path.join(RAW_DIR, r["video_id"], "thumbnail.jpg")
    if os.path.exists(thumb):
        display(IPImage(thumb, width=300))
```

## What to show your professor

Show three things:
1. The score distribution plot — is it bimodal or flat?
2. The top 5 thumbnails — do they look representative?
3. The bottom 5 thumbnails — do they look like clickbait?

If the groups look qualitatively different, the idea is valid.
If they look the same, the transcript signal is too weak and you need to rethink.

**Expected outcome:** Scores will range from ~0.1 to ~0.4 for CLIP on this task.
The key is not the absolute value but the relative ordering and group separation.

---

---

# PHASE 1 — Self-Supervised Training (your actual model)

## Goal
Replace frozen CLIP with your own trained encoders.
Train on (frame, transcript_segment) pairs from the same video.
The trained model should produce better-separated scores than CLIP.

## Why train your own instead of using CLIP

CLIP was trained on image-alt-text pairs from the internet.
Your task is frame-spoken-transcript alignment, which is a different distribution.
Fine-tuning on your specific domain should improve score separation.
This fine-tuning is your model contribution — what makes it a research project not just a CLIP wrapper.

## Architecture

```
Vision encoder:   ViT-B/16  (pretrained, partially frozen)
Text encoder:     BERT-base  (pretrained, partially frozen)
Projection head:  Linear(768, 512) + LayerNorm  for each tower
Loss:             InfoNCE (contrastive)
```

Both encoders share the same projection dimensionality (512).
The projection head is trained from scratch.
The encoders are fine-tuned with a low learning rate (1e-5).

## Step 1.1 — Build the training dataset from aligned_pairs.json

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json, os, random
import torchvision.transforms as T
from transformers import BertTokenizer

class FrameTranscriptDataset(Dataset):
    """
    Each item is one (frame, transcript_text) pair from the same video.
    Positive: same video, temporally aligned.
    Negatives: handled inside the InfoNCE loss (cross-batch).
    """
    
    def __init__(self, raw_dir, max_text_len=128, image_size=224):
        self.samples = []
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_text_len = max_text_len
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        for video_id in os.listdir(raw_dir):
            folder = os.path.join(raw_dir, video_id)
            pairs_path = os.path.join(folder, "aligned_pairs.json")
            frames_dir = os.path.join(folder, "frames")
            
            if not os.path.exists(pairs_path):
                continue
            
            with open(pairs_path) as f:
                pairs = json.load(f)
            
            for pair in pairs:
                text = pair.get("transcript_text", "").strip()
                frame_file = pair.get("frame_file", "")
                frame_path = os.path.join(frames_dir, frame_file)
                
                # Skip pairs with no speech or missing frame
                if len(text) < 5 or not os.path.exists(frame_path):
                    continue
                
                self.samples.append({
                    "frame_path": frame_path,
                    "text": text,
                    "video_id": video_id
                })
        
        print(f"Dataset: {len(self.samples)} valid frame-transcript pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and transform frame
        image = Image.open(sample["frame_path"]).convert("RGB")
        image = self.transform(image)
        
        # Tokenize transcript text
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "video_id": sample["video_id"]
        }
```

## Step 1.2 — Define the model

```python
import torch.nn as nn
from transformers import BertModel
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights

class TwoTowerModel(nn.Module):
    """
    Vision tower: ViT-B/16
    Text tower:   BERT-base
    Both project to d_model=512 shared space.
    """
    
    def __init__(self, d_model=512, freeze_encoders=True):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vision_encoder.heads = nn.Identity()  # remove classification head
        # ViT-B/16 output dim = 768
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        # BERT output dim = 768
        
        # Freeze encoders initially — only train projection
        if freeze_encoders:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            # Unfreeze last 2 transformer blocks of ViT
            for param in self.vision_encoder.encoder.layers[-2:].parameters():
                param.requires_grad = True
            # Unfreeze last 2 layers of BERT
            for param in self.text_encoder.encoder.layer[-2:].parameters():
                param.requires_grad = True
        
        # Projection heads — always trained
        self.vision_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.LayerNorm(d_model)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Learnable temperature for InfoNCE
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
    
    def encode_image(self, images):
        """images: [B, 3, 224, 224] → [B, 512] normalized"""
        features = self.vision_encoder(images)
        projected = self.vision_proj(features)
        return nn.functional.normalize(projected, dim=-1)
    
    def encode_text(self, input_ids, attention_mask):
        """tokens → [B, 512] normalized"""
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]  # [CLS] token
        projected = self.text_proj(cls_token)
        return nn.functional.normalize(projected, dim=-1)
    
    def forward(self, images, input_ids, attention_mask):
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(input_ids, attention_mask)
        return image_emb, text_emb
```

## Step 1.3 — InfoNCE loss

```python
def infonce_loss(image_emb, text_emb, logit_scale):
    """
    InfoNCE contrastive loss.
    Diagonal of similarity matrix = positive pairs (same video, same timestamp).
    Off-diagonal = negatives (different videos in the batch).
    
    image_emb: [B, 512]
    text_emb:  [B, 512]
    """
    logit_scale = logit_scale.exp().clamp(max=100)  # cap temperature
    
    # Similarity matrix [B, B]
    logits_i2t = logit_scale * image_emb @ text_emb.T
    logits_t2i = logit_scale * text_emb @ image_emb.T
    
    # Labels: diagonal is positive
    labels = torch.arange(len(image_emb), device=image_emb.device)
    
    loss_i2t = nn.functional.cross_entropy(logits_i2t, labels)
    loss_t2i = nn.functional.cross_entropy(logits_t2i, labels)
    
    return (loss_i2t + loss_t2i) / 2
```

## Step 1.4 — Training loop

```python
def train(raw_dir, epochs=10, batch_size=32, lr=1e-4, lr_encoder=1e-5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Dataset
    dataset = FrameTranscriptDataset(raw_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)
    
    # Model
    model = TwoTowerModel(d_model=512, freeze_encoders=True).to(device)
    
    # Separate learning rates: encoders get smaller lr
    optimizer = torch.optim.AdamW([
        {"params": model.vision_proj.parameters(), "lr": lr},
        {"params": model.text_proj.parameters(), "lr": lr},
        {"params": model.logit_scale, "lr": lr},
        {"params": model.vision_encoder.encoder.layers[-2:].parameters(), "lr": lr_encoder},
        {"params": model.text_encoder.encoder.layer[-2:].parameters(), "lr": lr_encoder},
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            optimizer.zero_grad()
            image_emb, text_emb = model(images, input_ids, attention_mask)
            loss = infonce_loss(image_emb, text_emb, model.logit_scale)
            loss.backward()
            
            # Gradient clipping — important for transformer fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs}  |  Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "stage1_model.pt")
    print("Model saved to stage1_model.pt")
    return model
```

---

# PHASE 2 — Inference: Score Any Thumbnail

```python
def score_thumbnail(model, thumbnail_path, transcript_path,
                    device="cuda", max_segments=20):
    """
    Given a trained model, score how consistent the thumbnail is
    with the video content (as represented by the transcript).
    
    Returns: float in [-1, 1]. Higher = more representative.
    """
    from transformers import BertTokenizer
    import torchvision.transforms as T
    
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Embed thumbnail
    thumb = Image.open(thumbnail_path).convert("RGB")
    thumb_tensor = transform(thumb).unsqueeze(0).to(device)
    with torch.no_grad():
        thumb_emb = model.encode_image(thumb_tensor).squeeze().cpu().numpy()
    
    # Embed transcript (average over segments)
    with open(transcript_path) as f:
        segments = json.load(f)
    
    if len(segments) > max_segments:
        indices = np.linspace(0, len(segments)-1, max_segments, dtype=int)
        segments = [segments[i] for i in indices]
    
    text_embeddings = []
    for seg in segments:
        text = seg["text"].strip()
        if len(text) < 5:
            continue
        enc = tokenizer(text, max_length=128, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = model.encode_text(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device)
            ).squeeze().cpu().numpy()
        text_embeddings.append(emb)
    
    if not text_embeddings:
        return None
    
    transcript_emb = np.mean(text_embeddings, axis=0)
    
    # Cosine similarity (embeddings already normalized)
    score = float(np.dot(thumb_emb, transcript_emb))
    return score
```

---

# PHASE 3 — Evaluation Against Manual Labels

```python
def evaluate_against_labels(raw_dir, score_fn):
    """
    Compare model scores to manual annotation labels.
    Requires annotation.json with filled thumbnail_label fields.
    
    score_fn: function(thumbnail_path, transcript_path) -> float
    """
    from sklearn.metrics import roc_auc_score, classification_report
    
    y_true = []  # 1 = representative, 0 = clickbait
    y_score = []
    
    for video_id in os.listdir(raw_dir):
        folder = os.path.join(raw_dir, video_id)
        ann_path = os.path.join(folder, "annotation.json")
        thumb_path = os.path.join(folder, "thumbnail.jpg")
        transcript_path = os.path.join(folder, "transcript.json")
        
        if not os.path.exists(ann_path):
            continue
        
        with open(ann_path) as f:
            ann = json.load(f)
        
        label = ann.get("thumbnail_label")
        confidence = ann.get("thumbnail_label_confidence")
        
        # Only use high/medium confidence labels
        if label not in ["representative", "clickbait"]:
            continue
        if confidence == "low":
            continue
        
        score = score_fn(thumb_path, transcript_path)
        if score is None:
            continue
        
        y_true.append(1 if label == "representative" else 0)
        y_score.append(score)
    
    if len(y_true) < 10:
        print("Not enough labeled samples for evaluation")
        return
    
    # AUC — does not require a threshold
    auc = roc_auc_score(y_true, y_score)
    print(f"ROC-AUC: {auc:.3f}")
    print(f"(0.5 = random, 1.0 = perfect)")
    
    # At threshold = mean score
    threshold = np.mean(y_score)
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    print(f"\nClassification at threshold={threshold:.3f}:")
    print(classification_report(y_true, y_pred,
                                 target_names=["clickbait", "representative"]))
    
    return auc
```

---

# Discussion Points for Your Professor

## What is new compared to existing work

| Aspect | Existing papers | This project |
|---|---|---|
| Ground truth | Creator thumbnail assumed clean | Explicitly questions this assumption |
| Training signal | Thumbnail similarity (circular) | Frame-transcript alignment (causal) |
| Thumbnail role | Training target | Evaluated as outsider |
| Label requirement | Implicit (all thumbnails = positive) | None for training, small set for evaluation |
| Dataset scope | Yahoo/curated (clean by selection) | YouTube (noisy by design) |

## The key claim to defend

> "Frame-transcript temporal alignment is a valid proxy for video semantic content,
> and a thumbnail that scores high against this learned representation
> is more likely to be representative of the video than one that scores low."

This claim is testable. The ROC-AUC score against manual labels is the test.
If AUC > 0.65, the claim holds. If AUC ≈ 0.5, the signal is too weak.

## Honest limitations to acknowledge

1. A composited thumbnail that is semantically correct but visually unlike any frame
   will score incorrectly low — the model penalizes visual novelty even if content is accurate
2. Videos with little speech (music, ASMR, nature) produce weak transcript embeddings
3. CLIP as proxy (Phase 0) underestimates the trained model's potential

---

# Minimum Viable Experiment

To have something to show:

```
50 videos → Phase 0 CLIP scores → distribution plot + 5 high/5 low qualitative examples
```

This takes:
- Data collection: ~1 hour (50 short videos, existing notebook)
- Phase 0 scoring: ~10 minutes
- Plot + inspection: ~30 minutes

Total: ~2 hours to have a result you can discuss.

If the qualitative examples look meaningful → proceed to Phase 1 training.
If not → come back and rethink the transcript embedding strategy.
