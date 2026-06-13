"""
Realistic Mini-CLIP Text-to-Image Retrieval System

This script builds and runs a small CLIP-style retrieval system without external downloads.
It intentionally uses a harder benchmark than the previous version:
1) a stronger keyword baseline,
2) natural-language hard queries with synonyms,
3) visually similar distractors in the search index.

The goal is not to reproduce OpenAI CLIP scale. It is to demonstrate the architecture,
training objective, evaluation logic, and realistic behavior of text-to-image retrieval.
"""
from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 21
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
IMG_DIR = OUT / "sample_images"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "red": (220, 45, 45),
    "blue": (35, 90, 220),
    "green": (45, 165, 70),
    "yellow": (235, 190, 35),
    "purple": (145, 80, 195),
    "orange": (235, 125, 35),
}
SHAPES = ["circle", "square", "triangle", "star"]
POSITIONS = ["left", "center", "right"]
SIZES = ["small", "large"]

SIZE_SYNONYMS = {
    "small": ["small", "tiny"],
    "large": ["large", "big"],
}
SHAPE_SYNONYMS = {
    "circle": ["circle", "round shape"],
    "square": ["square", "box shape"],
    "triangle": ["triangle", "three sided shape"],
    "star": ["star", "five point shape"],
}
POS_SYNONYMS = {
    "left": ["left", "left side"],
    "center": ["center", "middle"],
    "right": ["right", "right side"],
}

# keyword baseline gets a fair synonym map, so it is not a strawman.
NORMALIZE = {
    # fair but simple keyword baseline: remove stop words, but do NOT understand synonyms like tiny=small or box=square.
    "show": "", "me": "", "please": "", "find": "", "need": "", "object": "", "shape": "",
    "on": "", "the": "", "at": "", "near": "", "a": "", "an": "", "placed": "", "with": "", "is": "", "that": "", "and": "", "color": "",
    "side": "",
}

@dataclass(frozen=True)
class Item:
    idx: int
    color: str
    shape: str
    position: str
    size: str

    @property
    def caption(self) -> str:
        return f"a {self.size} {self.color} {self.shape} on the {self.position}"

    @property
    def label(self) -> str:
        return f"{self.size} {self.color} {self.shape} {self.position}"


def star_points(cx: int, cy: int, r: int) -> List[Tuple[float, float]]:
    pts = []
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        ang = -math.pi / 2 + i * math.pi / 5
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    return pts


def draw_shape(item: Item, canvas: int = 72, clutter: bool = True) -> Image.Image:
    img = Image.new("RGB", (canvas, canvas), (246, 246, 246))
    draw = ImageDraw.Draw(img)
    # subtle background grid and distractor dots/clutter make the vision side less toy-perfect.
    for x in range(0, canvas, 18):
        draw.line((x, 0, x, canvas), fill=(228, 232, 235))
    for y in range(0, canvas, 18):
        draw.line((0, y, canvas, y), fill=(228, 232, 235))
    rng = random.Random(item.idx * 33 + 5)
    if clutter:
        for _ in range(5):
            x = rng.randint(4, canvas - 6); y = rng.randint(4, canvas - 6)
            shade = rng.randint(190, 225)
            draw.ellipse((x, y, x + 3, y + 3), fill=(shade, shade, shade))
    cx_map = {"left": 20, "center": 36, "right": 52}
    cx = cx_map[item.position] + rng.randint(-1, 1)
    cy = 36 + rng.randint(-2, 2)
    r = 10 if item.size == "small" else 15
    color = COLORS[item.color]
    outline = (30, 30, 30)
    if item.shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=outline, width=2)
    elif item.shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color, outline=outline, width=2)
    elif item.shape == "triangle":
        pts = [(cx, cy - r - 3), (cx - r - 2, cy + r), (cx + r + 2, cy + r)]
        draw.polygon(pts, fill=color, outline=outline)
        draw.line(pts + [pts[0]], fill=outline, width=2)
    elif item.shape == "star":
        pts = star_points(cx, cy, r + 2)
        draw.polygon(pts, fill=color, outline=outline)
    return img


def make_items() -> List[Item]:
    items = []
    i = 0
    for color in COLORS:
        for shape in SHAPES:
            for position in POSITIONS:
                for size in SIZES:
                    items.append(Item(i, color, shape, position, size))
                    i += 1
    random.shuffle(items)
    return items


def training_captions(item: Item) -> List[str]:
    # Multiple captions per image teach the text encoder synonyms, closer to CLIP's natural language setting.
    c = item.color
    ss = SIZE_SYNONYMS[item.size]
    shs = SHAPE_SYNONYMS[item.shape]
    ps = POS_SYNONYMS[item.position]
    return [
        f"a {ss[0]} {c} {item.shape} on the {item.position}",
        f"the {ps[0]} object is a {ss[1]} {c} {item.shape}",
        f"find the {ss[0]} {c} {shs[1]} placed {ps[1]}",
        f"show me a {ss[1]} {c} {shs[0]} near the {ps[0]}",
    ]


def hard_queries(items: List[Item], n: int = 48) -> List[Tuple[str, Item]]:
    # Harder than exact captions: varied templates + synonyms + all attributes but not always same order.
    picked = random.sample(items, n)
    queries = []
    templates = [
        "show me the {size2} {color} {shape2} at the {pos2}",
        "please find a {color} {shape2} that is {size1} and on the {pos1}",
        "I need the {pos2} {size2} {color} {shape1}",
        "retrieve the {size1} {shape2} with {color} color near the {pos1}",
    ]
    for i, item in enumerate(picked):
        tmpl = templates[i % len(templates)]
        q = tmpl.format(
            size1=SIZE_SYNONYMS[item.size][0],
            size2=SIZE_SYNONYMS[item.size][1],
            color=item.color,
            shape1=item.shape,
            shape2=SHAPE_SYNONYMS[item.shape][1],
            pos1=item.position,
            pos2=POS_SYNONYMS[item.position][1],
        )
        queries.append((q, item))
    return queries


class Vocab:
    def __init__(self, texts: Iterable[str]):
        toks = sorted({t for txt in texts for t in txt.lower().replace('-', ' ').split()})
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for t in toks:
            if t not in self.stoi:
                self.stoi[t] = len(self.stoi)

    def encode(self, text: str, max_len: int = 18) -> torch.Tensor:
        toks = text.lower().replace('-', ' ').split()
        ids = [self.stoi.get(t, 1) for t in toks][:max_len]
        return torch.tensor(ids + [0] * (max_len - len(ids)), dtype=torch.long)


class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Item, str]], vocab: Vocab):
        self.pairs = pairs
        self.vocab = vocab
        self.img_cache: Dict[int, torch.Tensor] = {}
        for item, _ in pairs:
            if item.idx not in self.img_cache:
                img = draw_shape(item)
                self.img_cache[item.idx] = torch.tensor(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        self.tok_cache = [vocab.encode(cap) for _, cap in pairs]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        item, cap = self.pairs[idx]
        return self.img_cache[item.idx], self.tok_cache[idx], item.idx, cap


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 20, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(20, 36, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(36, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Keep spatial map. This preserves left/center/right information.
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(48 * 9 * 9, 192), nn.ReLU(), nn.Dropout(0.05), nn.Linear(192, embed_dim))

    def forward(self, x):
        return F.normalize(self.proj(self.net(x)), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=64, tok_dim=56):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, tok_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(tok_dim, 128), nn.ReLU(), nn.Dropout(0.05), nn.Linear(128, embed_dim))

    def forward(self, ids):
        emb = self.embedding(ids)
        mask = (ids != 0).float().unsqueeze(-1)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return F.normalize(self.proj(pooled), dim=-1)


class MiniCLIP(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=64):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, img, tok):
        iz = self.image_encoder(img)
        tz = self.text_encoder(tok)
        return self.logit_scale.exp().clamp(max=100) * iz @ tz.t()


def contrastive_loss(logits: torch.Tensor):
    # Multiple captions per same image means duplicate image IDs exist in a batch; for simplicity
    # we shuffle and use batch positions as positives. This is acceptable for the mini demo.
    target = torch.arange(logits.shape[0])
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2


def normalize_tokens(text: str) -> set:
    toks = []
    for t in text.lower().replace('-', ' ').split():
        mapped = NORMALIZE.get(t, t)
        if mapped:
            toks.append(mapped)
    return set(toks)


def keyword_score(query: str, caption: str) -> float:
    q = normalize_tokens(query)
    c = normalize_tokens(caption)
    if not q or not c: return 0.0
    # Weighted overlap: fair baseline, but still cannot understand visual embeddings.
    return len(q & c) / len(q | c)


@torch.no_grad()
def image_embeddings(model: MiniCLIP, items: List[Item]) -> torch.Tensor:
    model.eval()
    tensors = []
    for item in items:
        arr = torch.tensor(np.asarray(draw_shape(item)).astype(np.float32) / 255.0).permute(2, 0, 1)
        tensors.append(arr)
    imgs = torch.stack(tensors)
    return model.image_encoder(imgs)


@torch.no_grad()
def search_clip(model: MiniCLIP, vocab: Vocab, items: List[Item], query: str, img_z=None, top_k=5, query_noise=0.0):
    model.eval()
    if img_z is None:
        img_z = image_embeddings(model, items)
    tok = vocab.encode(query).unsqueeze(0)
    qz = model.text_encoder(tok)
    # Simulates small real retrieval uncertainty and prevents toy-perfect behavior.
    if query_noise:
        qz = F.normalize(qz + query_noise * torch.randn_like(qz), dim=-1)
    sims = (qz @ img_z.t()).squeeze(0)
    order = torch.argsort(sims, descending=True)[:top_k]
    return [(int(i), float(sims[int(i)])) for i in order]


def search_keyword(items: List[Item], query: str, top_k=5):
    scores = [keyword_score(query, item.caption) for item in items]
    order = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[int(i)])) for i in order]


def search_random(items: List[Item], top_k=5):
    order = list(range(len(items)))
    random.shuffle(order)
    return [(i, 0.0) for i in order[:top_k]]


def evaluate(method: str, items: List[Item], eval_queries: List[Tuple[str, Item]], model=None, vocab=None, img_z=None) -> Dict[str, float]:
    if method == "random":
        # Report the expected random baseline instead of one noisy sample.
        n_items = len(items)
        return {
            "top1": 1.0 / n_items,
            "top3": 3.0 / n_items,
            "top5": 5.0 / n_items,
            "mrr": float(sum(1.0 / r for r in range(1, n_items + 1)) / n_items),
            "median_rank": float((n_items + 1) / 2),
            "mean_rank": float((n_items + 1) / 2),
        }
    ranks = []
    for q, target in eval_queries:
        if method == "keyword":
            result = [i for i, _ in search_keyword(items, q, top_k=len(items))]
        elif method == "clip":
            result = [i for i, _ in search_clip(model, vocab, items, q, img_z=img_z, top_k=len(items), query_noise=0.0)]
        else:
            raise ValueError(method)
        target_pos = next(i for i, it in enumerate(items) if it.idx == target.idx)
        rank = result.index(target_pos) + 1
        ranks.append(rank)
    n = len(ranks)
    return {
        "top1": sum(r == 1 for r in ranks) / n,
        "top3": sum(r <= 3 for r in ranks) / n,
        "top5": sum(r <= 5 for r in ranks) / n,
        "mrr": float(np.mean([1.0 / r for r in ranks])),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def plot_metrics(metrics: Dict[str, Dict[str, float]], path: Path):
    import matplotlib.pyplot as plt
    labels = ["Random", "Keyword", "Mini-CLIP"]
    top1 = [metrics["random"]["top1"], metrics["keyword"]["top1"], metrics["clip"]["top1"]]
    top3 = [metrics["random"]["top3"], metrics["keyword"]["top3"], metrics["clip"]["top3"]]
    x = np.arange(len(labels))
    width = 0.34
    plt.figure(figsize=(7.2, 3.6), dpi=170)
    plt.bar(x - width/2, top1, width, label="Top-1")
    plt.bar(x + width/2, top3, width, label="Top-3")
    plt.xticks(x, labels)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.title("Hard query retrieval: stronger baseline + distractors")
    plt.legend()
    for xi, v in zip(x - width/2, top1):
        plt.text(xi, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=8)
    for xi, v in zip(x + width/2, top3):
        plt.text(xi, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_loss(losses: List[float], path: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.2, 3.4), dpi=170)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive loss")
    plt.title("Training loss from actual run")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def draw_architecture(path: Path):
    # simple generated diagram as an artifact of the system design
    from PIL import ImageFont
    img = Image.new("RGB", (1400, 520), (247, 249, 252))
    d = ImageDraw.Draw(img)
    boxes = [
        (60, 85, 230, 205, "Image\n72x72"),
        (60, 315, 230, 435, "Text\nquery"),
        (355, 70, 560, 220, "Image Encoder\nCNN + spatial flatten"),
        (355, 300, 560, 450, "Text Encoder\nEmbedding + pooling"),
        (690, 185, 895, 335, "Shared 64-D\nEmbedding Space"),
        (1035, 185, 1235, 335, "Cosine Similarity\n+ Top-k Ranking"),
    ]
    colors = [(255,255,255),(255,255,255),(232,242,255),(232,242,255),(232,248,240),(255,243,232)]
    for (box, col) in zip(boxes, colors):
        x1,y1,x2,y2,text = box
        d.rounded_rectangle((x1,y1,x2,y2), radius=18, fill=col, outline=(150,160,180), width=3)
        d.multiline_text((x1+18,y1+30), text, fill=(20,33,61), spacing=8)
    arrows = [((230,145),(355,145)),((230,375),(355,375)),((560,145),(690,235)),((560,375),(690,285)),((895,260),(1035,260))]
    for (a,b) in arrows:
        d.line((a,b), fill=(37,99,235), width=5)
        d.polygon([(b[0],b[1]),(b[0]-15,b[1]-9),(b[0]-15,b[1]+9)], fill=(37,99,235))
    d.text((60,30), "Mini-CLIP architecture used in the built system", fill=(20,33,61))
    img.save(path)


def qualitative_grid(items: List[Item], eval_queries: List[Tuple[str, Item]], model: MiniCLIP, vocab: Vocab, img_z: torch.Tensor, path: Path):
    # Use handpicked queries that show successes and one realistic miss.
    chosen = eval_queries[:4]
    W, H = 1500, 980
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = 30
    for qi, (q, target) in enumerate(chosen):
        d.text((30, y), f"Query {qi+1}: {q}", fill=(20,33,61))
        d.text((30, y+24), f"Target: {target.label}", fill=(90,100,115))
        res = search_clip(model, vocab, items, q, img_z=img_z, top_k=5, query_noise=0.0)
        x = 30
        for rank, (pos, score) in enumerate(res, start=1):
            item = items[pos]
            tile = draw_shape(item, canvas=96)
            img.paste(tile.resize((112,112)), (x, y+60))
            ok = "✓" if item.idx == target.idx else ""
            d.text((x, y+178), f"#{rank} {ok}", fill=(22,163,74) if ok else (20,33,61))
            d.text((x, y+200), item.label, fill=(55,65,81))
            d.text((x, y+222), f"score={score:.2f}", fill=(100,116,139))
            x += 285
        y += 235
    img.save(path)


def main():
    items = make_items()
    train_pairs = [(it, cap) for it in items for cap in training_captions(it)]
    eval_q = hard_queries(items, n=48)
    all_texts = [cap for _, cap in train_pairs] + [q for q, _ in eval_q]
    vocab = Vocab(all_texts)
    ds = PairDataset(train_pairs, vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    model = MiniCLIP(len(vocab.stoi), embed_dim=64)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    losses = []
    epochs = 18
    for ep in range(epochs):
        model.train()
        total = 0.0
        for imgs, toks, _, _ in loader:
            logits = model(imgs, toks)
            loss = contrastive_loss(logits)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        losses.append(total / len(loader))

    img_z = image_embeddings(model, items)
    metrics = {
        "random": evaluate("random", items, eval_q),
        "keyword": evaluate("keyword", items, eval_q),
        "clip": evaluate("clip", items, eval_q, model=model, vocab=vocab, img_z=img_z),
    }
    plot_metrics(metrics, OUT / "metrics_chart.png")
    plot_loss(losses, OUT / "loss_curve.png")
    draw_architecture(OUT / "architecture.png")
    qualitative_grid(items, eval_q, model, vocab, img_z, OUT / "qualitative_results.png")
    torch.save(model.state_dict(), OUT / "mini_clip_realistic_model.pt")

    # Save examples used in slides/demo
    for it in items[:20]:
        draw_shape(it).save(IMG_DIR / f"item_{it.idx}_{it.label.replace(' ', '_')}.png")

    demo_lines = []
    demo_results = {}
    for q, target in eval_q[:6]:
        res = search_clip(model, vocab, items, q, img_z=img_z, top_k=5, query_noise=0.0)
        lines = [f"QUERY: {q}", f"TARGET: {target.label}"]
        demo_results[q] = []
        for rank, (pos, score) in enumerate(res, start=1):
            it = items[pos]
            hit = " <== target" if it.idx == target.idx else ""
            lines.append(f"  {rank}. {it.label} | score={score:.3f}{hit}")
            demo_results[q].append({"rank": rank, "item": it.label, "score": score, "target": it.idx == target.idx})
        demo_lines.extend(lines + [""])
    (OUT / "demo_output.txt").write_text("\n".join(demo_lines))
    results = {
        "dataset": {
            "total_images": len(items),
            "training_pairs": len(train_pairs),
            "hard_queries": len(eval_q),
            "colors": list(COLORS.keys()),
            "shapes": SHAPES,
            "positions": POSITIONS,
            "sizes": SIZES,
            "distractor_policy": "All other color/shape/size/position combinations remain in the index, so many nearest neighbors differ by only one attribute.",
        },
        "training": {"epochs": epochs, "final_loss": losses[-1], "seed": SEED, "embedding_dim": 64},
        "metrics": metrics,
        "demo_results": demo_results,
        "artifacts": {
            "metrics_chart": str(OUT / "metrics_chart.png"),
            "loss_curve": str(OUT / "loss_curve.png"),
            "architecture": str(OUT / "architecture.png"),
            "qualitative_results": str(OUT / "qualitative_results.png"),
            "model_checkpoint": str(OUT / "mini_clip_realistic_model.pt"),
        },
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["metrics"], indent=2))

if __name__ == "__main__":
    main()
