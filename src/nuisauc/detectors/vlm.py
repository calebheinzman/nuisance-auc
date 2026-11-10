"""Text-conditioned zero-shot detectors and frozen image encoders.

A zero-shot score is image-embedding similarity to the mean positive prompt
minus similarity to the mean negative prompt. Image and text embeddings are
cached under `nuisauc.paths.cache_dir()`:

    img__<vlm>__<corpus>__cap<cap>.npy   (n, d) L2-normalised image embeddings
    txt__<vlm>.npz                       the 96-prompt pool, keys <stain>__pos/neg
    txt_r21__<vlm>.npz                   the four-template prompts, keys <corpus>__pos/neg

Model weights are fetched through Hugging Face into `nuisauc.paths.models_dir()`
(HF_HOME is set to it unless already set; it must be set before transformers,
open_clip or timm is first imported in the process).
"""
from __future__ import annotations

import gc
import os

import numpy as np

from nuisauc.data.corpora import load_any
from nuisauc.paths import cache_dir, models_dir

HF_ID = {"plip": "vinid/plip", "quiltnet": "wisdomik/QuiltNet-B-32",
         "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"}
VLMS = ["biomedclip", "plip", "quiltnet"]
GATED_VLM = "biomedclip"

# ---------------------------------------------- four templates per class
TEMPLATE_ZN_POS = [
    "a photomicrograph of a Ziehl-Neelsen stained sputum smear containing red rod-shaped acid-fast bacilli",
    "microscopy of sputum with red acid-fast bacilli against a blue background",
    "Ziehl-Neelsen stain showing Mycobacterium tuberculosis bacilli",
    "a sputum smear positive for acid-fast bacilli"]
TEMPLATE_ZN_NEG = [
    "a photomicrograph of a Ziehl-Neelsen stained sputum smear with no bacteria",
    "microscopy of sputum showing only blue stained background material",
    "Ziehl-Neelsen stain with no acid-fast organisms present",
    "a sputum smear negative for acid-fast bacilli"]
TEMPLATE_HE_POS = [
    "a histopathology image of lymph node tissue containing metastatic breast carcinoma",
    "H&E stained lymph node with tumor cells",
    "microscopy of lymph node showing metastatic tumor",
    "a lymph node section positive for metastasis"]
TEMPLATE_HE_NEG = [
    "a histopathology image of normal lymph node tissue",
    "H&E stained lymph node without tumor cells",
    "microscopy of benign lymph node tissue",
    "a lymph node section negative for metastasis"]
TEMPLATE_GI_POS = [
    "a photomicrograph of a Giemsa stained red blood cell infected with a malaria parasite",
    "a blood smear showing a red blood cell containing Plasmodium",
    "microscopy of a parasitized erythrocyte",
    "a thin blood film positive for malaria parasites"]
TEMPLATE_GI_NEG = [
    "a photomicrograph of a normal uninfected red blood cell",
    "a blood smear showing healthy erythrocytes",
    "microscopy of an uninfected red blood cell",
    "a thin blood film negative for malaria parasites"]

PROMPTS = {
    "Kaggle__local_128": (TEMPLATE_ZN_POS, TEMPLATE_ZN_NEG),
    "AFB__internal_81": (TEMPLATE_ZN_POS, TEMPLATE_ZN_NEG),
    "Wellgen__local_128": (TEMPLATE_ZN_POS, TEMPLATE_ZN_NEG),
    "pcam": (TEMPLATE_HE_POS, TEMPLATE_HE_NEG),
    "malaria": (TEMPLATE_GI_POS, TEMPLATE_GI_NEG),
}
STAIN = {"Kaggle__local_128": "ZN", "AFB__internal_81": "ZN",
         "Wellgen__local_128": "ZN", "pcam": "H&E", "malaria": "Giemsa"}

# --------------------------------------------- the 96-prompt pool
# Four tiers of four, per class per stain: A expert, B plain clinical,
# C minimal, D naive descriptive. Fixed; never edited.
POOL_ZN_POS = [
    "a photomicrograph of a Ziehl-Neelsen stained sputum smear containing red rod-shaped acid-fast bacilli",
    "Ziehl-Neelsen stain showing Mycobacterium tuberculosis bacilli against a methylene blue counterstain",
    "carbol fuchsin positive rods consistent with acid-fast bacilli in a sputum specimen",
    "brightfield microscopy of an AFB smear with beaded red bacilli",
    "a sputum smear positive for acid-fast bacilli",
    "a tuberculosis smear with bacteria present",
    "a slide showing evidence of tuberculosis infection",
    "a positive TB microscopy result",
    "acid-fast bacilli",
    "red bacteria",
    "bacilli present",
    "TB positive",
    "a microscope image where several thin bright red rod shaped objects can be seen lying against a pale blue background",
    "a photograph taken through a microscope showing small red stick like shapes scattered across the field",
    "an image of a stained slide in which elongated crimson structures are visible among blue debris",
    "a magnified view of a sample containing narrow red filaments",
]
POOL_ZN_NEG = [
    "a photomicrograph of a Ziehl-Neelsen stained sputum smear with no acid-fast bacilli",
    "Ziehl-Neelsen stain showing only methylene blue counterstained background material",
    "carbol fuchsin negative field with no acid-fast organisms in a sputum specimen",
    "brightfield microscopy of an AFB smear with no bacilli",
    "a sputum smear negative for acid-fast bacilli",
    "a tuberculosis smear with no bacteria present",
    "a slide showing no evidence of tuberculosis infection",
    "a negative TB microscopy result",
    "no acid-fast bacilli",
    "no red bacteria",
    "no bacilli present",
    "TB negative",
    "a microscope image showing only a pale blue background with no red rod shaped objects",
    "a photograph taken through a microscope showing no red stick like shapes in the field",
    "an image of a stained slide containing only blue debris and no crimson structures",
    "a magnified view of a sample with no red filaments",
]
POOL_HE_POS = [
    "a histopathology image of lymph node tissue containing metastatic breast carcinoma",
    "H&E stained lymph node section with infiltrating tumour cell nests",
    "hematoxylin and eosin section showing metastatic carcinoma deposits in a lymph node",
    "a lymph node biopsy with malignant epithelial cells effacing the nodal architecture",
    "a lymph node section positive for metastasis",
    "a biopsy showing cancer has spread to the lymph node",
    "a tissue sample containing tumour",
    "a positive cancer pathology slide",
    "metastatic tumour",
    "cancer cells",
    "malignant tissue",
    "tumour present",
    "a microscope image of pink and purple tissue in which a crowded region of large irregular dark nuclei disrupts the surrounding pattern",
    "a magnified tissue photograph showing an abnormal cluster of enlarged cells with dense nuclei",
    "a stained tissue section where part of the field is occupied by disorganised cells that look different from the rest",
    "a view of tissue containing a patch of atypical crowded cells",
]
POOL_HE_NEG = [
    "a histopathology image of normal lymph node tissue with no metastatic carcinoma",
    "H&E stained lymph node section with intact follicular architecture and no tumour",
    "hematoxylin and eosin section of a reactive lymph node without carcinoma deposits",
    "a benign lymph node biopsy with preserved nodal architecture",
    "a lymph node section negative for metastasis",
    "a biopsy showing no cancer in the lymph node",
    "a tissue sample with no tumour",
    "a negative cancer pathology slide",
    "normal tissue",
    "no cancer cells",
    "benign tissue",
    "no tumour present",
    "a microscope image of pink and purple tissue with a uniform pattern of evenly sized small nuclei throughout",
    "a magnified tissue photograph showing regularly arranged cells of similar size",
    "a stained tissue section where the whole field has a consistent orderly appearance",
    "a view of healthy tissue with no atypical cells",
]
POOL_GI_POS = [
    "a photomicrograph of a Giemsa stained red blood cell infected with a Plasmodium parasite",
    "a thin blood film showing an erythrocyte containing a malaria trophozoite ring form",
    "Giemsa stained blood smear with intraerythrocytic plasmodial parasites",
    "a parasitized red blood cell with a visible chromatin dot and cytoplasm",
    "a blood smear positive for malaria parasites",
    "a red blood cell infected with malaria",
    "a blood cell containing a parasite",
    "a positive malaria test slide",
    "malaria parasite",
    "infected cell",
    "parasitized erythrocyte",
    "malaria positive",
    "a microscope image of a round pinkish cell containing a small dark purple spot inside it",
    "a magnified view of a single blood cell with a distinct dark inclusion",
    "a photograph of a red blood cell that has a stained speck within its interior",
    "a close up of a blood cell with a dark blue dot inside",
]
POOL_GI_NEG = [
    "a photomicrograph of a Giemsa stained uninfected red blood cell",
    "a thin blood film showing an erythrocyte with no malaria parasite",
    "Giemsa stained blood smear with no intraerythrocytic parasites",
    "an unparasitized red blood cell with uniform cytoplasm and no chromatin dot",
    "a blood smear negative for malaria parasites",
    "a red blood cell not infected with malaria",
    "a blood cell containing no parasite",
    "a negative malaria test slide",
    "no malaria parasite",
    "uninfected cell",
    "normal erythrocyte",
    "malaria negative",
    "a microscope image of a round pinkish cell with an even interior and no dark spot",
    "a magnified view of a single blood cell with no inclusions",
    "a photograph of a red blood cell with a plain uniform appearance",
    "a close up of a blood cell with nothing inside it",
]
POOL = {"ZN": (POOL_ZN_POS, POOL_ZN_NEG), "H&E": (POOL_HE_POS, POOL_HE_NEG),
        "Giemsa": (POOL_GI_POS, POOL_GI_NEG)}
TIER = ["A_expert"] * 4 + ["B_clinical"] * 4 + ["C_minimal"] * 4 + \
       ["D_naive"] * 4


def set_hf_home() -> None:
    """Point the Hugging Face cache at models_dir() unless HF_HOME is set."""
    os.environ.setdefault("HF_HOME", str(models_dir()))


class VLM:
    """Uniform image/text embedding over open_clip and HF CLIP checkpoints."""

    def __init__(self, name: str, device: str):
        set_hf_home()
        self.name, self.device = name, device
        if name == "biomedclip":
            import open_clip
            self.m, _, self.pp = open_clip.create_model_and_transforms(
                HF_ID[name])
            self.tk = open_clip.get_tokenizer(HF_ID[name])
            self.kind = "open_clip"
        else:
            from transformers import CLIPModel, CLIPProcessor
            self.m = CLIPModel.from_pretrained(HF_ID[name])
            self.proc = CLIPProcessor.from_pretrained(HF_ID[name])
            self.kind = "hf"
        self.m.eval().to(device)

    def close(self) -> None:
        """Move the model off the GPU before dropping the reference."""
        try:
            self.m.to("cpu")
        except Exception:
            pass
        self.m = None

    def text(self, prompts: list[str]) -> np.ndarray:
        """L2-normalised text embeddings, one row per prompt."""
        import torch
        with torch.no_grad():
            if self.kind == "open_clip":
                e = self.m.encode_text(self.tk(prompts).to(self.device))
            else:
                t = self.proc(text=prompts, return_tensors="pt", padding=True,
                              truncation=True).to(self.device)
                e = self.m.get_text_features(**t)
            return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()

    def images(self, items: list, batch: int = 64) -> np.ndarray:
        """L2-normalised image embeddings, one row per item."""
        import torch
        with torch.no_grad():
            out = []
            for i in range(0, len(items), batch):
                imgs = [it["image"] for it in items[i:i + batch]]
                if self.kind == "open_clip":
                    x = torch.stack([self.pp(im) for im in imgs]).to(self.device)
                    e = self.m.encode_image(x)
                else:
                    x = self.proc(images=imgs, return_tensors="pt").to(self.device)
                    e = self.m.get_image_features(**x)
                out.append(torch.nn.functional.normalize(
                    e.float(), dim=-1).cpu().numpy())
            return np.concatenate(out)


def embed_cache(vlm_name: str, corpus: str, cap: int,
                dev: str) -> tuple[np.ndarray, dict]:
    """Cached (n, d) L2-normalised image embeddings, plus the 96-prompt text pool."""
    fi = cache_dir() / f"img__{vlm_name}__{corpus}__cap{cap}.npy"
    ft = cache_dir() / f"txt__{vlm_name}.npz"
    need_img, need_txt = not fi.exists(), not ft.exists()
    if not (need_img or need_txt):
        return np.load(fi), dict(np.load(ft))

    import torch
    vlm = None
    try:
        vlm = VLM(vlm_name, dev)
        if need_txt:
            T = {}
            for st, (p, n) in POOL.items():
                T[f"{st}__pos"] = vlm.text(p)
                T[f"{st}__neg"] = vlm.text(n)
            ft.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ft, **T)
        if need_img:
            items, _y = load_any(corpus, cap)
            E = vlm.images(items).astype(np.float32)
            fi.parent.mkdir(parents=True, exist_ok=True)
            np.save(fi, E)
            del items
    except Exception as ex:
        if vlm is not None:
            vlm.close()
        gc.collect()
        torch.cuda.empty_cache()
        # A silently skipped cell would bias any count over cells: abort.
        raise RuntimeError(f"aborting: {vlm_name} failed on {corpus}: "
                           f"{type(ex).__name__} {str(ex)[:120]}") from ex
    vlm.close()
    del vlm
    gc.collect()
    torch.cuda.empty_cache()
    return np.load(fi), dict(np.load(ft))


def prompt_embeddings(vlm_name: str, corpus: str, dev: str) -> dict:
    """Cached four-template prompt embeddings for every corpus (keys <corpus>__pos/neg).

    `corpus` is unused; the cache holds every corpus's prompts.
    """
    f = cache_dir() / f"txt_r21__{vlm_name}.npz"
    if f.exists():
        return dict(np.load(f))
    import torch
    vlm = VLM(vlm_name, dev)
    T = {}
    for c, (p, n) in PROMPTS.items():
        T[f"{c}__pos"] = vlm.text(p)
        T[f"{c}__neg"] = vlm.text(n)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **T)
    vlm.close()
    del vlm
    gc.collect()
    torch.cuda.empty_cache()
    return dict(np.load(f))


def unit(v: np.ndarray) -> np.ndarray:
    """v / (||v|| + 1e-9)."""
    return v / (np.linalg.norm(v) + 1e-9)


def score(E: np.ndarray, Tp: np.ndarray, Tn: np.ndarray, ip, ineg) -> np.ndarray:
    """Ensemble score from the selected positive and negative prompt indices."""
    return E @ unit(Tp[ip].mean(0)) - E @ unit(Tn[ineg].mean(0))


def encode(kind: str, spec: str | None, items: list, device: str) -> np.ndarray:
    """Spatial tokens: "fm" foundation model (8x8 grid at 224 px) or "resnet"
    ResNet-18 layer3 at 160 px (spec = checkpoint path or None for ImageNet)."""
    set_hf_home()
    from nuisauc.detectors._backbones import extract_fm_spatial, resnet_tokens
    if kind == "fm":
        return extract_fm_spatial(spec, items, batch_size=64, device=device,
                                  out_grid=8, input_size=224,
                                  verbose=False).astype(np.float32)
    return resnet_tokens(spec, items, 160, 64, device)
