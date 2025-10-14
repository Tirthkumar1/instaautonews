#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, re, json, urllib.parse
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv

# =========================
# Config / constants
# =========================
load_dotenv()

SCRIPTS_DIR = Path("out/scripts")
OUT_DIR     = Path("out/final_single")
TIMEOUT     = 25
UA          = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# IG portrait
W, H = 1080, 1350

# Image picking
MIN_W, MIN_H = 800, 600
MAX_CANDIDATES = 14

# Typography (override with .env or put fonts into assets/)
FONT_BOLD_PATH = os.getenv("FONT_BOLD_PATH", "assets/Inter-Bold.ttf")
FONT_REG_PATH  = os.getenv("FONT_REG_PATH",  "assets/Inter-Regular.ttf")

# UI text
BADGE_TEXT = os.getenv("BADGE_TEXT", "WORLD NEWS")   # small pill at top-left
BRAND_NAME = os.getenv("BRAND_NAME", "")             # optional footer: @brand
LOGO_PATH = Path(os.getenv("LOGO_PATH", "assets/logo.png"))
LOGO_MAX_WIDTH = int(os.getenv("LOGO_MAX_WIDTH", "320"))
LOGO_MAX_HEIGHT = int(os.getenv("LOGO_MAX_HEIGHT", "160"))

# Domain-specific badge palettes (bg RGBA, text RGB)
BADGE_PALETTES = {
    "bloomberg": ((114, 35, 142, 235), (255, 255, 255)),
    "reuters": ((245, 133, 16, 235), (20, 20, 20)),
    "cnn": ((189, 0, 0, 235), (255, 255, 255)),
    "bbc": ((28, 28, 28, 235), (255, 255, 255)),
    "techcrunch": ((0, 186, 94, 235), (0, 0, 0)),
    "coindesk": ((247, 173, 43, 235), (22, 22, 22)),
    "forbes": ((18, 18, 18, 235), (255, 255, 255)),
    "wsj": ((15, 30, 35, 235), (255, 255, 255)),
    "financialtimes": ((244, 214, 182, 235), (25, 25, 25)),
    "economist": ((204, 0, 0, 235), (255, 255, 255)),
}

def badge_colors_for_domain(domain: str):
    key = (domain or "").lower()
    for match, colors in BADGE_PALETTES.items():
        if match in key:
            bg_color, text_color = colors
            if len(bg_color) == 3:
                bg_color = (*bg_color, 235)
            return bg_color, text_color
    return (255, 255, 255, 235), (25, 25, 25)


_LOGO_CACHE: Optional[Image.Image] = None
_LOGO_FAILED = False

def load_logo() -> Optional[Image.Image]:
    global _LOGO_CACHE, _LOGO_FAILED
    if _LOGO_FAILED:
        return None
    if _LOGO_CACHE is not None:
        return _LOGO_CACHE
    path = LOGO_PATH
    if not path.exists():
        _LOGO_FAILED = True
        return None
    try:
        logo = Image.open(path).convert("RGBA")
        w, h = logo.size
        scale = min(LOGO_MAX_WIDTH / max(1, w), LOGO_MAX_HEIGHT / max(1, h), 1.0)
        if scale < 1.0:
            logo = logo.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        _LOGO_CACHE = logo
        return _LOGO_CACHE
    except Exception as exc:
        print(f"[logo] load failed -> {exc}")
        _LOGO_FAILED = True
        return None


# =========================
# Tiny utils
# =========================
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_font(path: str, size: int):
    try: return ImageFont.truetype(path, size)
    except Exception: return ImageFont.load_default()

def join_url(base: str, url: str) -> str:
    try: return urllib.parse.urljoin(base, url)
    except Exception: return url

def http_get(url: str, referer: str = "") -> Optional[requests.Response]:
    try:
        headers = {"User-Agent": UA}
        if referer: headers["Referer"] = referer
        r = requests.get(url, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        return r
    except Exception:
        return None

def download_bytes(url: str) -> Optional[bytes]:
    r = http_get(url, referer=url)
    return r.content if r else None

def open_image(raw: bytes) -> Optional[Image.Image]:
    try:
        img = Image.open(io.BytesIO(raw)); img.load(); return img
    except Exception:
        return None

def domain_of(url: str) -> str:
    try: return urllib.parse.urlparse(url).netloc
    except: return ""

# =========================
# Find candidates (article + reddit)
# =========================
def parse_srcset(srcset: str, base: str) -> List[str]:
    out=[]
    for part in re.split(r"\s*,\s*", srcset or ""):
        m = re.match(r"(.+?)\s+\d+w", part.strip())
        url = (m.group(1) if m else part.strip().split()[0]) if part.strip() else ""
        if url: out.append(join_url(base, url))
    return out

def image_candidates_from_html(html: str, page_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []

    # OG/Twitter
    for prop in ["og:image","og:image:url","og:image:secure_url"]:
        for tag in soup.find_all("meta", property=prop):
            c = tag.get("content")
            if c: urls.append(join_url(page_url, c))
    for name in ["twitter:image","twitter:image:src"]:
        for tag in soup.find_all("meta", attrs={"name":name}):
            c = tag.get("content")
            if c: urls.append(join_url(page_url, c))

    # link rel=image_src
    for link in soup.find_all("link", rel=True):
        rel = " ".join(link.get("rel") or [])
        if "image_src" in rel and link.get("href"):
            urls.append(join_url(page_url, link["href"]))

    # <img> + srcset
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if src: urls.append(join_url(page_url, src))
        if img.get("srcset"): urls += parse_srcset(img["srcset"], page_url)

    # rank: prefer share/hero/large & actual image extensions
    def rank(u: str) -> int:
        ul=u.lower(); score=0
        for kw in ["og","opengraph","share","social","hero","lead","large","twitter","card"]:
            if kw in ul: score += 5
        for ext in [".jpg",".jpeg",".png",".webp"]:
            if ul.endswith(ext): score += 2
        return -score

    seen=set(); ordered=[]
    for u in sorted(urls, key=rank):
        if u not in seen:
            seen.add(u); ordered.append(u)
    return ordered[:25]

def reddit_image_candidates(query: str) -> List[str]:
    # Unauthed JSON search (rate-limited, but fine for light use)
    q = urllib.parse.quote_plus(query or "")
    if not q: return []
    url = f"https://www.reddit.com/search.json?q={q}&sort=top&t=week&limit=10"
    r = http_get(url)
    if not r: return []
    try:
        data = r.json()
    except Exception:
        return []
    out=[]
    for ch in data.get("data",{}).get("children",[]):
        p = ch.get("data",{})
        u = p.get("url_overridden_by_dest","")
        if u.lower().endswith((".jpg",".jpeg",".png",".webp")):
            out.append(u)
            continue
        prev = p.get("preview",{})
        imgs = prev.get("images",[])
        if imgs:
            src = imgs[0].get("source",{}).get("url")
            if src: out.append(src.replace("&amp;","&"))
    return out

def pick_best_image(urls: List[str]) -> Optional[Image.Image]:
    best = None; best_px = 0; tried = 0
    for u in urls:
        if tried >= MAX_CANDIDATES: break
        raw = download_bytes(u)
        if not raw: continue
        img = open_image(raw)
        if not img: continue
        w,h = img.size; px = w*h
        if w < MIN_W or h < MIN_H:
            if best is None and px > best_px:
                best, best_px = img, px
            continue
        if px > best_px:
            best, best_px = img, px
        tried += 1
    return best

def best_image_for_post(title: str, link: str) -> Image.Image:
    candidates: List[str] = []
    if link:
        r = http_get(link)
        if r and r.text:
            candidates += image_candidates_from_html(r.text, link)
    if title:
        candidates += reddit_image_candidates(title)

    img = pick_best_image(candidates)
    if img is None:
        img = gradient_bg()
    return upscale_if_small(img)

# =========================
# Image shaping & UI
# =========================
def upscale_if_small(img: Image.Image) -> Image.Image:
    w0,h0 = img.size
    if w0 >= W and h0 >= H: return img
    scale = min(3.0, max(W/max(1,w0), H/max(1,h0)))
    up = img.resize((int(w0*scale), int(h0*scale)), Image.LANCZOS)
    if scale > 1.4:
        up = up.filter(ImageFilter.GaussianBlur(0.4))
    return up

def cover(img: Image.Image, w=W, h=H) -> Image.Image:
    if img.mode == "RGBA":
        img = img.convert("RGB")
    sw,sh = img.size
    scale = max(w/sw, h/sh)
    nw,nh = int(sw*scale), int(sh*scale)
    img = img.resize((nw,nh), Image.LANCZOS)
    x1 = (nw - w)//2; y1 = (nh - h)//2
    return img.crop((x1,y1,x1+w,y1+h))


def gradient_bg() -> Image.Image:
    top=(22,24,32); bottom=(10,11,14)
    img=Image.new("RGB",(W,H),top); px=img.load()
    for y in range(H):
        t=y/(H-1)
        c=(int(top[0]*(1-t)+bottom[0]*t),
           int(top[1]*(1-t)+bottom[1]*t),
           int(top[2]*(1-t)+bottom[2]*t))
        for x in range(W): px[x,y]=c
    return img

def draw_bottom_gradient(img: Image.Image, height: int = 520, opacity: int = 235):
    overlay = Image.new("RGBA", (W, height), (0,0,0,0))
    pix = overlay.load()
    for y in range(height):
        a = int(opacity * (y / (height-1)))
        for x in range(W):
            pix[x,y] = (0,0,0,a)
    img.paste(overlay, (0, H-height), overlay)

def soften_bottom_section(img: Image.Image, height: int = 520, blur_radius: int = 18, opacity: int = 220):
    region_height = min(height, H)
    if region_height <= 0:
        return
    blurred = img.filter(ImageFilter.GaussianBlur(blur_radius))
    mask_full = Image.new("L", (W, H), 0)
    start_y = max(0, H - region_height)
    mask_slice = Image.new("L", (W, region_height), 0)
    mask_pixels = mask_slice.load()
    for y in range(region_height):
        t = y / max(1, region_height - 1)
        alpha = int(opacity * t)
        for x in range(W):
            mask_pixels[x, y] = alpha
    mask_full.paste(mask_slice, (0, start_y))
    img.paste(blurred, (0, 0), mask_full)

def wrap_by_pixels(draw, text, font, max_w):
    words = (text or "").split()
    lines=[]; line=""
    for w in words:
        trial=(line+" "+w).strip()
        if draw.textbbox((0,0), trial, font=font)[2] <= max_w:
            line = trial
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)
    return lines

def draw_text_outline(draw, xy, text, font, fill=(255,255,255), outline=(0,0,0), width=4):
    x,y = xy
    for dx in (-width,0,width):
        for dy in (-width,0,width):
            if dx==0 and dy==0: continue
            draw.text((x+dx,y+dy), text, font=font, fill=outline)
    draw.text((x,y), text, font=font, fill=fill)

def draw_badge(draw, text, x, y, font, bg_color=(255,255,255,235), text_color=(25,25,25)):
    # larger pill, like your reference
    pad_x, pad_y = 24, 14
    bbox = draw.textbbox((0,0), text, font=font)
    w = (bbox[2]-bbox[0]) + pad_x*2
    h = (bbox[3]-bbox[1]) + pad_y*2
    if len(bg_color) == 3:
        bg = (*bg_color, 235)
    else:
        bg = bg_color
    draw.rounded_rectangle((x, y, x+w, y+h), radius=18, fill=bg)
    draw.text((x+pad_x, y+pad_y), text, font=font, fill=text_color)

def compose_single(headline: str, src_domain: str, bg: Image.Image) -> Image.Image:
    # 1) background fit
    img = cover(bg)

    # 2) gradient bottom for legibility
    soften_bottom_section(img, height=520, blur_radius=18, opacity=220)
    draw_bottom_gradient(img, height=520, opacity=235)
    draw = ImageDraw.Draw(img, "RGBA")

    # 3) fonts
    # Start large; shrink until the text fits <=3 lines
    badge_font = load_font(FONT_REG_PATH, 48)   # bigger badge
    small_font = load_font(FONT_REG_PATH, 30)

    # headline dynamic sizing
    max_width = W - 96
    target_lines = 3
    size = 112   # start big
    min_size = 64
    while size >= min_size:
        h1 = load_font(FONT_BOLD_PATH, size)
        lines = wrap_by_pixels(draw, (headline or "").upper(), h1, max_width)
        if len(lines) <= target_lines:
            break
        size -= 6
    h1_font = load_font(FONT_BOLD_PATH, max(min_size, size))
    lines = wrap_by_pixels(draw, (headline or "").upper(), h1_font, max_width)

    # 4) badge top-left
    badge_bg, badge_fg = badge_colors_for_domain(src_domain)
    draw_badge(draw, BADGE_TEXT, x=48, y=48, font=badge_font, bg_color=badge_bg, text_color=badge_fg)

    # 5) headline lines near bottom
    y = H - 380
    for ln in lines:
        bb = draw.textbbox((0,0), ln, font=h1_font)
        lh = bb[3]-bb[1]
        draw_text_outline(draw, (48, y), ln, h1_font, fill=(255,255,255), outline=(0,0,0), width=5)
        y += lh + 10

    # 6) footer/logo
    logo = load_logo()
    if logo:
        lw, lh = logo.size
        img.paste(logo, (W - 48 - lw, H - 48 - lh), logo)
    else:
        footer = f"@{BRAND_NAME}" if BRAND_NAME else ""
        if footer:
            fb = draw.textbbox((0,0), footer, font=small_font)
            draw.text((W-48-(fb[2]-fb[0]), H-48-(fb[3]-fb[1])), footer, font=small_font, fill=(230,230,230))

    return img


def generate_single_image(post_id: str, headline: str, link: str, force: bool = False) -> Path:
    title = (headline or "").strip()
    out_path = OUT_DIR / f"{post_id}.jpg"
    if out_path.exists() and not force:
        print(f"[single] exists -> {out_path.name}")
        return out_path

    src_domain = domain_of(link)
    bg = best_image_for_post(title, link)
    final = compose_single(headline=title, src_domain=src_domain, bg=bg).convert("RGB")
    ensure_dir(OUT_DIR)
    final.save(out_path, "JPEG", quality=92, optimize=True, progressive=True)
    print(f"[single] saved -> {out_path}")
    return out_path

# =========================
# Per-post driver
# =========================
def process_script(fp: Path):
    data = json.loads(fp.read_text(encoding="utf-8"))
    post_id = (data.get("id") or fp.stem)[:64]
    title = (data.get("title") or data.get("slides", [{}])[0].get("h1", "") or "").strip()
    link = data.get("link", "")
    generate_single_image(post_id=post_id, headline=title, link=link)

# =========================
# Main
# =========================
def main():
    ensure_dir(OUT_DIR)
    files = sorted(SCRIPTS_DIR.glob("*.json"))
    if not files:
        print("[single] no scripts in out/scripts/")
        return
    for fp in files:
        process_script(fp)

if __name__ == "__main__":
    main()
