#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, re, json, time, hashlib, datetime, urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import feedparser
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

from bg_from_web import generate_single_image

try:
    from instagrapi import Client
except ImportError:
    Client = None

# =========================
# Config
# =========================
load_dotenv()

FEEDS = [u.strip() for u in os.getenv("FEEDS", "").split(",") if u.strip()]
DEEPAI_KEYS = [k.strip() for k in os.getenv("DEEPAI_KEYS", "").split(",") if k.strip()]
assert len(DEEPAI_KEYS) >= 3, "Need 3 DeepAI keys in .env"

TOP_N = int(os.getenv("TOP_N", "3"))  # number of stories per hour

DATA_DIR = Path("out")
SCRIPTS_DIR = DATA_DIR / "scripts"
BG_DIR = DATA_DIR / "backgrounds"
IMG_DIR = DATA_DIR / "final_images"
STATE_DB = DATA_DIR / "state.json"

IG_POST_ENABLED = os.getenv("IG_POST_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
IG_USERNAME = os.getenv("IG_USERNAME", "").strip()
IG_PASSWORD = os.getenv("IG_PASSWORD", "").strip()
IG_SESSION_PATH = Path(os.getenv("IG_SESSION_PATH", "out/ig_session.json"))
IG_CAPTION_TEMPLATE = os.getenv("IG_CAPTION_TEMPLATE", "{title}\n\n{summary}\n\n{read_more}")
IG_HASHTAGS = os.getenv("IG_HASHTAGS", "#news #worldnews #AEYWorldNews #DailyBrief")
CAPTION_BODY_MAX_CHARS = int(os.getenv("CAPTION_BODY_MAX_CHARS", "650"))
CAPTION_SUMMARY_ENABLED = os.getenv("CAPTION_SUMMARY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
CAPTION_SUMMARY_MODEL = os.getenv("CAPTION_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6").strip()
CAPTION_SUMMARY_MIN_TOKENS = int(os.getenv("CAPTION_SUMMARY_MIN_TOKENS", "45"))
CAPTION_SUMMARY_MAX_TOKENS = int(os.getenv("CAPTION_SUMMARY_MAX_TOKENS", "130"))
CAPTION_SUMMARY_INPUT_MAX_CHARS = int(os.getenv("CAPTION_SUMMARY_INPUT_MAX_CHARS", "3500"))


UA = "Mozilla/5.0"
TIMEOUT = 25

# Canvas
W, H = 1080, 1350
MARGIN_X, MARGIN_Y = 64, 80
PANEL = (0,0,0,180)
FG = (245,245,245)
SUB = (215,215,215)
LINE_SP = 1.22

# Fonts
FONT_H_BOLD = "assets/Inter-Bold.ttf"
FONT_B_REG  = "assets/Inter-Regular.ttf"

for p in (SCRIPTS_DIR, BG_DIR, IMG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# =========================
# State
# =========================
def load_state() -> Dict:
    if not STATE_DB.exists(): return {"seen_ids": {}}
    return json.loads(STATE_DB.read_text(encoding="utf-8"))

def save_state(st: Dict): STATE_DB.write_text(json.dumps(st, indent=2), encoding="utf-8")

def story_id(link: str, title: str) -> str:
    return hashlib.md5(((link or "")+"|"+(title or "")).encode("utf-8")).hexdigest()

def domain_of(url: str) -> str:
    try: return urllib.parse.urlparse(url).netloc
    except: return ""

def short(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "..."

def load_font(path: str, size: int):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()




_SUMMARY_PIPELINE = None
_SUMMARY_PIPELINE_FAILED = False

def _get_summarizer():
    global _SUMMARY_PIPELINE, _SUMMARY_PIPELINE_FAILED
    if not CAPTION_SUMMARY_ENABLED:
        return None
    if _SUMMARY_PIPELINE is not None:
        return _SUMMARY_PIPELINE
    if _SUMMARY_PIPELINE_FAILED:
        return None
    if hf_pipeline is None:
        print("[SUM] transformers package not available; install `transformers` to enable improved captions.")
        _SUMMARY_PIPELINE_FAILED = True
        return None
    try:
        _SUMMARY_PIPELINE = hf_pipeline("summarization", model=CAPTION_SUMMARY_MODEL)
    except Exception as exc:
        print(f"[SUM] Unable to load summarization model '{CAPTION_SUMMARY_MODEL}': {exc}")
        _SUMMARY_PIPELINE_FAILED = True
        return None
    return _SUMMARY_PIPELINE

def _unique_sentences(lines: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for raw in lines:
        text = (raw or "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        if not text.endswith((".", "!", "?")):
            text += "."
        ordered.append(text)
    return ordered

def _trim_to_limit(text: str, limit: int) -> str:
    txt = text.strip()
    if len(txt) <= limit:
        return txt
    words = txt.split()
    trimmed = []
    total = 0
    for word in words:
        delta = len(word) + (1 if trimmed else 0)
        if total + delta > limit:
            break
        trimmed.append(word)
        total += delta
    if trimmed:
        return ' '.join(trimmed).rstrip(' ,;:-') + '...'
    return txt[:limit].rstrip(' ,;:-') + '...'

def clean_html_text(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"<[^>]+>", " ", value).replace("\xa0", " " ).strip()

def summarize_for_caption(script: Dict, item: Dict) -> str:
    slides = script.get("slides", []) or []
    harvested: List[str] = []
    for slide in slides:
        harvested.append(slide.get("h1", ""))
        harvested.extend(slide.get("bullets") or [])
        harvested.append(slide.get("sub", ""))
    rss_summary = clean_html_text(item.get("summary", "") or "")
    if rss_summary:
        harvested.append(rss_summary)
    segments = _unique_sentences(harvested)
    source_text = " ".join(segments)
    if len(source_text) > CAPTION_SUMMARY_INPUT_MAX_CHARS:
        source_text = source_text[:CAPTION_SUMMARY_INPUT_MAX_CHARS]
    summarizer = _get_summarizer()
    summary = ""
    if summarizer and source_text:
        max_tokens = max(20, CAPTION_SUMMARY_MAX_TOKENS)
        min_tokens = max(10, min(CAPTION_SUMMARY_MIN_TOKENS, max_tokens - 5))
        try:
            result = summarizer(
                source_text,
                max_length=max_tokens,
                min_length=min_tokens,
                do_sample=False,
            )
            if result and isinstance(result, list):
                summary = (result[0].get("summary_text") or "").strip()
        except Exception as exc:
            print(f"[SUM] Summarizer failed, falling back to simple caption: {exc}")
    if not summary:
        summary = source_text
    summary = re.sub(r"\s+", " ", summary).strip()
    return _trim_to_limit(summary, CAPTION_BODY_MAX_CHARS)

class SafeDict(dict):
    def __missing__(self, key):
        return ""

class InstagramPoster:
    def __init__(self):
        self.enabled = IG_POST_ENABLED
        self.username = IG_USERNAME
        self.password = IG_PASSWORD
        self.session_path = IG_SESSION_PATH
        self.client: Optional["Client"] = None

    def _log(self, message: str) -> None:
        print(f"[IG] {message}")

    def ensure_client(self):
        if not self.enabled:
            return None
        if Client is None:
            self._log("Dependency instagrapi missing. Run `pip install instagrapi` to enable uploads.")
            return None
        if not self.username or not self.password:
            self._log("Missing IG_USERNAME/IG_PASSWORD environment variables; skipping upload.")
            return None
        if self.client is not None:
            return self.client
        client = Client()
        if self.session_path.exists():
            try:
                stored = json.loads(self.session_path.read_text(encoding="utf-8"))
                client.load_settings(stored)
            except Exception as exc:
                self._log(f"Could not reuse saved session: {exc}")
        try:
            client.login(self.username, self.password)
        except Exception as exc:
            self._log(f"Login failed: {exc}")
            return None
        try:
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            self.session_path.write_text(json.dumps(client.get_settings()), encoding="utf-8")
        except Exception as exc:
            self._log(f"Could not persist Instagram session: {exc}")
        self.client = client
        self._log(f"Ready to post as {self.username}")
        return client

    def post_photo(self, image_path: Path, caption: str) -> bool:
        if not self.enabled:
            self._log("Posting disabled; generated assets only.")
            return False
        client = self.ensure_client()
        if client is None:
            return False
        try:
            client.photo_upload(str(image_path), caption)
            self._log(f"Uploaded {image_path.name}")
            return True
        except Exception as exc:
            self._log(f"Upload failed: {exc}")
            return False

INSTAGRAM_POSTER = InstagramPoster()

def build_caption(script: Dict, item: Dict) -> str:
    title = (script.get("title") or "").strip()
    link = script.get("link", "") or item.get("link", "")
    summary = summarize_for_caption(script, item)
    read_more = f"Read more -> {link}" if link else ""
    tags = IG_HASHTAGS.strip()
    context = SafeDict(title=title, link=link, domain=domain_of(link), summary=summary, read_more=read_more, hashtags=tags, hashtags_line=(("\n\n" + tags) if tags else ""))
    template = IG_CAPTION_TEMPLATE or "{title}\n\n{summary}\n\n{read_more}"
    caption = template.format_map(context).strip()
    if tags and "{hashtags" not in template:
        caption = caption.rstrip()
        if caption:
            caption += "\n\n"
        caption += tags
    return caption.strip()

# =========================
# Step 1: fetch & rank
# =========================
def fetch_rss_items(feeds: List[str]) -> List[Dict]:
    items=[]
    for url in feeds:
        try:
            fp=feedparser.parse(url)
            for e in fp.entries:
                title=e.get("title","").strip()
                link=e.get("link","").strip()
                summ=(e.get("summary") or e.get("description") or "").strip()
                ts=time.mktime(e.get("published_parsed") or e.get("updated_parsed") or time.gmtime())
                items.append({"title":title,"link":link,"summary":summ,"ts":ts,"domain":domain_of(link)})
        except: continue
    return items

def rank_items(items: List[Dict]) -> List[Dict]:
    now=time.time()
    ranked=[]
    for it in items:
        hours=max(1.0,(now-it["ts"])/3600)
        score=10.0/hours
        it["score"]=score
        ranked.append(it)
    ranked.sort(key=lambda x:x["score"], reverse=True)
    return ranked

# =========================
# Step 2: script builder
# =========================
def script_from_item(it: Dict) -> Dict:
    hook=short(it["title"],90)
    sub=f"Source: {it['domain']}"
    bullets=[b.strip() for b in re.split(r'[.;\n]+', it["summary"]) if len(b.strip())>6][:5]
    so_what=[f"Why it matters: {short(b,120)}" for b in bullets[:3]] or ["Why it matters: developing"]
    cta="Follow for more."
    return {
        "id": story_id(it["link"], it["title"]),
        "title": it["title"],
        "link": it["link"],
        "slides":[
            {"h1": hook, "sub": short(sub,40)},
            {"h1": "Key details", "bullets": bullets[:3]},
            {"h1": "Why it matters", "bullets": so_what+[cta]},
        ]
    }

# =========================
# Step 3: backgrounds
# =========================
def build_prompt(title: str) -> str:
    return f"carousel background: {title}. minimal editorial, subtle gradient, soft lighting, no text"

def try_download(url: str) -> Optional[bytes]:
    try:
        r=requests.get(url,headers={"User-Agent":UA},timeout=TIMEOUT);r.raise_for_status()
        return r.content
    except: return None

def deepai_image(prompt: str, key_index: int) -> Optional[Image.Image]:
    api_key=DEEPAI_KEYS[key_index]
    ck=hashlib.md5(f"{key_index}|{prompt}".encode()).hexdigest()
    cp=BG_DIR/"_cache"/f"{ck}.jpg"; cp.parent.mkdir(parents=True, exist_ok=True)
    if cp.exists():
        try: return Image.open(cp).convert("RGB")
        except: pass
    try:
        r=requests.post("https://api.deepai.org/api/text2img",data={"text":prompt},headers={"api-key":api_key},timeout=TIMEOUT)
        r.raise_for_status(); data=r.json(); url=data.get("output_url")
        raw=try_download(url); img=Image.open(io.BytesIO(raw)).convert("RGB")
        img.save(cp,"JPEG",quality=90); return img
    except: return None

def cover(img: Image.Image, mode="center") -> Image.Image:
    sw,sh=img.size; scale=max(W/sw,H/sh); nw,nh=int(sw*scale),int(sh*scale)
    img=img.resize((nw,nh),Image.LANCZOS)
    if mode=="top": x1=(nw-W)//2; y1=0
    elif mode=="bottom": x1=(nw-W)//2; y1=nh-H
    else: x1=(nw-W)//2; y1=(nh-H)//2
    return img.crop((x1,y1,x1+W,y1+H))

def gradient_bg() -> Image.Image:
    top=(22,24,32); bottom=(10,11,14); img=Image.new("RGB",(W,H),top); px=img.load()
    for y in range(H):
        t=y/(H-1); r=int(top[0]*(1-t)+bottom[0]*t); g=int(top[1]*(1-t)+bottom[1]*t); b=int(top[2]*(1-t)+bottom[2]*t)
        for x in range(W): px[x,y]=(r,g,b)
    return img

def make_backgrounds(pid: str, title: str) -> List[Path]:
    prompt=build_prompt(title); outdir=BG_DIR/pid; outdir.mkdir(parents=True,exist_ok=True)
    plan=[("slide_01.jpg","center",0),("slide_02.jpg","top",1),("slide_03.jpg","bottom",2)]
    paths=[]
    for fname,mode,key in plan:
        img=deepai_image(prompt,key) or gradient_bg()
        img=cover(img,mode=mode); path=outdir/fname; img.save(path,"JPEG",quality=92); paths.append(path)
    return paths

# =========================
# Step 4: render slides
# =========================
def draw_wrapped(draw,text,font,box,color):
    if not text:return box[1]
    x1,y1,x2,y2=box; max_w=x2-x1; words=text.split(); lines=[]; line=""
    for w in words:
        trial=(line+" "+w).strip()
        if draw.textbbox((0,0),trial,font=font)[2]<=max_w: line=trial
        else: lines.append(line); line=w
    if line: lines.append(line)
    asc,desc=font.getmetrics(); lh=int((asc+desc)*LINE_SP); y=y1
    for ln in lines:
        if y+lh>y2: break
        draw.text((x1,y),ln,font=font,fill=color); y+=lh
    return y

def render_slide(bg: Image.Image, slide: Dict) -> Image.Image:
    img=bg.convert("RGB"); draw=ImageDraw.Draw(img,"RGBA")
    px1=MARGIN_X-12; py1=MARGIN_Y-12; px2=W-MARGIN_X+12; py2=H-MARGIN_Y+12
    draw.rounded_rectangle((px1,py1,px2,py2),radius=32,fill=PANEL)
    h1f=load_font(FONT_H_BOLD,68); subf=load_font(FONT_B_REG,34); bodyf=load_font(FONT_B_REG,40)
    x1=MARGIN_X+32; y=MARGIN_Y+32; x2=W-MARGIN_X-32
    y=draw_wrapped(draw,slide.get("h1",""),h1f,(x1,y,x2,y+420),FG)
    if slide.get("sub"): y=draw_wrapped(draw,slide["sub"],subf,(x1,y,x2,y+120),SUB)
    for b in (slide.get("bullets") or [])[:5]:
        y=draw_wrapped(draw,"• "+b,bodyf,(x1,y,x2,y+200),FG)
    return img

def render_post(pid: str, slides: List[Dict]) -> List[Path]:
    bg_paths=[BG_DIR/pid/f"slide_{i:02}.jpg" for i in (1,2,3)]
    outdir=IMG_DIR/pid; outdir.mkdir(parents=True,exist_ok=True)
    paths=[]
    for i,slide in enumerate(slides[:3],start=1):
        bg=Image.open(bg_paths[i-1]).convert("RGB")
        comp=render_slide(bg,slide)
        outp=outdir/f"slide_{i:02}.jpg"; comp.save(outp,"JPEG",quality=90); paths.append(outp)
    return paths

# =========================
# One hour job
# =========================
def one_hour_run():
    st = load_state()
    raw = fetch_rss_items(FEEDS)
    ranked = rank_items(raw)
    new: List[Dict] = []
    for it in ranked:
        sid = story_id(it["link"], it["title"])
        if sid in st["seen_ids"]:
            continue
        st["seen_ids"][sid] = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "link": it["link"],
            "title": it["title"],
        }
        new.append(it)
        if len(new) >= TOP_N:
            break
    if not new:
        save_state(st)
        print("[INFO] No new items.")
        return
    for it in new:
        sid = story_id(it["link"], it["title"])
        scr = script_from_item(it)
        script_path = SCRIPTS_DIR / f"{scr['id']}.json"
        script_path.write_text(json.dumps(scr, indent=2), encoding="utf-8")
        make_backgrounds(scr["id"], scr["title"])
        slides = render_post(scr["id"], scr["slides"])
        single_path = generate_single_image(post_id=scr["id"], headline=scr["title"], link=scr["link"])
        caption = build_caption(scr, it)
        posted = INSTAGRAM_POSTER.post_photo(single_path, caption)
        entry = st["seen_ids"].setdefault(sid, {})
        entry.update({
            "script": str(script_path),
            "slides": [str(p) for p in slides],
            "single": str(single_path),
            "caption": caption,
            "instagram_posted": bool(posted),
        })
        if posted:
            try:
                single_path.unlink(missing_ok=True)
                entry["single_deleted"] = True
            except Exception as exc:
                entry.setdefault("delete_error", str(exc))
        slide_names = [p.name for p in slides]
        status = "posted" if posted else "ready"
        print(f"[GEN] {scr['id']} -> slides={slide_names} single={single_path.name} status={status}")
    save_state(st)

# =========================
# Main loop
# =========================
def main():
    print(f"[BOOT] Autogen running. Polling hourly. IG posting: {'ON' if IG_POST_ENABLED else 'OFF'}")
    while True:
        try: one_hour_run()
        except Exception as e: print("[ERR]",e)
        time.sleep(900)

if __name__=="__main__":
    main()
