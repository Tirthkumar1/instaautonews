#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor source Instagram accounts for new reels, download them locally,
and optionally repost them to the configured destination account.

Usage:
    python reels_mirror.py

Environment (.env):
    IG_USERNAME / IG_PASSWORD / IG_SESSION_PATH
        Credentials and session cache for the destination account.
    IG_POST_ENABLED
        Mirror uploads only when true; otherwise videos are just downloaded.
    REELS_SOURCE_USERS
        Comma-separated list of usernames to watch (e.g. "account1,account2").
    REELS_POLL_INTERVAL
        Seconds between polls (default 900 / 15 minutes).
    REELS_MAX_PER_USER
        Maximum reels to fetch per user on each run (default 5).
    REELS_CAPTION_TEMPLATE
        Optional template for repost captions, e.g.
        "{original_caption}\n\nRepost via @{source_username}"
"""

import json
import os
import re
import subprocess
import time
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image

try:
    from instagrapi import Client
except ImportError:  # pragma: no cover - dependency should be installed manually
    Client = None  # type: ignore

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore


# =========================
# Config & constants
# =========================
load_dotenv()

DATA_DIR = Path("out")
STATE_PATH = DATA_DIR / "reels_state.json"
DOWNLOAD_DIR = DATA_DIR / "reels_downloads"

def parse_source_modes(raw: str) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            username, mode = chunk.split(":", 1)
            username = username.strip()
            mode = mode.strip().lower() or "raw"
        else:
            username = chunk
            mode = "raw"
        mapping[username.lower()] = {"username": username, "mode": mode}
    return mapping


SOURCE_MODES = parse_source_modes(os.getenv("REELS_SOURCE_USERS", ""))
REELS_SOURCE_USERS: List[str] = [entry["username"] for entry in SOURCE_MODES.values()]

REELS_POLL_INTERVAL = int(os.getenv("REELS_POLL_INTERVAL", "900"))  # 15 min default
REELS_MAX_PER_USER = int(os.getenv("REELS_MAX_PER_USER", "5"))
REELS_CAPTION_TEMPLATE = os.getenv(
    "REELS_CAPTION_TEMPLATE",
    "Repost from @{source_username}\n\n{original_caption}",
)

IG_POST_ENABLED = os.getenv("IG_POST_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
IG_USERNAME = os.getenv("IG_USERNAME", "").strip()
IG_PASSWORD = os.getenv("IG_PASSWORD", "").strip()
IG_SESSION_PATH = Path(os.getenv("IG_SESSION_PATH", "out/ig_session.json"))

REBRAND_LOGO_PATH = Path(os.getenv("REBRAND_LOGO_PATH", "assets/reels_logo.png"))
REBRAND_LOGO_MAX_WIDTH = int(os.getenv("REBRAND_LOGO_MAX_WIDTH", "512"))
REBRAND_LOGO_MAX_HEIGHT = int(os.getenv("REBRAND_LOGO_MAX_HEIGHT", "256"))
REBRAND_PADDING_RATIO = float(os.getenv("REBRAND_PADDING_RATIO", "0.25"))
REBRAND_PADDING_PX = int(os.getenv("REBRAND_PADDING_PX", "24"))


# =========================
# Helpers
# =========================
def ensure_dirs() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> Dict[str, Dict[str, str]]:
    if not STATE_PATH.exists():
        return {"processed": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"processed": {}}


def save_state(state: Dict[str, Dict[str, str]]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def init_client() -> Optional["Client"]:
    if Client is None:
        print("[reels] instagrapi not installed. Run `pip install instagrapi`.")
        return None
    if not IG_USERNAME or not IG_PASSWORD:
        print("[reels] Missing IG_USERNAME / IG_PASSWORD.")
        return None

    cl = Client()
    if IG_SESSION_PATH.exists():
        try:
            cl.load_settings(json.loads(IG_SESSION_PATH.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[reels] Could not load session settings: {exc}")
    try:
        cl.login(IG_USERNAME, IG_PASSWORD)
    except Exception as exc:
        print(f"[reels] Login failed: {exc}")
        return None
    try:
        IG_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        IG_SESSION_PATH.write_text(json.dumps(cl.get_settings()), encoding="utf-8")
    except Exception as exc:
        print(f"[reels] Could not persist session: {exc}")
    return cl


def human_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def summarize_caption(text: str, limit: int = 2200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def caption_for_repost(source_username: str, caption_text: str) -> str:
    original = summarize_caption(caption_text or "", 1000).strip()
    template = REELS_CAPTION_TEMPLATE or "{original_caption}"
    context = {
        "source_username": source_username,
        "original_caption": original,
        "timestamp": human_ts(),
    }
    try:
        caption = template.format_map(context)
    except KeyError as exc:
        print(f"[reels] Caption template missing key {exc}; falling back.")
        caption = original
    if not caption.strip():
        caption = f"Repost from @{source_username}"
    return caption[:2200]


def extract_first_frame(video_path: Path) -> Optional[Path]:
    frame_path = video_path.with_suffix(".frame0.png")
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(frame_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return frame_path
    except Exception as exc:
        print(f"[reels] ffmpeg extract frame failed: {exc}")
        return None


def _union_bbox(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return x1, y1, x2 - x1, y2 - y1


def detect_username_box(frame_path: Path, username: str) -> Optional[Tuple[int, int, int, int]]:
    if pytesseract is None:
        print("[reels] pytesseract not installed; cannot detect username.")
        return None
    try:
        img = Image.open(frame_path)
    except Exception as exc:
        print(f"[reels] Could not open frame for detection: {exc}")
        return None

    try:
        data = pytesseract.image_to_data(img, output_type=TesseractOutput.DICT)
    except Exception as exc:
        print(f"[reels] OCR failed: {exc}")
        return None

    target = username.lower().lstrip("@")
    tokens = data["text"]
    boxes = list(
        zip(
            data["left"],
            data["top"],
            data["width"],
            data["height"],
        )
    )

    best_box = None
    token_count = len(tokens)
    for i in range(token_count):
        if not tokens[i].strip():
            continue
        current = ""
        group = []
        for j in range(i, token_count):
            token = tokens[j].strip()
            if not token:
                break
            current += token.lower()
            group.append(boxes[j])
            if target in current:
                best_box = _union_bbox(group)
                break
            if len(current) > len(target) + 4:
                break
        if best_box:
            break

    if not best_box:
        print(f"[reels] Unable to locate username @{username} in frame.")
        return None

    x, y, w, h = best_box
    pad_x = int(w * REBRAND_PADDING_RATIO) + REBRAND_PADDING_PX
    pad_y = int(h * REBRAND_PADDING_RATIO) + REBRAND_PADDING_PX
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = w + pad_x * 2
    h = h + pad_y * 2
    img_w, img_h = img.size
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return (x, y, w, h)


def prepare_logo_for_overlay(video_path: Path, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[Path, int, int, int, int]]:
    if not REBRAND_LOGO_PATH.exists():
        print(f"[reels] Logo not found at {REBRAND_LOGO_PATH}")
        return None
    try:
        logo = Image.open(REBRAND_LOGO_PATH).convert("RGBA")
    except Exception as exc:
        print(f"[reels] Failed to load logo: {exc}")
        return None

    target_w = min(bbox[2], REBRAND_LOGO_MAX_WIDTH)
    target_h = min(bbox[3], REBRAND_LOGO_MAX_HEIGHT)
    scale = min(target_w / max(1, logo.width), target_h / max(1, logo.height))
    scale = min(scale, 1.0)
    scaled_w = max(1, int(logo.width * scale))
    scaled_h = max(1, int(logo.height * scale))
    scaled = logo.resize((scaled_w, scaled_h), Image.LANCZOS)
    temp_path = video_path.with_suffix(".logo.png")
    try:
        scaled.save(temp_path)
    except Exception as exc:
        print(f"[reels] Could not save scaled logo: {exc}")
        return None
    x, y, w, h = bbox
    overlay_x = x + max(0, (w - scaled_w) // 2)
    overlay_y = y + max(0, (h - scaled_h) // 2)
    return temp_path, scaled_w, scaled_h, overlay_x, overlay_y


def rebrand_reel(video_path: Path, username: str) -> Optional[Path]:
    frame_path = extract_first_frame(video_path)
    if frame_path is None:
        return None

    bbox = detect_username_box(frame_path, username)
    try:
        frame_path.unlink(missing_ok=True)
    except Exception:
        pass

    if bbox is None:
        return None

    logo_info = prepare_logo_for_overlay(video_path, bbox)
    if logo_info is None:
        return None

    logo_path, logo_w, logo_h, overlay_x, overlay_y = logo_info
    draw_x, draw_y, draw_w, draw_h = bbox

    output_path = video_path.with_suffix(".rebrand.mp4")
    filter_complex = (
        f"[1:v]scale={logo_w}:{logo_h}[logo];"
        f"[0:v]drawbox=x={draw_x}:y={draw_y}:w={draw_w}:h={draw_h}:color=black@1:t=fill[base];"
        f"[base][logo]overlay=x={overlay_x}:y={overlay_y}"
    )
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(logo_path),
        "-filter_complex",
        filter_complex,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[reels] ffmpeg rebrand failed: {exc}")
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    finally:
        try:
            logo_path.unlink(missing_ok=True)
        except Exception:
            pass

    return output_path


def fetch_new_reels(
    cl: "Client", state: Dict[str, Dict[str, str]], username: str
) -> List[Dict]:
    user_key = username.lower()
    processed = state.setdefault("processed", {})
    user_state = processed.setdefault(user_key, {})

    try:
        user_id = cl.user_id_from_username(username)
        medias = cl.user_clips(user_id, amount=REELS_MAX_PER_USER)
    except Exception as exc:
        print(f"[reels] Failed to fetch clips for @{username}: {exc}")
        return []

    new_medias = []
    for media in medias:
        media_id = str(media.pk)
        if media_id in user_state:
            continue
        new_medias.append(
            {
                "pk": media_id,
                "username": username,
                "caption": getattr(media, "caption_text", "") or "",
                "taken_at": getattr(media, "taken_at", None),
            }
        )
    # Process newest first
    new_medias.sort(key=lambda x: x.get("taken_at") or dt.datetime.min, reverse=True)
    return new_medias


def download_clip(cl: "Client", media_pk: str, username: str) -> Optional[Path]:
    dest = DOWNLOAD_DIR / f"{username}_{media_pk}.mp4"
    try:
        cl.clip_download(media_pk, str(dest))
        return dest
    except Exception as exc:
        print(f"[reels] Download failed for {media_pk}: {exc}")
        return None


def repost_clip(
    cl: "Client", video_path: Path, caption: str, original_username: str
) -> bool:
    if not IG_POST_ENABLED:
        print(f"[reels] Posting disabled; saved {video_path.name}")
        return False
    try:
        cl.clip_upload(str(video_path), caption)
        print(f"[reels] Uploaded repost of @{original_username} -> {video_path.name}")
        return True
    except Exception as exc:
        print(f"[reels] Upload failed for {video_path.name}: {exc}")
        return False


def process_sources(cl: "Client") -> None:
    ensure_dirs()
    state = load_state()
    any_updates = False

    for user_key, meta in SOURCE_MODES.items():
        username = meta["username"]
        mode = meta.get("mode", "raw")
        new_medias = fetch_new_reels(cl, state, username)
        if not new_medias:
            continue
        print(f"[reels] Found {len(new_medias)} new reels for @{username}")
        for media in new_medias:
            pk = media["pk"]
            caption = caption_for_repost(username, media.get("caption", ""))
            video_path = download_clip(cl, pk, user_key)
            if video_path is None:
                continue
            final_path = video_path
            processed = False
            if mode == "rebrand":
                branded = rebrand_reel(video_path, username)
                if branded is not None:
                    final_path = branded
                    processed = True
                else:
                    print(f"[reels] Rebrand failed for @{username}; using original video.")
            posted = repost_clip(cl, final_path, caption, username)
            try:
                final_path.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[reels] Cleanup failed for {final_path.name}: {exc}")
            if processed:
                try:
                    video_path.unlink(missing_ok=True)
                except Exception:
                    pass
            state["processed"][user_key][pk] = human_ts()
            state.setdefault("log", []).append(
                {
                    "pk": pk,
                    "source": username,
                    "caption": caption,
                    "posted": posted,
                    "mode": mode,
                    "ts": human_ts(),
                }
            )
            any_updates = True

    if any_updates:
        save_state(state)


# =========================
# Main
# =========================
def validate_sources() -> bool:
    if not REELS_SOURCE_USERS:
        print("[reels] No source usernames set. Populate REELS_SOURCE_USERS in .env")
        return False
    valid_modes = {"raw", "rebrand"}
    bad = [meta for meta in SOURCE_MODES.values() if meta.get("mode") not in valid_modes]
    if bad:
        values = ", ".join(f"@{meta['username']}({meta.get('mode')})" for meta in bad)
        print(f"[reels] Invalid mode for: {values}. Allowed: raw, rebrand")
        return False
    if any(meta.get("mode") == "rebrand" for meta in SOURCE_MODES.values()):
        if pytesseract is None:
            print("[reels] pytesseract is required for rebrand mode. Install with `pip install pytesseract` and ensure Tesseract OCR is available on PATH.")
            return False
        if not REBRAND_LOGO_PATH.exists():
            print(f"[reels] Logo missing at {REBRAND_LOGO_PATH}. Set REBRAND_LOGO_PATH to your overlay asset.")
            return False
    return True


def main() -> None:
    if not validate_sources():
        return
    cl = init_client()
    if cl is None:
        return

    print(
        "[reels] Mirror running. Sources: "
        + ", ".join(f"@{u}" for u in REELS_SOURCE_USERS)
    )
    while True:
        try:
            process_sources(cl)
        except Exception as exc:
            print(f"[reels] Error: {exc}")
        time.sleep(REELS_POLL_INTERVAL)


if __name__ == "__main__":
    main()
