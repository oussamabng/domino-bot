#!/usr/bin/env python3
from __future__ import annotations
import os
import io
import math
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import cv2
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Integer, String, DateTime, ForeignKey, Text,
    select, func, BigInteger, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, MessageHandler, filters,
    CallbackQueryHandler, ContextTypes, ConversationHandler
)

# ========= Config, Paths & Logging ========= #
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TARGET_SCORE_DEFAULT = int(os.getenv("TARGET_SCORE", "100"))

# Fixed vision constants (no env needed)
MAX_IMAGE_SIDE = 1400
PIP_REL_MINR = 0.022   # ~2.2% of tile short side
PIP_REL_MAXR = 0.11    # ~11% of tile short side

# --------- i18n: strings --------- #
# Add/adjust keys here; keep placeholders {x} consistent across languages.
I18N: Dict[str, Dict[str, str]] = {
    "en": {
        "app.db_init": "DB initialized at {path}",
        "app.missing_token": "BOT_TOKEN missing. Put it in .env.",
        "cmds.set": "bot commands set",
        "help.text": (
            "Commands:\n"
            "/newgame — create a new game and set team names\n"
            "/score — show the scoreboard\n"
            "/addscore — manually add score to a team\n"
            "/reset — reset the current game to 0–0\n"
            "/lang — change language (English / Arabic)\n"
            "Or use the buttons under the scoreboard."
        ),
        "start.no_game": "👋 Use /newgame to start, then add photos for each team. I’ll count the pips inside the tiles.",
        "newgame.exists": "There is already an active game. Use /reset to clear scores or /start to open the dashboard.",
        "newgame.askA": "🆕 New game! Send <b>Team A</b> name:",
        "newgame.askB": "Great. Now send <b>Team B</b> name:",
        "score.no_game": "No active game. Use /newgame to start.",
        "manual.choose_team": "Choose team to add manual score:",
        "manual.prompt_points": "Send the number of points to add (e.g., 12)",
        "manual.invalid": "Please send a valid integer, e.g., 12",
        "reset.ask": "Reset scores to 0–0?",
        "reset.canceled": "Reset canceled.",
        "reset.yes": "✅ Yes, reset",
        "reset.no": "❌ Cancel",
        "dashboard.add_photo_A": "📷 Add photo → {team}",
        "dashboard.add_photo_B": "📷 Add photo → {team}",
        "dashboard.add_manual": "✏️ Add score manually",
        "dashboard.show_score": "📊 Show score",
        "dashboard.reset": "🔄 Reset",
        "dashboard.end": "🏁 End game",
        "expect.photo.A": "Send a photo for <b>{team}</b> now.",
        "expect.photo.B": "Send a photo for <b>{team}</b> now.",
        "photo.unexpected": "Tap a button first (Add photo → Team A/B), then send the photo.",
        "photo.no_game": "No active game — use /newgame first.",
        "photo.hint.none": "I couldn’t find any domino tiles in that photo. Try a closer shot with good light.",
        "photo.hint.some": "Detected <b>{total}</b> pips from {count} tile(s).",
        "photo.added": "{hint} → added to <b>{team}</b>.",
        "end.done": "Game ended. Use /newgame to start a fresh one.",
        "winner.text": "🏆 <b>{team}</b> reached {target} first and wins!\n\nStart another game? Keep winner '<b>{team}</b>' and rename the loser (<i>{loser}</i>)?",
        "winner.keep_rename": "🔁 Keep winner & rename loser",
        "winner.rename_prompt": "Send the new name for the losing team (winner '<b>{team}</b>' will be kept).",
        "winner.no_ctx": "No winner context found. Use /newgame to start.",
        "wizard.canceled": "Canceled.",
        "main.title": "Domino Game",
        "main.target": "Target",
        "main.rounds": "Rounds",
        "main.status": "Status",
        "main.active": "active",
        "main.finished": "finished",
        "progress.block": "█",
        "progress.empty": "░",
        "callback.no_active": "No active game. Use /newgame to start one.",
        "callback.no_changes": "No changes to update.",
        "lang.current": "Current language: <b>{lang_name}</b>",
        "lang.choose": "Choose your language:",
        "lang.en": "English",
        "lang.ar": "العربية",
        "lang.set_ok": "Language set to <b>{lang_name}</b>.",
    },
    "ar": {
        "app.db_init": "تم تهيئة قاعدة البيانات في {path}",
        "app.missing_token": "مفقود BOT_TOKEN. يرجى إضافته في .env",
        "cmds.set": "تم ضبط أوامر البوت",
        "help.text": (
            "الأوامر:\n"
            "/newgame — ابدأ لعبة جديدة واختر أسماء الفرق\n"
            "/score — عرض لوحة النتائج\n"
            "/addscore — إضافة نقاط يدويًا لفريق\n"
            "/reset — تصفير النتيجة إلى 0–0\n"
            "/lang — تغيير اللغة (العربية / الإنجليزية)\n"
            "أو استخدم الأزرار تحت لوحة النتائج."
        ),
        "start.no_game": "👋 استخدم /newgame للبدء، ثم أرسل صورًا لكل فريق. سأعد النقاط على قطع الدومينو.",
        "newgame.exists": "هناك لعبة نشطة بالفعل. استخدم /reset لمسح النقاط أو /start لفتح اللوحة.",
        "newgame.askA": "🆕 لعبة جديدة! أرسل اسم <b>الفريق A</b>:",
        "newgame.askB": "جيد. الآن أرسل اسم <b>الفريق B</b>:",
        "score.no_game": "لا توجد لعبة نشطة. استخدم /newgame للبدء.",
        "manual.choose_team": "اختر الفريق لإضافة النقاط يدويًا:",
        "manual.prompt_points": "أرسل عدد النقاط المراد إضافتها (مثال: 12)",
        "manual.invalid": "أرسل عددًا صحيحًا صالحًا، مثل 12",
        "reset.ask": "تصفير النتيجة إلى 0–0؟",
        "reset.canceled": "تم الإلغاء.",
        "reset.yes": "✅ نعم، صفّر",
        "reset.no": "❌ إلغاء",
        "dashboard.add_photo_A": "📷 أضف صورة → {team}",
        "dashboard.add_photo_B": "📷 أضف صورة → {team}",
        "dashboard.add_manual": "✏️ إضافة نقاط يدويًا",
        "dashboard.show_score": "📊 عرض النتيجة",
        "dashboard.reset": "🔄 تصفير",
        "dashboard.end": "🏁 إنهاء اللعبة",
        "expect.photo.A": "أرسل صورة لفريق <b>{team}</b> الآن.",
        "expect.photo.B": "أرسل صورة لفريق <b>{team}</b> الآن.",
        "photo.unexpected": "اضغط زرًا أولًا (أضف صورة → الفريق A/B) ثم أرسل الصورة.",
        "photo.no_game": "لا توجد لعبة نشطة — استخدم /newgame أولًا.",
        "photo.hint.none": "لم أعثر على قطع دومينو في الصورة. جرّب صورة أقرب مع إضاءة جيدة.",
        "photo.hint.some": "تم اكتشاف <b>{total}</b> نقطة من {count} قطعة.",
        "photo.added": "{hint} → تمت الإضافة إلى <b>{team}</b>.",
        "end.done": "تم إنهاء اللعبة. استخدم /newgame لبدء لعبة جديدة.",
        "winner.text": "🏆 <b>{team}</b> وصل إلى {target} أولًا وفاز!\n\nهل تريد لعبة أخرى؟ سنبقي الفائز '<b>{team}</b>' ونعيد تسمية الخاسر (<i>{loser}</i>)؟",
        "winner.keep_rename": "🔁 إبقاء الفائز وإعادة تسمية الخاسر",
        "winner.rename_prompt": "أرسل الاسم الجديد للفريق الخاسر (سيُحتفظ بالفائز '<b>{team}</b>').",
        "winner.no_ctx": "لا توجد معلومات عن الفائز. استخدم /newgame للبدء.",
        "wizard.canceled": "تم الإلغاء.",
        "main.title": "لعبة الدومينو",
        "main.target": "الهدف",
        "main.rounds": "الجولات",
        "main.status": "الحالة",
        "main.active": "نشطة",
        "main.finished": "منتهية",
        "progress.block": "█",
        "progress.empty": "░",
        "callback.no_active": "لا توجد لعبة نشطة. استخدم /newgame للبدء.",
        "callback.no_changes": "لا توجد تغييرات لتحديثها.",
        "lang.current": "اللغة الحالية: <b>{lang_name}</b>",
        "lang.choose": "اختر لغتك:",
        "lang.en": "English",
        "lang.ar": "العربية",
        "lang.set_ok": "تم تغيير اللغة إلى <b>{lang_name}</b>.",
    },
}

LANG_NAMES = {"en": "English", "ar": "العربية"}
DEFAULT_LANG = "en"

def t(chat_lang: str, key: str, **kwargs: Any) -> str:
    lang = chat_lang if chat_lang in I18N else DEFAULT_LANG
    s = I18N.get(lang, {}).get(key) or I18N[DEFAULT_LANG].get(key, key)
    try:
        return s.format(**kwargs)
    except Exception:
        return s

# Colored logs (errors red)
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:   "\033[90m",
        logging.INFO:    "\033[92m",
        logging.WARNING: "\033[93m",
        logging.ERROR:   "\033[91m",
        logging.CRITICAL:"\033[41m",
    }
    RESET = "\033[0m"
    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{super().format(record)}{self.RESET}"

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("domino_bot")

if not BOT_TOKEN:
    logger.critical("BOT_TOKEN missing. Put it in .env.")
    raise SystemExit(I18N["en"]["app.missing_token"])

# ========= Database ========= #
class Base(DeclarativeBase): pass

class Game(Base):
    __tablename__ = "games"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    team_a_name: Mapped[str] = mapped_column(String(64))
    team_b_name: Mapped[str] = mapped_column(String(64))
    team_a_score: Mapped[int] = mapped_column(Integer, default=0)
    team_b_score: Mapped[int] = mapped_column(Integer, default=0)
    target_score: Mapped[int] = mapped_column(Integer, default=TARGET_SCORE_DEFAULT)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active|finished
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    rounds: Mapped[List["Round"]] = relationship("Round", back_populates="game", cascade="all, delete-orphan")

class Round(Base):
    __tablename__ = "rounds"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    team: Mapped[str] = mapped_column(String(1))  # 'A' or 'B'
    points: Mapped[int] = mapped_column(Integer)
    method: Mapped[str] = mapped_column(String(16))  # 'opencv' | 'manual'
    image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    game: Mapped[Game] = relationship("Game", back_populates="rounds")

class ChatPref(Base):
    __tablename__ = "chat_prefs"
    __table_args__ = (UniqueConstraint("chat_id", name="uq_chat_prefs_chat"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    lang: Mapped[str] = mapped_column(String(8), default=DEFAULT_LANG)

DB_PATH = BASE_DIR / 'domino.db'
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}", echo=False, future=True)
Base.metadata.create_all(engine)
logger.info(t(DEFAULT_LANG, "app.db_init", path=str(DB_PATH)))

# ========= Small utils ========= #
def timeit(label: str):
    start = time.perf_counter()
    def _done(extra: str = ""):
        dur = (time.perf_counter() - start) * 1000
        logger.debug("%s | %.2f ms %s", label, dur, extra)
    return _done

def order_points(pts):
    pts = np.array(pts, dtype="float32")  # <-- force numpy
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _merge_points(points: List[Tuple[float, float]], min_dist: float) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for (x, y) in points:
        for i, (mx, my) in enumerate(merged):
            if (mx - x) ** 2 + (my - y) ** 2 <= min_dist ** 2:
                merged[i] = ((mx + x) / 2.0, (my + y) / 2.0)
                break
        else:
            merged.append((x, y))
    return merged

# ========= Language helpers ========= #
def get_lang(session: Session, chat_id: int) -> str:
    pref = session.execute(select(ChatPref).where(ChatPref.chat_id == chat_id)).scalar_one_or_none()
    return pref.lang if pref else DEFAULT_LANG

def set_lang(session: Session, chat_id: int, lang: str) -> str:
    lang = lang if lang in I18N else DEFAULT_LANG
    pref = session.execute(select(ChatPref).where(ChatPref.chat_id == chat_id)).scalar_one_or_none()
    if not pref:
        pref = ChatPref(chat_id=chat_id, lang=lang)
        session.add(pref)
    else:
        pref.lang = lang
    session.commit()
    return pref.lang

# ========= Computer Vision ========= #
def _resize_cap(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if max(h, w) <= MAX_IMAGE_SIDE:
        return img_bgr
    s = MAX_IMAGE_SIDE / max(h, w)
    return cv2.resize(img_bgr, (int(w * s), int(h * s)))

def white_neutral_mask_lab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    bright = cv2.inRange(L, 190, 255)
    a_neutral = cv2.inRange(cv2.absdiff(A, 128), 0, 14)
    b_neutral = cv2.inRange(cv2.absdiff(B, 128), 0, 14)
    mask = cv2.bitwise_and(bright, cv2.bitwise_and(a_neutral, b_neutral))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    return mask

def _rects_from_contours(cnts, img_bgr, min_area=800, fill_thresh=0.5, aspect_range=(1.3, 4.5)):
    rects = []
    H, W = img_bgr.shape[:2]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect
        if rw == 0 or rh == 0:
            continue
        long_side, short_side = max(rw, rh), min(rw, rh)
        ratio = long_side / (short_side + 1e-6)

        if not (aspect_range[0] <= ratio <= aspect_range[1]):
            continue

        box = cv2.boxPoints(rect).astype(np.float32)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [box.astype(int)], -1, 255, -1)
        mean_val = cv2.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), mask=mask)[0]
        if mean_val < 100:
            continue

        rects.append((box, (int(long_side), int(short_side))))

    return rects

def _merge_nearby_rects(rects, merge_dist=50):
    merged = []
    centers = []
    for box, (lw, sh) in rects:
        cx, cy = np.mean(box[:,0]), np.mean(box[:,1])
        keep = True
        for i, (mcx, mcy) in enumerate(centers):
            if (cx - mcx) ** 2 + (cy - mcy) ** 2 <= merge_dist ** 2:
                keep = False
                break
        if keep:
            merged.append((box, (lw, sh)))
            centers.append((cx, cy))
    return merged

def find_domino_rects(img_bgr):
    rects = []
    mask = white_neutral_mask_lab(img_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects += _rects_from_contours(cnts, img_bgr, min_area=600, fill_thresh=0.4, aspect_range=(1.2, 4.5))

    if len(rects) < 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 60, 160)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects += _rects_from_contours(cnts, img_bgr, min_area=700, fill_thresh=0.3, aspect_range=(1.1, 5.0))

    return _merge_nearby_rects(rects, merge_dist=60)

def count_pips_on_tile(tile_bgr: np.ndarray) -> int:
    if tile_bgr is None or tile_bgr.size == 0:
        return 0

    h, w = tile_bgr.shape[:2]
    if h > w:
        tile_bgr = cv2.rotate(tile_bgr, cv2.ROTATE_90_CLOCKWISE)
        h, w = tile_bgr.shape[:2]

    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    band = max(2, int(0.08 * w))
    cx0, cx1 = w//2 - band//2, w//2 + band//2
    gray[:, cx0:cx1] = np.clip(gray[:, cx0:cx1] + 60, 0, 255)

    k = max(3, int(0.12 * min(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    mside = max(1, min(h, w) // 2)
    r_min = max(2, int(PIP_REL_MINR * mside))
    r_max = max(r_min + 1, int(PIP_REL_MAXR * mside))
    area_min = math.pi * (r_min * 0.8) ** 2
    area_max = math.pi * (r_max * 1.25) ** 2

    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    keep_centers: List[Tuple[float, float]] = []
    for i in range(1, num):
        x, y, wcc, hcc, area = stats[i]
        if area < area_min or area > area_max:
            continue
        per = 2 * (wcc + hcc)
        if per <= 0:
            continue
        circ = 4 * math.pi * area / (per * per)
        extent = area / max(1, wcc * hcc)
        if circ < 0.60 or extent < 0.45:
            continue
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        keep_centers.append((float(xs.mean()), float(ys.mean())))
    merged = _merge_points(keep_centers, min_dist=max(5.0, r_min * 0.8))
    return len(merged)

def detect_dominoes_and_score(img_bgr: np.ndarray, debug_save: str | None = None) -> Tuple[int, int]:
    tmr = timeit("detect_dominoes_and_score")
    if img_bgr is None or img_bgr.size == 0:
        logger.warning("detect_dominoes_and_score: empty image")
        return 0, 0

    img_bgr = _resize_cap(img_bgr)
    overlay = img_bgr.copy()

    tiles = find_domino_rects(img_bgr)
    logger.info("domino candidates=%d", len(tiles))

    if not tiles:
        tmr("| total=0 dominoes=0")
        return 0, 0

    total = 0
    for idx, (box, (lw, sh)) in enumerate(tiles, start=1):
        rect = order_points(box)
        w, h = lw, sh
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_bgr, M, (w, h))

        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        H, W = warped.shape[:2]
        left = warped[:, : W // 2]
        right = warped[:, W // 2 :]

        l = count_pips_on_tile(left)
        r = count_pips_on_tile(right)
        subtotal = l + r
        total += subtotal

        cv2.polylines(overlay, [box.astype(int)], True, (0, 255, 0), 2)
        cx, cy = np.mean(box[:, 0]), np.mean(box[:, 1])
        cv2.putText(overlay, f"{l}+{r}={subtotal}", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        logger.debug("tile#%d | left=%d right=%d subtotal=%d", idx, l, r, total)

    if debug_save:
        cv2.imwrite(debug_save, overlay)
        logger.info("debug image saved → %s", debug_save)

    tmr(f"| total={total} dominoes={len(tiles)}")
    return total, len(tiles)

# ========= Bot UX ========= #
A_NAME, B_NAME = range(2)
CB_ADD_PHOTO_A = "add_photo_A"
CB_ADD_PHOTO_B = "add_photo_B"
CB_SHOW_SCORE  = "show_score"
CB_ADD_MANUAL  = "add_manual"
CB_RESET_GAME  = "reset_game"
CB_RESET_CONFIRM = "reset_confirm"
CB_RESET_CANCEL  = "reset_cancel"
CB_END_GAME    = "end_game"
CB_NEXT_RENAME_LOSER = "next_rename_loser"
CB_MANUAL_TEAM_A = "manual_team_A"
CB_MANUAL_TEAM_B = "manual_team_B"
CB_LANG_EN = "lang_en"
CB_LANG_AR = "lang_ar"

async def safe_edit(query, *args, **kwargs):
    try:
        return await query.edit_message_text(*args, **kwargs)
    except BadRequest as e:
        if "Message is not modified" in str(e):
            # Try to answer with localized string
            chat_id = query.message.chat_id
            with Session(engine) as s:
                lang = get_lang(s, chat_id)
            await query.answer(t(lang, "callback.no_changes"))
            logger.debug("safe_edit: not-modified chat_id=%s", chat_id)
            return None
        logger.exception("safe_edit error")
        raise

def main_keyboard(game: Game, lang: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(t(lang, "dashboard.add_photo_A", team=game.team_a_name), callback_data=CB_ADD_PHOTO_A),
         InlineKeyboardButton(t(lang, "dashboard.add_photo_B", team=game.team_b_name), callback_data=CB_ADD_PHOTO_B)],
        [InlineKeyboardButton(t(lang, "dashboard.add_manual"), callback_data=CB_ADD_MANUAL)],
        [InlineKeyboardButton(t(lang, "dashboard.show_score"), callback_data=CB_SHOW_SCORE),
         InlineKeyboardButton(t(lang, "dashboard.reset"), callback_data=CB_RESET_GAME)],
        [InlineKeyboardButton(t(lang, "dashboard.end"), callback_data=CB_END_GAME)],
    ]
    return InlineKeyboardMarkup(rows)

def progress_bar(score: int, target: int, lang: str, width: int = 12) -> str:
    filled = min(width, int(round((score / max(1, target)) * width)))
    return t(lang, "progress.block") * filled + t(lang, "progress.empty") * (width - filled)

def render_scoreboard(game: Game, rounds_count: int, lang: str) -> str:
    t_name = t(lang, "main.target")
    rounds_name = t(lang, "main.rounds")
    status_name = t(lang, "main.status")
    status_val = t(lang, "main.finished") if game.status == "finished" else t(lang, "main.active")
    return (
        f"<b>{t(lang, 'main.title')}</b> — {t_name} <b>{game.target_score}</b>\n"
        f"<b>{game.team_a_name}</b>: {game.team_a_score}  {progress_bar(game.team_a_score, game.target_score, lang)}\n"
        f"<b>{game.team_b_name}</b>: {game.team_b_score}  {progress_bar(game.team_b_score, game.target_score, lang)}\n"
        f"{rounds_name}: <i>{rounds_count}</i>\n"
        f"{status_name}: <b>{status_val}</b>"
    )

# ========= DB helpers ========= #
def get_active_game(session: Session, chat_id: int) -> Optional[Game]:
    g = session.execute(select(Game).where(Game.chat_id == chat_id, Game.status == "active")).scalar_one_or_none()
    return g

def create_game(session: Session, chat_id: int, team_a: str, team_b: str, target: int) -> Game:
    g = Game(chat_id=chat_id, team_a_name=team_a.strip()[:64], team_b_name=team_b.strip()[:64], target_score=target)
    session.add(g)
    session.commit()
    session.refresh(g)
    logger.info("created game id=%s chat_id=%s A='%s' B='%s' target=%d", g.id, chat_id, g.team_a_name, g.team_b_name, target)
    return g

def add_points(session: Session, game: Game, team: str, points: int, method: str, image_path: Optional[str] = None) -> Game:
    points = max(0, int(points))
    r = Round(game_id=game.id, team=team, points=points, method=method, image_path=image_path)
    session.add(r)
    if team == 'A':
        game.team_a_score += points
    else:
        game.team_b_score += points
    game.updated_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(game)
    logger.info("add_points game_id=%s team=%s points=%d method=%s image=%s totals A=%d B=%d",
                game.id, team, points, method, bool(image_path), game.team_a_score, game.team_b_score)
    return game

def reset_game_scores(session: Session, game: Game):
    game.team_a_score = 0
    game.team_b_score = 0
    game.updated_at = datetime.now(timezone.utc)
    session.commit()
    logger.warning("reset scores game_id=%s", game.id)

def finish_game(session: Session, game: Game):
    game.status = "finished"
    session.commit()
    logger.warning("finish game_id=%s final A=%d B=%d", game.id, game.team_a_score, game.team_b_score)

# ========= Handlers ========= #
async def set_commands(app: Application):
    # Set default (English) commands; Telegram also supports per-language, but we keep it simple
    commands = [
        BotCommand("start", "Start the bot / show menu"),
        BotCommand("newgame", "Start new game (set team names)"),
        BotCommand("score", "Show current score"),
        BotCommand("addscore", "Manually add score"),
        BotCommand("reset", "Reset the current game to 0–0"),
        BotCommand("help", "How to use the bot"),
        BotCommand("lang", "Change language (English / Arabic)"),
    ]
    await app.bot.set_my_commands(commands)
    logger.info(I18N["en"]["cmds.set"])

# ----- Language command ----- #
async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "lang.en"), callback_data=CB_LANG_EN),
         InlineKeyboardButton(t(lang, "lang.ar"), callback_data=CB_LANG_AR)],
    ])
    await update.message.reply_html(f"{t(lang, 'lang.current', lang_name=LANG_NAMES.get(lang, lang))}\n\n{t(lang, 'lang.choose')}", reply_markup=kb)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
        game = get_active_game(s, chat_id)
    if game:
        with Session(engine) as s:
            rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0
        await update.message.reply_html(render_scoreboard(game, rounds_count, lang), reply_markup=main_keyboard(game, lang))
    else:
        await update.message.reply_text(t(lang, "start.no_game"))

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
    await update.message.reply_text(t(lang, "help.text"))

# --- New game wizard --- #
A_NAME, B_NAME = range(2)

async def newgame(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
        prev = get_active_game(s, chat_id)
    if prev:
        await update.message.reply_text(t(lang, "newgame.exists"))
        return ConversationHandler.END
    await update.message.reply_html(t(lang, "newgame.askA"))
    return A_NAME

async def set_team_a(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = (update.message.text or "Team A").strip()
    context.user_data["team_a"] = name
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
    await update.message.reply_html(t(lang, "newgame.askB"))
    return B_NAME

async def set_team_b(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    team_a = context.user_data.get("team_a", "Team A")
    team_b = (update.message.text or "Team B").strip()
    target = TARGET_SCORE_DEFAULT
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
        game = create_game(s, chat_id, team_a, team_b, target)
        rounds_count = 0
    await update.message.reply_html(render_scoreboard(game, rounds_count, lang), reply_markup=main_keyboard(game, lang))
    return ConversationHandler.END

async def cancel_wizard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
    await update.message.reply_text(t(lang, "wizard.canceled"))
    return ConversationHandler.END

# --- Dashboard callbacks --- #
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
        game = get_active_game(s, chat_id)

    # Language switches (works even without a game)
    if query.data in (CB_LANG_EN, CB_LANG_AR):
        new_lang = "en" if query.data == CB_LANG_EN else "ar"
        with Session(engine) as s:
            set_lang(s, chat_id, new_lang)
        await safe_edit(
            query,
            t(new_lang, "lang.set_ok", lang_name=LANG_NAMES[new_lang]) + "\n\n" +
            t(new_lang, "lang.current", lang_name=LANG_NAMES[new_lang]) + "\n\n" +
            t(new_lang, "lang.choose"),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(t(new_lang, "lang.en"), callback_data=CB_LANG_EN),
                 InlineKeyboardButton(t(new_lang, "lang.ar"), callback_data=CB_LANG_AR)]
            ])
        )
        return

    if not game and query.data not in (CB_LANG_EN, CB_LANG_AR, CB_NEXT_RENAME_LOSER):
        await query.edit_message_text(t(lang, "callback.no_active"))
        return

    data = query.data
    if data == CB_SHOW_SCORE and game:
        with Session(engine) as s:
            rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0
        await safe_edit(query, render_scoreboard(game, rounds_count, lang), parse_mode=ParseMode.HTML, reply_markup=main_keyboard(game, lang))
        return

    if data == CB_ADD_PHOTO_A and game:
        context.chat_data["expect_photo_for"] = 'A'
        await query.message.reply_html(t(lang, "expect.photo.A", team=game.team_a_name))
        return

    if data == CB_ADD_PHOTO_B and game:
        context.chat_data["expect_photo_for"] = 'B'
        await query.message.reply_html(t(lang, "expect.photo.B", team=game.team_b_name))
        return

    if data == CB_ADD_MANUAL and game:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(game.team_a_name, callback_data=CB_MANUAL_TEAM_A),
                                    InlineKeyboardButton(game.team_b_name, callback_data=CB_MANUAL_TEAM_B)]])
        await query.edit_message_text(t(lang, "manual.choose_team"), reply_markup=kb)
        return

    if data in (CB_MANUAL_TEAM_A, CB_MANUAL_TEAM_B) and game:
        team = 'A' if data == CB_MANUAL_TEAM_A else 'B'
        context.user_data['manual_team'] = team
        await query.edit_message_text(t(lang, "manual.prompt_points"))
        return

    if data == CB_RESET_GAME and game:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(t(lang, "reset.yes"), callback_data=CB_RESET_CONFIRM),
                                    InlineKeyboardButton(t(lang, "reset.no"), callback_data=CB_RESET_CANCEL)]])
        await query.edit_message_text(t(lang, "reset.ask"), reply_markup=kb)
        return

    if data == CB_RESET_CONFIRM and game:
        with Session(engine) as s:
            game = get_active_game(s, chat_id)
            if not game:
                await query.edit_message_text(t(lang, "score.no_game"))
                return
            reset_game_scores(s, game)
            rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0
            lang = get_lang(s, chat_id)
        await query.edit_message_text(render_scoreboard(game, rounds_count, lang), parse_mode=ParseMode.HTML, reply_markup=main_keyboard(game, lang))
        return

    if data == CB_RESET_CANCEL:
        await query.edit_message_text(t(lang, "reset.canceled"))
        return

    if data == CB_END_GAME and game:
        with Session(engine) as s:
            g = get_active_game(s, chat_id)
            if not g:
                await query.edit_message_text(t(lang, "score.no_game"))
                return
            finish_game(s, g)
        await query.edit_message_text(t(lang, "end.done"))
        return

    if data == CB_NEXT_RENAME_LOSER:
        win_name = context.chat_data.get('winner_name')
        if not win_name:
            await query.edit_message_text(t(lang, "winner.no_ctx"))
            return
        context.chat_data['await_loser_name'] = True
        await query.edit_message_text(t(lang, "winner.rename_prompt", team=win_name), parse_mode=ParseMode.HTML)
        return

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)

    # Await loser rename after win
    if context.chat_data.get('await_loser_name'):
        loser_new = (update.message.text or '').strip()
        win_name = context.chat_data.get('winner_name')
        target = TARGET_SCORE_DEFAULT
        if not loser_new or not win_name:
            await update.message.reply_text(t(lang, "manual.invalid"))
            return
        with Session(engine) as s:
            game = create_game(s, chat_id, win_name, loser_new, target)  # winner kept as Team A
            rounds_count = 0
            lang = get_lang(s, chat_id)
        context.chat_data.pop('await_loser_name', None)
        context.chat_data.pop('winner_name', None)
        await update.message.reply_html("✅ " + t(lang, "score.no_game").replace("No active game.", "New match created!"))  # light reuse
        await update.message.reply_html(render_scoreboard(game, rounds_count, lang), reply_markup=main_keyboard(game, lang))
        return

    # Manual score flow
    manual_team = context.user_data.get('manual_team')
    if manual_team:
        txt = (update.message.text or '').strip()
        try:
            points = int(txt)
        except ValueError:
            await update.message.reply_text(t(lang, "manual.invalid"))
            return
        with Session(engine) as s:
            game = get_active_game(s, chat_id)
            if not game:
                await update.message.reply_text(t(lang, "score.no_game"))
                context.user_data.pop('manual_team', None)
                return
            add_points(s, game, manual_team, points, method='manual', image_path=None)
            rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0

            winner = None
            if game.team_a_score >= game.target_score and game.team_b_score < game.target_score:
                winner = ('A', game.team_a_name)
            elif game.team_b_score >= game.target_score and game.team_a_score < game.target_score:
                winner = ('B', game.team_b_name)

        context.user_data.pop('manual_team', None)
        if winner:
            await declare_winner_and_offer_next(update, context, winner, game)
        else:
            await update.message.reply_html(render_scoreboard(game, rounds_count, lang), reply_markup=main_keyboard(game, lang))
        return

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
    team = context.chat_data.get("expect_photo_for")
    user = update.effective_user.id if update.effective_user else None

    if not team:
        await update.message.reply_text(t(lang, "photo.unexpected"))
        return

    photo = update.message.photo[-1]
    file = await photo.get_file()
    bio = io.BytesIO(); await file.download_to_memory(out=bio); bio.seek(0)
    file_bytes = np.frombuffer(bio.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    h, w = img.shape[:2] if img is not None else (0, 0)
    logger.info("photo received chat_id=%s user=%s team=%s size=%dx%d", chat_id, user, team, h, w)

    total, n_domino = detect_dominoes_and_score(img)

    with Session(engine) as s:
        game = get_active_game(s, chat_id)
        if not game:
            await update.message.reply_text(t(lang, "photo.no_game"))
            return
        ts = int(time.time())
        subdir = IMG_DIR / f"game_{game.id}"
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{ts}_{'A' if team=='A' else 'B'}.jpg"
        try:
            cv2.imwrite(str(path), img); saved = True
        except Exception:
            logger.exception("failed to save image %s", path); saved = False

        add_points(s, game, team, total, method='opencv', image_path=str(path) if saved else None)
        rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0

        winner = None
        if game.team_a_score >= game.target_score and game.team_b_score < game.target_score:
            winner = ('A', game.team_a_name)
        elif game.team_b_score >= game.target_score and game.team_a_score < game.target_score:
            winner = ('B', game.team_b_name)

    context.chat_data.pop("expect_photo_for", None)

    if n_domino == 0:
        hint = t(lang, "photo.hint.none")
    else:
        hint = t(lang, "photo.hint.some", total=total, count=n_domino)

    team_name = game.team_a_name if team == 'A' else game.team_b_name
    msg = t(lang, "photo.added", hint=hint, team=team_name)

    await update.message.reply_html(msg)

    if winner:
        await declare_winner_and_offer_next(update, context, winner, game)
    else:
        await update.message.reply_html(render_scoreboard(game, rounds_count, lang), reply_markup=main_keyboard(game, lang))

async def declare_winner_and_offer_next(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                        winner: Tuple[str, str], game: Game):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
        g = s.get(Game, game.id)
        if g and g.status != 'finished':
            finish_game(s, g)
    (team_key, team_name) = winner
    loser_name = game.team_b_name if team_key == 'A' else game.team_a_name

    context.chat_data['winner_name'] = team_name

    text = t(lang, "winner.text", team=team_name, target=game.target_score, loser=loser_name)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "winner.keep_rename"), callback_data=CB_NEXT_RENAME_LOSER)],
    ])
    await update.message.reply_html(text, reply_markup=kb)

# --- Simple commands --- #
async def score_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
        game = get_active_game(s, chat_id)
        if not game:
            await update.message.reply_text(t(lang, "score.no_game"))
            return
        rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0
    await update.message.reply_html(render_scoreboard(game, rounds_count, lang), reply_markup=main_keyboard(game, lang))

async def addscore_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
        game = get_active_game(s, chat_id)
    if not game:
        await update.message.reply_text(t(lang, "score.no_game"))
        return
    kb = InlineKeyboardMarkup([[InlineKeyboardButton(game.team_a_name, callback_data=CB_MANUAL_TEAM_A),
                                InlineKeyboardButton(game.team_b_name, callback_data=CB_MANUAL_TEAM_B)]])
    await update.message.reply_text(t(lang, "manual.choose_team"), reply_markup=kb)

async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        lang = get_lang(s, chat_id)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton(t(lang, "reset.yes"), callback_data=CB_RESET_CONFIRM),
                                InlineKeyboardButton(t(lang, "reset.no"), callback_data=CB_RESET_CANCEL)]])
    await update.message.reply_text(t(lang, "reset.ask"), reply_markup=kb)

# ========= App bootstrap ========= #
async def on_startup(app: Application):
    await set_commands(app)
    logger.info("bot started and ready")

def build_app() -> Application:
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("newgame", newgame)],
        states={
            A_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_team_a)],
            B_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_team_b)],
        },
        fallbacks=[CommandHandler("cancel", cancel_wizard)],
        name="newgame_conv",
        persistent=False,
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("lang", lang_cmd))
    app.add_handler(conv)
    app.add_handler(CommandHandler("score", score_cmd))
    app.add_handler(CommandHandler("addscore", addscore_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))

    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.post_init = on_startup
    return app

if __name__ == "__main__":
    logger.info("launching application polling...")
    application = build_app()
    application.run_polling(close_loop=False)
