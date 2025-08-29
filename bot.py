#!/usr/bin/env python3
from __future__ import annotations
import os
import io
import math
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Integer, String, DateTime, ForeignKey, Text,
    select, func, BigInteger
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
    raise SystemExit("BOT_TOKEN missing. Put it in .env.")

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

DB_PATH = BASE_DIR / 'domino.db'
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}", echo=False, future=True)
Base.metadata.create_all(engine)
logger.info("DB initialized at %s", DB_PATH)

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

# ========= Computer Vision ========= #
def _resize_cap(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if max(h, w) <= MAX_IMAGE_SIDE:
        return img_bgr
    s = MAX_IMAGE_SIDE / max(h, w)
    return cv2.resize(img_bgr, (int(w * s), int(h * s)))

def white_neutral_mask_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Bright neutral areas (white domino faces)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # Bright (not too strict), and near-neutral chroma
    bright = cv2.inRange(L, 190, 255)
    a_neutral = cv2.inRange(cv2.absdiff(A, 128), 0, 14)
    b_neutral = cv2.inRange(cv2.absdiff(B, 128), 0, 14)
    mask = cv2.bitwise_and(bright, cv2.bitwise_and(a_neutral, b_neutral))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    return mask

def _rects_from_contours(cnts, img_bgr, min_area=800, fill_thresh=0.5, aspect_range=(1.3, 4.5)):
    """Turn contours into domino rect candidates with sanity checks."""
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

        # Check whiteness fill inside rect
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [box.astype(int)], -1, 255, -1)
        mean_val = cv2.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), mask=mask)[0]
        if mean_val < 100:  # not bright enough
            continue

        rects.append((box, (int(long_side), int(short_side))))

    return rects


def _merge_nearby_rects(rects, merge_dist=50):
    """Merge duplicate rects if centers are too close."""
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

    # 1. White-mask method (as before, but looser fill & ratio)
    mask = white_neutral_mask_lab(img_bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects += _rects_from_contours(cnts, img_bgr, min_area=600, fill_thresh=0.4, aspect_range=(1.2, 4.5))

    # 2. Edge fallback if not enough tiles
    if len(rects) < 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 60, 160)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects += _rects_from_contours(cnts, img_bgr, min_area=700, fill_thresh=0.3, aspect_range=(1.1, 5.0))

    # 3. Merge duplicates with larger radius
    return _merge_nearby_rects(rects, merge_dist=60)


def count_pips_on_tile(tile_bgr: np.ndarray) -> int:
    """
    Count pips on a single tile image that is roughly horizontal.
    Steps:
      • Remove a vertical center band (divider).
      • BlackHat + Otsu (dark blobs on bright).
      • Connected components with strong filters (area, circularity, extent).
    """
    if tile_bgr is None or tile_bgr.size == 0:
        return 0

    h, w = tile_bgr.shape[:2]
    # Make sure it's horizontal
    if h > w:
        tile_bgr = cv2.rotate(tile_bgr, cv2.ROTATE_90_CLOCKWISE)
        h, w = tile_bgr.shape[:2]

    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Remove center divider by zeroing a vertical band (~8% of width)
    band = max(2, int(0.08 * w))
    cx0, cx1 = w//2 - band//2, w//2 + band//2
    gray[:, cx0:cx1] = np.clip(gray[:, cx0:cx1] + 60, 0, 255)  # brighten band so divider won't trigger

    # Dark-on-light enhancement
    k = max(3, int(0.12 * min(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    # Radius bounds from tile size
    mside = max(1, min(h, w) // 2)  # half-tile for one side
    r_min = max(2, int(PIP_REL_MINR * mside))
    r_max = max(r_min + 1, int(PIP_REL_MAXR * mside))
    area_min = math.pi * (r_min * 0.8) ** 2
    area_max = math.pi * (r_max * 1.25) ** 2

    # Connected components + filters
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    keep_centers: List[Tuple[float, float]] = []
    for i in range(1, num):
        x, y, wcc, hcc, area = stats[i]
        if area < area_min or area > area_max:
            continue
        per = 2 * (wcc + hcc)
        if per <= 0:
            continue
        circ = 4 * math.pi * area / (per * per)       # circularity
        extent = area / max(1, wcc * hcc)             # filledness in bbox
        if circ < 0.60 or extent < 0.45:
            continue
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        keep_centers.append((float(xs.mean()), float(ys.mean())))

    # Merge very close centroids (reflection / double blobs)
    merged = _merge_points(keep_centers, min_dist=max(5.0, r_min * 0.8))
    return len(merged)


def detect_dominoes_and_score(img_bgr: np.ndarray, debug_save: str | None = None) -> Tuple[int, int]:
    """
    Detect domino tiles, count pips on each half.
    Returns (total_pips, n_tiles_found).
    If debug_save is provided, writes an annotated image for inspection.
    """
    t = timeit("detect_dominoes_and_score")
    if img_bgr is None or img_bgr.size == 0:
        logger.warning("detect_dominoes_and_score: empty image")
        return 0, 0

    img_bgr = _resize_cap(img_bgr)
    overlay = img_bgr.copy()

    # Detect candidate rects
    tiles = find_domino_rects(img_bgr)
    logger.info("domino candidates=%d", len(tiles))

    if not tiles:
        t("| total=0 dominoes=0")
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

        # Split halves
        H, W = warped.shape[:2]
        left = warped[:, : W // 2]
        right = warped[:, W // 2 :]

        l = count_pips_on_tile(left)
        r = count_pips_on_tile(right)
        subtotal = l + r
        total += subtotal

        # Draw debug overlay
        cv2.polylines(overlay, [box.astype(int)], True, (0, 255, 0), 2)
        cx, cy = np.mean(box[:, 0]), np.mean(box[:, 1])
        cv2.putText(overlay, f"{l}+{r}={subtotal}", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        logger.debug("tile#%d | left=%d right=%d subtotal=%d", idx, l, r, total)

    if debug_save:
        cv2.imwrite(debug_save, overlay)
        logger.info("debug image saved → %s", debug_save)

    t(f"| total={total} dominoes={len(tiles)}")
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

async def safe_edit(query, *args, **kwargs):
    try:
        return await query.edit_message_text(*args, **kwargs)
    except BadRequest as e:
        if "Message is not modified" in str(e):
            await query.answer("No changes to update.")
            logger.debug("safe_edit: not-modified chat_id=%s", query.message.chat_id)
            return None
        logger.exception("safe_edit error")
        raise

def main_keyboard(game: Game) -> InlineKeyboardMarkup:
    rows = [

        [InlineKeyboardButton("✏️ Add score manually", callback_data=CB_ADD_MANUAL)],
        [InlineKeyboardButton("📊 Show score", callback_data=CB_SHOW_SCORE),
         InlineKeyboardButton("🔄 Reset", callback_data=CB_RESET_GAME)],
        [InlineKeyboardButton("🏁 End game", callback_data=CB_END_GAME)],
    ]
    return InlineKeyboardMarkup(rows)

def progress_bar(score: int, target: int, width: int = 12) -> str:
    filled = min(width, int(round((score / max(1, target)) * width)))
    return "█" * filled + "░" * (width - filled)

def render_scoreboard(game: Game, rounds_count: int) -> str:
    t = game.target_score
    a, b = game.team_a_score, game.team_b_score
    return (
        f"<b>Domino Game</b> — Target <b>{t}</b>\n"
        f"<b>{game.team_a_name}</b>: {a}  {progress_bar(a, t)}\n"
        f"<b>{game.team_b_name}</b>: {b}  {progress_bar(b, t)}\n"
        f"Rounds: <i>{rounds_count}</i>\n"
        f"Status: <b>{game.status}</b>"
    )

# ========= DB helpers ========= #
def get_active_game(session: Session, chat_id: int) -> Optional[Game]:
    g = session.execute(select(Game).where(Game.chat_id == chat_id, Game.status == "active")).scalar_one_or_none()
    logger.debug("get_active_game chat_id=%s -> %s", chat_id, g.id if g else None)
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
    commands = [
        BotCommand("start", "Start the bot / show menu"),
        BotCommand("newgame", "Start new game (set team names)"),
        BotCommand("score", "Show current score"),
        BotCommand("addscore", "Manually add score"),
        BotCommand("reset", "Reset the current game to 0–0"),
        BotCommand("help", "How to use the bot"),
    ]
    await app.bot.set_my_commands(commands)
    logger.info("bot commands set")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    logger.info("/start chat_id=%s user=%s", chat_id, user.id if user else None)
    with Session(engine) as s:
        game = get_active_game(s, chat_id)
    if game:
        with Session(engine) as s:
            rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0
        await update.message.reply_html(render_scoreboard(game, rounds_count), reply_markup=main_keyboard(game))
    else:
        await update.message.reply_text("👋 Use /newgame to start, then add photos for each team. I’ll count the pips inside the tiles.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/newgame — create a new game and set team names\n"
        "/score — show the scoreboard\n"
        "/addscore — manually add score to a team\n"
        "/reset — reset the current game to 0–0\n"
        "Or use the buttons under the scoreboard."
    )

# --- New game wizard --- #
A_NAME, B_NAME = range(2)

async def newgame(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    logger.info("/newgame chat_id=%s", chat_id)
    with Session(engine) as s:
        prev = get_active_game(s, chat_id)
    if prev:
        await update.message.reply_text("There is already an active game. Use /reset to clear scores or /start to open the dashboard.")
        return ConversationHandler.END
    await update.message.reply_text("🆕 New game! Send <b>Team A</b> name:", parse_mode=ParseMode.HTML)
    return A_NAME

async def set_team_a(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = (update.message.text or "Team A").strip()
    context.user_data["team_a"] = name
    logger.info("set_team_a -> '%s'", name)
    await update.message.reply_text("Great. Now send <b>Team B</b> name:", parse_mode=ParseMode.HTML)
    return B_NAME

async def set_team_b(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    team_a = context.user_data.get("team_a", "Team A")
    team_b = (update.message.text or "Team B").strip()
    target = TARGET_SCORE_DEFAULT
    logger.info("set_team_b -> '%s' | creating game chat_id=%s", team_b, chat_id)
    with Session(engine) as s:
        game = create_game(s, chat_id, team_a, team_b, target)
        rounds_count = 0
    await update.message.reply_html(render_scoreboard(game, rounds_count), reply_markup=main_keyboard(game))
    return ConversationHandler.END

async def cancel_wizard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("newgame wizard canceled")
    await update.message.reply_text("Canceled.")
    return ConversationHandler.END

# --- Dashboard callbacks --- #
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    logger.info("callback chat_id=%s data=%s", chat_id, query.data)

    with Session(engine) as s:
        game = get_active_game(s, chat_id)
        if not game and query.data not in (CB_NEXT_RENAME_LOSER,):
            await query.edit_message_text("No active game. Use /newgame to start one.")
            return

    data = query.data
    if data == CB_SHOW_SCORE and game:
        with Session(engine) as s:
            rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0
        await safe_edit(query, render_scoreboard(game, rounds_count), parse_mode=ParseMode.HTML, reply_markup=main_keyboard(game))
        return

    if data == CB_ADD_PHOTO_A and game:
        context.chat_data["expect_photo_for"] = 'A'
        logger.info("expecting photo for team A chat_id=%s", chat_id)
        await query.message.reply_html(f"Send a photo for <b>{game.team_a_name}</b> now.")
        return

    if data == CB_ADD_PHOTO_B and game:
        context.chat_data["expect_photo_for"] = 'B'
        logger.info("expecting photo for team B chat_id=%s", chat_id)
        await query.message.reply_html(f"Send a photo for <b>{game.team_b_name}</b> now.")
        return

    if data == CB_ADD_MANUAL and game:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton(game.team_a_name, callback_data=CB_MANUAL_TEAM_A),
                                    InlineKeyboardButton(game.team_b_name, callback_data=CB_MANUAL_TEAM_B)]])
        await query.edit_message_text("Select team for manual points:", reply_markup=kb)
        return

    if data in (CB_MANUAL_TEAM_A, CB_MANUAL_TEAM_B) and game:
        team = 'A' if data == CB_MANUAL_TEAM_A else 'B'
        context.user_data['manual_team'] = team
        logger.info("manual add flow started team=%s chat_id=%s", team, chat_id)
        await query.edit_message_text("Send the number of points to add (e.g., 12)")
        return

    if data == CB_RESET_GAME and game:
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Yes, reset", callback_data=CB_RESET_CONFIRM),
                                    InlineKeyboardButton("❌ Cancel", callback_data=CB_RESET_CANCEL)]])
        await query.edit_message_text("Reset scores to 0–0?", reply_markup=kb)
        return

    if data == CB_RESET_CONFIRM and game:
        with Session(engine) as s:
            game = get_active_game(s, chat_id)
            if not game:
                await query.edit_message_text("No active game.")
                return
            reset_game_scores(s, game)
            rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0
        await query.edit_message_text(render_scoreboard(game, rounds_count), parse_mode=ParseMode.HTML, reply_markup=main_keyboard(game))
        return

    if data == CB_RESET_CANCEL:
        await query.edit_message_text("Reset canceled.")
        return

    if data == CB_END_GAME and game:
        with Session(engine) as s:
            game = get_active_game(s, chat_id)
            if not game:
                await query.edit_message_text("No active game.")
                return
            finish_game(s, game)
        await query.edit_message_text("Game ended. Use /newgame to start a fresh one.")
        return

    # Winner flow: keep winner, rename loser
    if data == CB_NEXT_RENAME_LOSER:
        win_name = context.chat_data.get('winner_name')
        if not win_name:
            await query.edit_message_text("No winner context found. Use /newgame to start.")
            return
        context.chat_data['await_loser_name'] = True
        await query.edit_message_text(
            f"Send the new name for the losing team (winner '<b>{win_name}</b>' will be kept).",
            parse_mode=ParseMode.HTML
        )
        return

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # Await loser rename after win
    if context.chat_data.get('await_loser_name'):
        loser_new = (update.message.text or '').strip()
        win_name = context.chat_data.get('winner_name')
        target = TARGET_SCORE_DEFAULT
        if not loser_new or not win_name:
            await update.message.reply_text("Please send a valid losing team name.")
            return
        with Session(engine) as s:
            game = create_game(s, chat_id, win_name, loser_new, target)  # winner kept as Team A
            rounds_count = 0
        context.chat_data.pop('await_loser_name', None)
        context.chat_data.pop('winner_name', None)
        await update.message.reply_html("New match created!", reply_markup=main_keyboard(game))
        await update.message.reply_html(render_scoreboard(game, rounds_count), reply_markup=main_keyboard(game))
        return

    # Manual score flow
    manual_team = context.user_data.get('manual_team')
    if manual_team:
        txt = (update.message.text or '').strip()
        logger.info("manual points received chat_id=%s team=%s text='%s'", chat_id, manual_team, txt)
        try:
            points = int(txt)
        except ValueError:
            await update.message.reply_text("Please send a valid integer, e.g., 12")
            return
        with Session(engine) as s:
            game = get_active_game(s, chat_id)
            if not game:
                await update.message.reply_text("No active game.")
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
            await update.message.reply_html(render_scoreboard(game, rounds_count), reply_markup=main_keyboard(game))
        return

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    team = context.chat_data.get("expect_photo_for")
    user = update.effective_user.id if update.effective_user else None

    if not team:
        logger.info("photo ignored (no expect_photo_for) chat_id=%s user=%s", chat_id, user)
        await update.message.reply_text("Tap a button first (Add photo → Team A/B), then send the photo.")
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
            await update.message.reply_text("No active game — use /newgame first.")
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
        hint = "I couldn’t find any domino tiles in that photo. Try a closer shot with good light."
    else:
        hint = f"Detected <b>{total}</b> pips from {n_domino} tile(s)."

    team_name = game.team_a_name if team == 'A' else game.team_b_name
    msg = f"{hint} → added to <b>{team_name}</b>."
    logger.info("opencv result chat_id=%s game_id=%s team=%s pips=%d dominoes=%d",
                chat_id, game.id, team, total, n_domino)

    await update.message.reply_html(msg)

    if winner:
        await declare_winner_and_offer_next(update, context, winner, game)
    else:
        await update.message.reply_html(render_scoreboard(game, rounds_count), reply_markup=main_keyboard(game))

async def declare_winner_and_offer_next(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                        winner: Tuple[str, str], game: Game):
    (team_key, team_name) = winner
    with Session(engine) as s:
        g = s.get(Game, game.id)
        if g and g.status != 'finished':
            finish_game(s, g)
    loser_name = game.team_b_name if team_key == 'A' else game.team_a_name

    # Save next-game context so we can rename only the loser
    context.chat_data['winner_name'] = team_name

    text = (
        f"🏆 <b>{team_name}</b> reached {game.target_score} first and wins!\n\n"
        f"Start another game? Keep winner '<b>{team_name}</b>' and rename the loser (<i>{loser_name}</i>)?"
    )
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔁 Keep winner & rename loser", callback_data=CB_NEXT_RENAME_LOSER)],
    ])
    logger.warning("winner declared game_id=%s winner=%s", game.id, team_name)
    await update.message.reply_html(text, reply_markup=kb)

# --- Simple commands --- #
async def score_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        game = get_active_game(s, chat_id)
        if not game:
            await update.message.reply_text("No active game. Use /newgame to start.")
            return
        rounds_count = s.scalar(select(func.count(Round.id)).where(Round.game_id == game.id)) or 0
    logger.debug("/score chat_id=%s game_id=%s", chat_id, game.id)
    await update.message.reply_html(render_scoreboard(game, rounds_count), reply_markup=main_keyboard(game))

async def addscore_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    with Session(engine) as s:
        game = get_active_game(s, chat_id)
    if not game:
        await update.message.reply_text("No active game. Use /newgame first.")
        return
    logger.debug("/addscore chat_id=%s game_id=%s", chat_id, game.id)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton(game.team_a_name, callback_data=CB_MANUAL_TEAM_A),
                                InlineKeyboardButton(game.team_b_name, callback_data=CB_MANUAL_TEAM_B)]])
    await update.message.reply_text("Choose team to add manual score:", reply_markup=kb)

async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.debug("/reset chat_id=%s", update.effective_chat.id)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Yes, reset", callback_data=CB_RESET_CONFIRM),
                                InlineKeyboardButton("❌ Cancel", callback_data=CB_RESET_CANCEL)]])
    await update.message.reply_text("Reset scores to 0–0?", reply_markup=kb)

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
