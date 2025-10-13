import logging
import re
import time
import math
import random
import unicodedata
from pathlib import Path
from typing import List
import gradio as gr
import gc
from PIL import Image, ImageDraw, ImageFont
import modules.scripts as scripts
from modules.processing import process_images, Processed
from modules.sd_samplers import samplers
from modules.sd_schedulers import schedulers
from modules.shared import state


BASE_DIR = Path(scripts.basedir())
LORA_DIR = BASE_DIR / "models" / "Lora"
OUTPUT_ROOT = BASE_DIR / "outputs" / "lora_sampler_scheduler_grid_forge"
output_dir = OUTPUT_ROOT
output_dir.mkdir(parents=True, exist_ok=True)
cells_dir = output_dir / "cells"
cells_dir.mkdir(exist_ok=True)


logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\u00A0", " ")
    name = " ".join(name.strip().split())
    name = name.lower().replace(" ", "_").replace("+", "pp").replace("-", "_")
    return name


valid_samplers = [s for s in samplers if hasattr(s, "name")]
valid_schedulers = [s for s in schedulers if hasattr(s, "name")]

sampler_label_to_name = {
    getattr(s, "label", s.name): normalize_name(s.name) for s in valid_samplers
}
scheduler_label_to_name = {
    getattr(sch, "label", sch.name): normalize_name(sch.name) for sch in valid_schedulers
}

sampler_lookup = {}
for s in valid_samplers:
    name = normalize_name(s.name)
    sampler_lookup[name] = name
    label = getattr(s, "label", None)
    if label:
        sampler_lookup[normalize_name(label)] = name
    if hasattr(s, "aliases") and isinstance(s.aliases, (list, tuple)):
        for alias in s.aliases:
            sampler_lookup[normalize_name(str(alias))] = name

scheduler_lookup = {}
for sch in valid_schedulers:
    name = normalize_name(sch.name)
    scheduler_lookup[name] = name
    label = getattr(sch, "label", None)
    if label:
        scheduler_lookup[normalize_name(label)] = name
    if hasattr(sch, "aliases") and isinstance(sch.aliases, (list, tuple)):
        for alias in sch.aliases:
            scheduler_lookup[normalize_name(str(alias))] = name


label_positions = ["⬆️ Top", "⬇️ Bottom", "⬅️ Left", "➡️ Right"]

SIDE_TO_AXIS = {
    "⬆️ Top": "x",
    "⬇️ Bottom": "x",
    "⬅️ Left": "y",
    "➡️ Right": "y"
}

SAMPLER_CHOICES = list(sampler_label_to_name.keys())
SCHEDULER_CHOICES = list(scheduler_label_to_name.keys())

font_cache = {}


def get_cached_font(font_path, size):
    key = (font_path, size)
    if key in font_cache:
        return font_cache[key]
    try:
        font = ImageFont.truetype(font_path, size)
    except Exception as e:
        print(
            f"⚠️ Failed to load font '{font_path}' at size {size}, using default. Error: {e}")
        font = ImageFont.load_default()

    font_cache[key] = font
    return font


def get_cell_indices(w_idx, s_idx_row, s_idx_col,
                     orientation, n_samplers, n_schedulers, n_weights,
                     lora_label_side="⬆️ Top"):

    if orientation == "horizontal":
        row = s_idx_row * n_samplers + s_idx_col
        col = w_idx

    else:
        row = (n_weights - 1) - w_idx
        col = s_idx_row * n_samplers + s_idx_col

    return row, col


def get_next_grid_index(output_root: Path, prefix: str = "grid_") -> str:

    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)

    pattern = re.compile(
        rf"^{re.escape(prefix)}{timestamp}_(\d+)\.(?:png|webp)$"
    )

    existing = [
        f for f in output_root.iterdir()
        if f.is_file() and pattern.match(f.name)
    ]

    indices = [int(pattern.match(f.name).group(1)) for f in existing]
    next_idx = max(indices, default=0) + 1

    return f"{prefix}{timestamp}_{next_idx:03d}"


def determine_grid_orientation(label_pos_dict):
    lora_axis = SIDE_TO_AXIS.get(label_pos_dict.get("lora"), "y")
    return "horizontal" if lora_axis == "x" else "vertical"


def get_lora_files() -> List[str]:
    if not LORA_DIR.exists():
        return []

    exts = (".safetensors", ".pt")
    lora_names = {
        path.stem
        for ext in exts
        for path in LORA_DIR.rglob(f"*{ext}")
    }

    return sorted(lora_names)


def wrap_text_to_fit(text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""
    bbox_cache = {}

    def get_cached_bbox(s):
        if s not in bbox_cache:
            bbox_cache[s] = font.getbbox(s)
        return bbox_cache[s]

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = get_cached_bbox(test_line)
        if (bbox[2] - bbox[0]) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            split_word = ""
            for char in word:
                test_word = split_word + char
                bbox = get_cached_bbox(test_word)
                if (bbox[2] - bbox[0]) <= max_width:
                    split_word = test_word
                else:
                    lines.append(split_word)
                    split_word = char
            current_line = split_word if split_word else ""
    if current_line:
        lines.append(current_line)
    return lines


def compute_label_and_cell_layout(padding, cell_w, cell_h, label_side, font_path, label_text):
    if not label_text or not isinstance(label_text, str):
        label_text = " "

    font_size = 42
    font = get_cached_font(font_path, font_size)

    max_label_dim = cell_w if label_side in ["⬆️ Top", "⬇️ Bottom"] else cell_h

    for size in range(font_size, 10, -1):
        font = get_cached_font(font_path, size)
        lines = wrap_text_to_fit(label_text, font, max_label_dim)

        total_text_w = max(font.getbbox(line)[2]
                           for line in lines) if lines else 0
        if total_text_w <= max_label_dim:
            break

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    spacing = int(line_h * 0.2)

    label_text_h = line_h * len(lines) + spacing * (len(lines) - 1)
    label_text_w = max(font.getbbox(
        line)[2] - font.getbbox(line)[0] for line in lines) if lines else 0

    layout = {
        "font": font,
        "lines": lines,
        "label_text_w": label_text_w,
        "label_text_h": label_text_h,
        "label_box_w": max(label_text_w, cell_w if label_side in ["⬆️ Top", "⬇️ Bottom"] else label_text_h),
        "label_box_h": max(label_text_h, cell_h if label_side in ["⬆️ Top", "⬇️ Bottom"] else label_text_w),
    }

    return layout


def draw_labels_on_grid(draw_context, grid_image, layout, label_side, x_pos, y_pos, box_w, box_h, ui_padding):
    x, y = x_pos, y_pos
    lines = layout["lines"]
    font = layout["font"]
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    spacing = int(line_h * 0.2)
    total_text_h = layout["label_text_h"]
    total_text_w = layout["label_text_w"]

    if label_side in ["⬆️ Top", "⬇️ Bottom"]:
        y_text = y + ui_padding + (box_h - total_text_h - 2 * ui_padding) // 2
        for line in lines:
            bbox = font.getbbox(line)
            text_w = bbox[2] - bbox[0]
            x_text = x + (box_w - text_w) // 2
            draw_context.text((x_text, y_text), line,
                              font=font, fill=(0, 0, 0))
            y_text += line_h + spacing

    elif label_side in ["⬅️ Left", "➡️ Right"]:
        temp_label_img = Image.new(
            "RGBA", (total_text_w, total_text_h), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_label_img)

        y_text_temp = 0
        for line in lines:
            bbox = font.getbbox(line)
            text_w = bbox[2] - bbox[0]
            x_text_temp = (total_text_w - text_w) // 2
            temp_draw.text((x_text_temp, y_text_temp), line,
                           font=font, fill=(0, 0, 0))
            y_text_temp += line_h + spacing

        if label_side == "⬅️ Left":
            rotated_label_img = temp_label_img.transpose(Image.ROTATE_90)
        else:
            rotated_label_img = temp_label_img.transpose(Image.ROTATE_270)

        paste_x = x + (box_w - rotated_label_img.width) // 2
        paste_y = y + ui_padding + \
            (box_h - rotated_label_img.height - 2 * ui_padding) // 2

        grid_image.paste(rotated_label_img,
                         (paste_x, paste_y), rotated_label_img)


def compute_grid_dimensions(weights, samplers, schedulers, orientation):
    if orientation == "horizontal":
        rows = len(schedulers) * len(samplers)
        cols = len(weights)
    else:
        rows = len(weights)
        cols = len(schedulers) * len(samplers)
    return rows, cols


def compute_weights(min_txt, max_txt, step_txt):
    try:
        min_w = float(min_txt) if min_txt else 0.0
        max_w = float(max_txt) if max_txt else 1.0
        step_w = float(step_txt) if step_txt else 0.1
        if max_w < min_w:
            raise ValueError(
                "The maximum weight must be greater than or equal to the minimum weight.")
        if step_w <= 0:
            raise ValueError("The step value must be greater than zero.")

        weights = []
        i = 0
        while True:
            w = round(min_w + i * step_w, 6)
            if w > max_w + 1e-6:
                break
            weights.append(w)
            if math.isclose(w, max_w, rel_tol=1e-5):
                break
            i += 1

        if not weights:
            raise ValueError(
                "The specified LoRA weight range did not generate any values. Please check the min, max, and step values.")

        return weights
    except ValueError as e:
        raise ValueError(f"Invalid LoRA weight values: {e}")


def validate_label_positions(label_pos_dict, grid_type="Grid"):
    all_selected_positions = list(label_pos_dict.values())
    if len(set(all_selected_positions)) < len(all_selected_positions):
        seen = set()
        duplicates = [
            x for x in all_selected_positions if x in seen or seen.add(x)]
        if duplicates:
            raise ValueError(
                f"{grid_type}: Label conflict(s) detected on the same side: {', '.join(set(duplicates))}. "
                "Please choose different sides for the labels."
            )

    axes_used = {"x": [], "y": []}
    for label_type, position in label_pos_dict.items():
        axis = SIDE_TO_AXIS.get(position)
        if axis:
            axes_used[axis].append((label_type, position))

    if len(axes_used["x"]) == len(label_pos_dict) or len(axes_used["y"]) == len(label_pos_dict):
        print(
            f"{grid_type}: All labels are positioned on a single axis. This may lead to overcrowding.")


def create_fallback_image(width, height, sampler, scheduler, font_path, lora_name=None, weight=None):
    print(
        f"🛠️ Fallback image created for LoRA '{lora_name}' with weight {weight:.2f}")

    padding = 20
    cell_w, cell_h = width, height

    label_text_lines = ["⚠️ ERROR"]
    if lora_name and weight is not None:
        label_text_lines.append(f"LoRA: {lora_name} ({weight:.2f})")
    label_text_lines.append(f"Sampler: {sampler}")
    label_text_lines.append(f"Scheduler: {scheduler}")
    label_text = "\n".join(label_text_lines)

    layout = compute_label_and_cell_layout(
        padding, cell_w, cell_h, "⬆️ Top", font_path, label_text
    )

    total_w = cell_w + 2 * padding
    total_h = cell_h + layout["label_box_h"] + 6 + 2 * padding

    fallback = Image.new("RGB", (total_w, total_h), "lightgray")
    draw = ImageDraw.Draw(fallback)

    draw_labels_on_grid(
        draw,
        fallback,
        layout,
        "⬆️ Top",
        padding,
        padding,
        layout["label_box_w"],
        layout["label_box_h"],
        ui_padding=padding
    )

    inner = Image.new("RGB", (cell_w, cell_h), "dimgray")
    cell_x = padding
    cell_y = padding + layout["label_box_h"] + 6
    fallback.paste(inner, (cell_x, cell_y))

    return fallback


def downscale_if_needed(image, total_cells):
    canvas_w, canvas_h = image.width, image.height
    MAX_DIMENSION = 16380
    if canvas_w > MAX_DIMENSION or canvas_h > MAX_DIMENSION:
        scale = min(MAX_DIMENSION / canvas_w, MAX_DIMENSION / canvas_h)
        image = image.resize(
            (int(canvas_w * scale), int(canvas_h * scale)), Image.LANCZOS)
        print(f"🪄 Downscaled: {image.width}×{image.height}", flush=True)
    return image


def generate_safe_image(
    p,
    sampler,
    scheduler,
    weight,
    lora_name,
    font_path,
    trigger_word="",
    from_pairs=False
):
    try:
        apply_params(p, sampler, scheduler, weight, lora_name, trigger_word)

        res = process_images(p)
        img = res.images[0] if res and getattr(res, "images", None) else None
        seed_used = getattr(res, "seed", p.seed)

        if img is None:
            raise ValueError("No image returned")

    except Exception as e:
        print(f"⚠️ Generation failed: {e}")
        img = create_fallback_image(
            p.width, p.height, sampler, scheduler, font_path, lora_name, weight
        )
        seed_used = p.seed

    return img, seed_used


def safe_lookup(value, lookup_dict, label, manual=False):

    if value in lookup_dict:
        return lookup_dict[value]

    value_lower = str(value).lower().strip()
    for k, v in lookup_dict.items():
        if k.lower() == value_lower:
            if manual:
                logging.warning(
                    f"{label} '{value}' not found, replaced with '{v}'")
                print(f"⚠️ {label} '{value}' autocorrected to '{v}'")
            return v

    if manual:
        logging.warning(f"{label} '{value}' not found in list")
        print(f"⚠️ {label} '{value}' not found in list, using as-is")
    return value


def apply_params(p, sampler_name, scheduler_name, lora_weight, lora_name, trigger_word):
    current_prompt = str(p.prompt or "").strip()

    current_prompt = re.sub(
        rf"<lora:{re.escape(lora_name)}:\d+(\.\d+)?>", "", current_prompt
    )

    current_prompt = re.sub(r"[,\s]+", " ", current_prompt).strip(" ,")

    if trigger_word:
        pattern = re.compile(rf"\b{re.escape(trigger_word)}\b", re.IGNORECASE)
        current_prompt = pattern.sub("", current_prompt).strip(" ,")

    if math.isclose(lora_weight, 0.0, abs_tol=1e-6):
        p.prompt = current_prompt
        p.sampler_name = sampler_name
        p.scheduler = scheduler_name
        return p

    new_prompt_parts = [current_prompt]

    new_prompt_parts.append(f"<lora:{lora_name}:{lora_weight:.2f}>")

    if trigger_word:
        new_prompt_parts.append(trigger_word)

    clean_prompt = ", ".join(
        [part.strip(" ,") for part in new_prompt_parts if part.strip()]
    )

    p.prompt = clean_prompt
    p.sampler_name = sampler_name
    p.scheduler = scheduler_name

    return p


def validate_cells_for_grid(cells, strict=False):
    errors = []
    valid_count = 0

    for r_idx, row in enumerate(cells):
        for c_idx, cell in enumerate(row):
            if cell is None:
                errors.append(f"❌ Cell [{r_idx}, {c_idx}] is None")
            elif not isinstance(cell, Image.Image):
                errors.append(f"❌ Cell [{r_idx}, {c_idx}] is not an image")
            else:
                valid_count += 1

    if errors:
        print("\n🚫 Grid validation issues:")
        for err in errors:
            print("   " + err)
        print(f"⚠️ Valid cells: {valid_count}, Invalid cells: {len(errors)}")
        return False if strict else True

    return True


def build_grid_labels(
    weights,
    samplers,
    schedulers,
    orientation,
    label_pos_map,
    lora_name,
    mode="XY",
    pairs=None,
    trigger_word: str = "",
    show_trigger_word: bool = True
):
    labels = []

    def format_lora_label(weight):
        text = f"{lora_name}\n{weight:.2f}"
        if trigger_word and show_trigger_word:
            text += f"\n{trigger_word}"
        return text

    if mode == "XY":
        if orientation == "horizontal":
            for s_idx_row, scheduler_name in enumerate(schedulers):
                for s_idx_col, sampler_name in enumerate(samplers):
                    col_index = s_idx_row * len(samplers) + s_idx_col
                    labels.append({
                        "text": sampler_name,
                        "side": label_pos_map["sampler"],
                        "index": col_index,
                        "type": "sampler",
                        "axis": "x"
                    })
                    labels.append({
                        "text": scheduler_name,
                        "side": label_pos_map["scheduler"],
                        "index": col_index,
                        "type": "scheduler",
                        "axis": "x"
                    })
            for w_idx, weight in enumerate(weights):
                labels.append({
                    "text": format_lora_label(weight),
                    "side": label_pos_map["lora"],
                    "index": w_idx,
                    "type": "lora",
                    "axis": "y"
                })
        else:
            for w_idx, weight in enumerate(weights):
                labels.append({
                    "text": format_lora_label(weight),
                    "side": label_pos_map["lora"],
                    "index": w_idx,
                    "type": "lora",
                    "axis": "y"
                })
            for s_idx_row, scheduler_name in enumerate(schedulers):
                for s_idx_col, sampler_name in enumerate(samplers):
                    col_index = s_idx_row * len(samplers) + s_idx_col
                    labels.append({
                        "text": sampler_name,
                        "side": label_pos_map["sampler"],
                        "index": col_index,
                        "type": "sampler",
                        "axis": "x"
                    })
                    labels.append({
                        "text": scheduler_name,
                        "side": label_pos_map["scheduler"],
                        "index": col_index,
                        "type": "scheduler",
                        "axis": "x"
                    })

    elif mode == "Batch" and pairs:
        if orientation == "horizontal":
            for p_idx, pair_str in enumerate(pairs):
                sampler, scheduler = [x.strip()
                                      for x in pair_str.split(",", 1)]
                labels.append({
                    "text": sampler,
                    "side": label_pos_map["sampler"],
                    "index": p_idx,
                    "type": "sampler",
                    "axis": "y"
                })
                labels.append({
                    "text": scheduler,
                    "side": label_pos_map["scheduler"],
                    "index": p_idx,
                    "type": "scheduler",
                    "axis": "y"
                })
            for w_idx, weight in enumerate(weights):
                labels.append({
                    "text": format_lora_label(weight),
                    "side": label_pos_map["lora"],
                    "index": w_idx,
                    "type": "lora",
                    "axis": "x"
                })
        else:
            for w_idx, weight in enumerate(weights):
                labels.append({
                    "text": format_lora_label(weight),
                    "side": label_pos_map["lora"],
                    "index": w_idx,
                    "type": "lora",
                    "axis": "y"
                })
            for p_idx, pair_str in enumerate(pairs):
                sampler, scheduler = [x.strip()
                                      for x in pair_str.split(",", 1)]
                labels.append({
                    "text": sampler,
                    "side": label_pos_map["sampler"],
                    "index": p_idx,
                    "type": "sampler",
                    "axis": "x"
                })
                labels.append({
                    "text": scheduler,
                    "side": label_pos_map["scheduler"],
                    "index": p_idx,
                    "type": "scheduler",
                    "axis": "x"
                })

    seen = set()
    unique_labels = []
    for lbl in labels:
        key = (lbl["text"], lbl["side"], lbl["index"])
        if key not in seen:
            unique_labels.append(lbl)
            seen.add(key)

    return unique_labels


def compute_label_margins(
    all_labels_info,
    cell_w,
    cell_h,
    font_path,
    ui_padding,
    gap_em=0.5
):

    font = get_cached_font(font_path, 42)
    ascent, descent = font.getmetrics()
    em_px = ascent + descent

    gap_px = int(gap_em * em_px)

    margins = {
        "⬆️ Top":    ui_padding,
        "⬇️ Bottom": ui_padding,
        "⬅️ Left":   ui_padding,
        "➡️ Right":  ui_padding,
    }

    for lbl in all_labels_info:
        side = lbl["side"]
        layout = compute_label_and_cell_layout(
            padding=ui_padding,
            cell_w=cell_w,
            cell_h=cell_h,
            label_side=side,
            font_path=font_path,
            label_text=lbl["text"],
        )
        text_h = layout["label_text_h"]

        margins[side] = ui_padding + text_h + gap_px

    return margins


def draw_grouped_labels(
    labels, draw_context, grid_image, side, font_path,
    grid_w, grid_h, cell_w, cell_h,
    label_margins, ui_padding, debug_layout=False
):
    GAP = 25
    is_horizontal = side in ["⬆️ Top", "⬇️ Bottom"]

    left_margin = label_margins["⬅️ Left"]
    top_margin = label_margins["⬆️ Top"]
    right_margin = label_margins["➡️ Right"]
    bottom_margin = label_margins["⬇️ Bottom"]

    inner_w = grid_w * cell_w + (grid_w - 1) * ui_padding
    inner_h = grid_h * cell_h + (grid_h - 1) * ui_padding

    for label_data in labels:
        text = label_data["text"]
        index = label_data["index"]
        span = int(label_data.get("span", 1))

        layout = compute_label_and_cell_layout(
            padding=ui_padding,
            cell_w=cell_w,
            cell_h=cell_h,
            label_side=side,
            font_path=font_path,
            label_text=text
        )

        if is_horizontal:
            box_w = cell_w * span + (span - 1) * ui_padding
            box_h = layout["label_text_h"]
        else:
            box_w = layout["label_text_h"]
            box_h = cell_h * span + (span - 1) * ui_padding

        if side == "⬆️ Top":
            x_pos = left_margin + index * (cell_w + ui_padding)
            y_pos = top_margin - (GAP + box_h)

        elif side == "⬇️ Bottom":
            x_pos = left_margin + index * (cell_w + ui_padding)
            y_pos = top_margin + inner_h + GAP

        elif side == "⬅️ Left":
            x_pos = left_margin - (GAP + box_w)
            y_pos = top_margin + index * (cell_h + ui_padding)

        elif side == "➡️ Right":
            x_pos = left_margin + inner_w + GAP
            y_pos = top_margin + index * (cell_h + ui_padding)

        else:
            x_pos, y_pos = 0, 0

        draw_labels_on_grid(
            draw_context, grid_image, layout, side,
            x_pos, y_pos, box_w, box_h, ui_padding
        )

        if debug_layout:
            color = {
                "⬆️ Top":     (255,   0,   0),
                "⬇️ Bottom":  (255,   0,   0),
                "⬅️ Left":   (0, 128, 255),
                "➡️ Right":  (0, 128, 255),
            }[side]
            draw_context.rectangle(
                [x_pos, y_pos, x_pos + box_w, y_pos + box_h],
                outline=color, width=2
            )


def create_grid_image(
    images,
    grid_w, grid_h,
    cell_w, cell_h,
    all_labels_info,
    show_labels=True,
    font_path="Barlow-SemiBold.ttf",
    padding=10,
    gap_em=0.5,
    bg_color=(255, 255, 255)
):

    if show_labels:
        margins = compute_label_margins(
            all_labels_info=all_labels_info,
            cell_w=cell_w,
            cell_h=cell_h,
            font_path=font_path,
            ui_padding=padding,
            gap_em=gap_em
        )
    else:
        margins = {side: padding for side in [
            "⬆️ Top", "⬇️ Bottom", "⬅️ Left", "➡️ Right"]}

    inner_w = grid_w * cell_w + (grid_w - 1) * padding
    inner_h = grid_h * cell_h + (grid_h - 1) * padding

    total_w = margins["⬅️ Left"] + inner_w + margins["➡️ Right"]
    total_h = margins["⬆️ Top"] + inner_h + margins["⬇️ Bottom"]
    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(images):
        col = idx % grid_w
        row = idx // grid_w
        x0 = margins["⬅️ Left"] + col * (cell_w + padding)
        y0 = margins["⬆️ Top"] + row * (cell_h + padding)
        resized = img.resize((cell_w, cell_h))
        canvas.paste(resized, (x0, y0))
        resized.close()

    if show_labels:
        grouped = {side: [] for side in margins}
        for lbl in all_labels_info:
            grouped[lbl["side"]].append(lbl)

        for side in ["⬆️ Top", "⬇️ Bottom", "⬅️ Left", "➡️ Right"]:
            draw_grouped_labels(
                labels=grouped[side],
                draw_context=draw,
                grid_image=canvas,
                side=side,
                font_path=font_path,
                grid_w=grid_w,
                grid_h=grid_h,
                cell_w=cell_w,
                cell_h=cell_h,
                label_margins=margins,
                ui_padding=padding,
                debug_layout=False
            )

    return canvas


def save_cell_image(img, save_flag: bool, cells_dir: Path, filename: str):
    if save_flag and img:
        cells_dir.mkdir(parents=True, exist_ok=True)
        img.save(cells_dir / filename)


def create_label_radios(group_name: str, defaults=("⬆️ Top", "⬅️ Left", "➡️ Right")):

    lora_pos = gr.Radio(label_positions,
                        value=defaults[0],
                        label=f"🎨 {group_name} LoRA Label Position")
    sampler_pos = gr.Radio(label_positions,
                           value=defaults[1],
                           label=f"🗳️ {group_name} Sampler Label Position")
    scheduler_pos = gr.Radio(label_positions,
                             value=defaults[2],
                             label=f"📆 {group_name} Scheduler Label Position")
    return lora_pos, sampler_pos, scheduler_pos


def update_pairs(action: str,
                 cur: list = None,
                 sampler: str = "",
                 scheduler: str = "",
                 text: str = ""):

    if action == "parse":
        lines = [", ".join(map(str.strip, line.split(",", 1)))
                 for line in text.splitlines() if "," in line]
        unique = list(dict.fromkeys(lines))
        return unique, str(len(unique))

    if action == "add":
        if isinstance(sampler, (list, tuple)):
            sampler = sampler[0] if sampler else ""
        if isinstance(scheduler, (list, tuple)):
            scheduler = scheduler[0] if scheduler else ""

        formatted = f"{sampler.strip()}, {scheduler.strip()}"
        new_list = cur[:] if cur else []
        if sampler and scheduler and formatted not in new_list:
            new_list.append(formatted)

        text_val = "\n".join(new_list)
        return new_list, gr.update(value=text_val), str(len(new_list))

    return [], "", "0"


def safe_processed(p, images, seed, subseed, subseed_strength, info, comments):
    processed = Processed(p, images, seed, subseed,
                          subseed_strength, info, comments)
    if not isinstance(getattr(processed, "info", ""), str):
        processed.info = str(processed.info)
    if not isinstance(getattr(processed, "comments", ""), str):
        processed.comments = str(processed.comments)
    return processed


class Script(scripts.Script):
    def __init__(self):
        super().__init__()

    def title(self):
        return "🧪 LoRa x Sampler x Scheduler Grid (Forge)"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        lora_choices = get_lora_files() or []

        shared_lora = gr.State("")
        shared_trigger = gr.State("")
        shared_min_w = gr.State("")
        shared_max_w = gr.State("")
        shared_step_w = gr.State("")
        shared_pos_prompt = gr.State("")
        shared_neg_prompt = gr.State("")
        shared_seed = gr.State("")
        shared_steps = gr.State(35)
        shared_cfg = gr.State(5.0)
        shared_width = gr.State(832)
        shared_height = gr.State(1216)
        shared_padding = gr.State(20)

        mode_selector = gr.Radio(
            ["XY Grid", "Batch Grid"], value="XY Grid", label="🏁 Grid Mode")

        stop_btn = gr.Button("🛑 Stop Generation")

        stop_btn.click(lambda: state.interrupt(), [], [])

        with gr.Group() as xy_group:
            lora_label_pos_xy, sampler_label_pos_xy, scheduler_label_pos_xy = \
                create_label_radios("XY", defaults=(
                    "⬆️ Top", "⬅️ Left", "➡️ Right"))

            with gr.Row():
                with gr.Column():
                    samplers_dropdown = gr.Dropdown(
                        choices=list(sampler_label_to_name.keys()),
                        multiselect=True,
                        label="🗳️ Sampler(s)"
                    )
                    select_all_samplers_btn = gr.Button("✅ Select All")
                    clear_all_samplers_btn = gr.Button("🧹 Clear All")
                with gr.Column():
                    schedulers_dropdown = gr.Dropdown(
                        choices=list(scheduler_label_to_name.keys()),
                        multiselect=True,
                        label="📆 Scheduler(s)"
                    )
                    select_all_schedulers_btn = gr.Button("✅ Select All")
                    clear_all_schedulers_btn = gr.Button("🧹 Clear All")

            select_all_samplers_btn.click(
                lambda: gr.update(value=list(sampler_label_to_name.keys())),
                inputs=[],
                outputs=[samplers_dropdown]
            )

            clear_all_samplers_btn.click(
                lambda: gr.update(value=[]),
                inputs=[],
                outputs=[samplers_dropdown]
            )

            select_all_schedulers_btn.click(
                lambda: gr.update(value=list(scheduler_label_to_name.keys())),
                inputs=[],
                outputs=[schedulers_dropdown]
            )

            clear_all_schedulers_btn.click(
                lambda: gr.update(value=[]),
                inputs=[],
                outputs=[schedulers_dropdown]
            )

            with gr.Row():
                min_w = gr.Textbox(label="⬅️ Weight from",
                                   placeholder="e.g., 0.0")
                max_w = gr.Textbox(label="➡️ Weight to",
                                   placeholder="e.g., 1.0")
                step_w = gr.Textbox(label="📈 Weight step",
                                    placeholder="e.g., 0.5")

            lora_dropdown_xy = gr.Dropdown(
                choices=lora_choices, label="🎨 LoRA", value=None)
            trigger_word = gr.Textbox(
                label="⚡ Trigger word (optional)", placeholder="e.g.: cinematic lighting")

            pos_prompt = gr.Textbox(label="✅ Positive Prompt",
                                    placeholder="What to include", lines=3)
            neg_prompt = gr.Textbox(label="⛔ Negative Prompt",
                                    placeholder="What to avoid", lines=2)
            seed = gr.Textbox(label="🎲 Seed (optional)",
                              placeholder="Leave blank for random")

            steps = gr.Slider(1, 100, value=35, step=1, label="🚀 Steps")
            cfg_scale = gr.Slider(1.0, 30.0, value=5, step=0.1, label="🎯 CFG")

            with gr.Row():
                width = gr.Slider(256, 2048, value=832,
                                  step=64, label="↔️ Width")
                height = gr.Slider(256, 2048, value=1216,
                                   step=64, label="↕️ Height")

            padding = gr.Slider(0, 200, value=20, step=1,
                                label="📏 Cell padding")

            save_formats = gr.CheckboxGroup(choices=["WEBP", "PNG"], value=[
                                            "WEBP"], label="💾 Save As")
            show_labels = gr.Checkbox(label="📝 Show labels", value=True)
            save_cells = gr.Checkbox(
                label="💾 Save each cell", value=False)
            show_trigger_word = gr.Checkbox(
                label="🧠 Show trigger word", value=False)

        with gr.Group() as batch_group:
            lora_label_pos_b, sampler_label_pos_b, scheduler_label_pos_b = \
                create_label_radios("Batch", defaults=(
                    "⬆️ Top", "⬅️ Left", "➡️ Right"))

            with gr.Row():
                dropdown_sampler_b = gr.Dropdown(
                    choices=list(sampler_label_to_name.keys()),
                    label="🗳️ Sampler(s)"
                )
                dropdown_scheduler_b = gr.Dropdown(
                    choices=list(scheduler_label_to_name.keys()),
                    label="📆 Scheduler(s)"
                )

            with gr.Row():
                add_pair_btn = gr.Button("➕ Add pair")
                clear_pairs_btn = gr.Button("🧹 Clear All Pairs")

            pair_list = gr.Textbox(
                label="🔗 Added Pairs", placeholder="Sampler, Scheduler per line", lines=6
            )
            pair_count = gr.Textbox(
                label="🧮 Number of pairs", interactive=False
            )
            pair_state = gr.State([])

            pair_list.change(
                fn=lambda txt: update_pairs("parse", text=txt),
                inputs=[pair_list],
                outputs=[pair_state, pair_count]
            )

            add_pair_btn.click(
                fn=lambda s, sch, cur: update_pairs("add", cur, s, sch),
                inputs=[dropdown_sampler_b, dropdown_scheduler_b, pair_state],
                outputs=[pair_state, pair_list, pair_count]
            )

            clear_pairs_btn.click(
                fn=lambda: update_pairs("clear"),
                inputs=[],
                outputs=[pair_state, pair_list, pair_count]
            )

            lora_dropdown_b = gr.Dropdown(
                choices=lora_choices, label="🎨 LoRA", value=None)
            trigger_word_b = gr.Textbox(
                label="⚡ Trigger word (optional)", placeholder="e.g.: fantasy armor")

            with gr.Row():
                min_w_b = gr.Textbox(
                    label="⬅️ Weight from", placeholder="e.g., 0.5")
                max_w_b = gr.Textbox(
                    label="➡️ Weight to", placeholder="e.g., 0.7")
                step_w_b = gr.Textbox(label="📈 Weight step",
                                      placeholder="e.g., 0.1")

            pos_prompt_b = gr.Textbox(label="✅ Positive Prompt",
                                      placeholder="What to include", lines=3)
            neg_prompt_b = gr.Textbox(label="⛔ Negative Prompt",
                                      placeholder="What to avoid", lines=2)
            seed_b = gr.Textbox(label="🎲 Seed (optional)",
                                placeholder="Leave blank for random")

            steps_b = gr.Slider(1, 100, value=35, step=1, label="🚀 Steps")
            cfg_scale_b = gr.Slider(
                1.0, 30.0, value=5, step=1, label="🎯 CFG")

            with gr.Row():
                width_b = gr.Slider(256, 2048, value=832,
                                    step=1, label="↔️ Width")
                height_b = gr.Slider(256, 2048, value=1216,
                                     step=1, label="↕️ Height")

            padding_b = gr.Slider(0, 200, value=20, step=1,
                                  label="📏 Cell padding")
            save_formats_b = gr.CheckboxGroup(choices=["WEBP", "PNG"], value=[
                                              "WEBP"], label="💾 Save formats")
            show_labels_b = gr.Checkbox(label="📝 Show labels", value=True)
            save_cells_b = gr.Checkbox(
                label="💾 Save each cell", value=False)
            show_trigger_word_b = gr.Checkbox(
                label="🧠 Show trigger word", value=False)

        xy_group.visible = True
        batch_group.visible = False
        mode_selector.change(
            fn=lambda mode: (
                gr.update(visible=(mode == "XY Grid")),
                gr.update(visible=(mode == "Batch Grid"))
            ),
            inputs=[mode_selector],
            outputs=[xy_group, batch_group]
        )

        mode_selector.change(
            fn=lambda lora, trig, minw, maxw, stepw, pos, neg, seed, steps, cfg, w, h, pad: (
                gr.update(value=lora), gr.update(value=trig),
                gr.update(value=minw), gr.update(
                    value=maxw), gr.update(value=stepw),
                gr.update(value=pos), gr.update(
                    value=neg), gr.update(value=seed),
                gr.update(value=steps), gr.update(value=cfg),
                gr.update(value=w), gr.update(value=h), gr.update(value=pad),

                gr.update(value=lora), gr.update(value=trig),
                gr.update(value=minw), gr.update(
                    value=maxw), gr.update(value=stepw),
                gr.update(value=pos), gr.update(
                    value=neg), gr.update(value=seed),
                gr.update(value=steps), gr.update(value=cfg),
                gr.update(value=w), gr.update(value=h), gr.update(value=pad)
            ),
            inputs=[
                shared_lora, shared_trigger,
                shared_min_w, shared_max_w, shared_step_w,
                shared_pos_prompt, shared_neg_prompt,
                shared_seed, shared_steps, shared_cfg,
                shared_width, shared_height, shared_padding
            ],
            outputs=[
                lora_dropdown_xy, trigger_word, min_w, max_w, step_w,
                pos_prompt, neg_prompt, seed, steps, cfg_scale,
                width, height, padding,

                lora_dropdown_b, trigger_word_b, min_w_b, max_w_b, step_w_b,
                pos_prompt_b, neg_prompt_b, seed_b, steps_b, cfg_scale_b,
                width_b, height_b, padding_b
            ]
        )

        sync_map = [
            (lora_dropdown_xy, lora_dropdown_b, shared_lora),
            (trigger_word,      trigger_word_b,  shared_trigger),
            (min_w,             min_w_b,         shared_min_w),
            (max_w,             max_w_b,         shared_max_w),
            (step_w,            step_w_b,        shared_step_w),
            (pos_prompt,        pos_prompt_b,    shared_pos_prompt),
            (neg_prompt,        neg_prompt_b,    shared_neg_prompt),
            (seed,              seed_b,          shared_seed),
            (steps,             steps_b,         shared_steps),
            (cfg_scale,         cfg_scale_b,     shared_cfg),
            (width,             width_b,         shared_width),
            (height,            height_b,        shared_height),
            (padding,           padding_b,       shared_padding),
        ]

        for xy, b, shared in sync_map:
            xy.change(lambda x: x, inputs=[xy], outputs=[shared])
            b.change(lambda x: x, inputs=[b], outputs=[shared])

        return [
            mode_selector, stop_btn,
            lora_label_pos_xy, sampler_label_pos_xy, scheduler_label_pos_xy,
            samplers_dropdown, schedulers_dropdown, min_w, max_w, step_w,
            lora_dropdown_xy, trigger_word, pos_prompt, neg_prompt, seed,
            steps, cfg_scale, width, height, padding,
            save_formats, show_labels, save_cells, show_trigger_word,
            lora_label_pos_b, sampler_label_pos_b, scheduler_label_pos_b,
            dropdown_sampler_b, dropdown_scheduler_b, pair_list, pair_count,
            pair_state, lora_dropdown_b, trigger_word_b, min_w_b, max_w_b,
            step_w_b, pos_prompt_b, neg_prompt_b, seed_b, steps_b, cfg_scale_b,
            width_b, height_b, padding_b, save_formats_b, show_labels_b, save_cells_b, show_trigger_word_b,
            add_pair_btn, clear_pairs_btn
        ]

    def run(self, p, *args):
        (
            mode_selector, stop_btn,
            lora_label_pos_xy, sampler_label_pos_xy, scheduler_label_pos_xy,
            samplers_dropdown, schedulers_dropdown, min_w, max_w, step_w,
            lora_dropdown_xy, trigger_word, pos_prompt, neg_prompt, seed,
            steps, cfg_scale, width, height, padding,
            save_formats, show_labels, save_cells, show_trigger_word,
            lora_label_pos_b, sampler_label_pos_b, scheduler_label_pos_b,
            dropdown_sampler_b, dropdown_scheduler_b, pair_list, pair_count,
            pair_state, lora_dropdown_b, trigger_word_b, min_w_b, max_w_b,
            step_w_b, pos_prompt_b, neg_prompt_b, seed_b, steps_b, cfg_scale_b,
            width_b, height_b, padding_b, save_formats_b, show_labels_b, save_cells_b, show_trigger_word_b,
            add_pair_btn, clear_pairs_btn
        ) = args

        results = []
        total_generations = 0
        error_messages = []
        mode = mode_selector.value if hasattr(
            mode_selector, "value") else mode_selector

        print(f"🏁 Generation started in mode: {mode}", flush=True)

        if state.interrupted:
            print("🧹 Resetting interrupted state before start", flush=True)
            state.interrupted = False

        if mode == "XY Grid":
            if not (str(pos_prompt).strip() or str(neg_prompt).strip()):
                error_messages.append(
                    "❌ Error: Positive and Negative prompts cannot be empty. Please enter at least one of them.")
        elif mode == "Batch Grid":
            if not (pos_prompt_b or neg_prompt_b):
                error_messages.append(
                    "❌ Error: Positive and Negative prompts cannot be empty. Please enter at least one of them.")

        if mode == "XY Grid":
            if not lora_dropdown_xy:
                error_messages.append("❌ Error: Please select a LoRA model.")
            if not samplers_dropdown:
                error_messages.append(
                    "❌ Error: Please select at least one Sampler.")
            if not schedulers_dropdown:
                error_messages.append(
                    "❌ Error: Please select at least one Scheduler.")

            try:
                current_weights = compute_weights(min_w, max_w, step_w)
            except ValueError as e:
                error_messages.append(f"❌ Error with XY Grid weights: {e}")

            label_positions_map = {
                "lora": lora_label_pos_xy,
                "sampler": sampler_label_pos_xy,
                "scheduler": scheduler_label_pos_xy
            }
            try:
                validate_label_positions(label_positions_map, "XY Grid")
            except ValueError as e:
                error_messages.append(
                    str(f"Configuration Error (XY Grid): {str(e)}"))

        elif mode == "Batch Grid":
            pairs = pair_state
            if not lora_dropdown_b:
                error_messages.append("❌ Error: Please select a LoRA model.")
            if not pairs:
                error_messages.append(
                    "❌ Error: No Sampler, Scheduler pairs selected. Please add at least one pair.")

            try:
                current_weights_b = compute_weights(min_w_b, max_w_b, step_w_b)
            except ValueError as e:
                error_messages.append(f"❌ Error with Batch Grid weights: {e}")

            label_positions_map_b = {
                "lora": lora_label_pos_b,
                "sampler": sampler_label_pos_b,
                "scheduler": scheduler_label_pos_b
            }
            try:
                validate_label_positions(label_positions_map_b, "Batch Grid")
            except ValueError as e:
                error_messages.append(
                    str(f"Configuration Error (Batch Grid): {str(e)}"))

        if error_messages:
            full_error_message = "\n".join(error_messages)
            logger.error(full_error_message)

            final_seed = getattr(p, "seed", 0)
            final_subseed = getattr(p, "subseed", 0)
            final_subseed_strength = getattr(p, "subseed_strength", 0.0)

            p.seed = final_seed
            p.extra_generation_params = {"seed": final_seed}

            gc.collect()

            return safe_processed(
                p, results, final_seed, final_subseed, final_subseed_strength,
                full_error_message, ""
            )

        try:
            user_seed = seed.strip()
            sd = int(user_seed) if user_seed else random.randint(1, 2**32 - 1)
            seed_was_manual = bool(user_seed)
        except ValueError:
            sd = random.randint(1, 2**32 - 1)
            seed_was_manual = False

        if seed_was_manual:
            p.seed = sd
            p.subseed = sd
            p.subseed_strength = 0.0
        else:
            p.seed = sd
            p.subseed = sd
            p.subseed_strength = 0.0

        p.extra_generation_params = {
            "seed": sd,
            "subseed": sd,
            "subseed_strength": 0.0
        }

        final_seed = sd
        final_subseed = p.subseed
        final_subseed_strength = p.subseed_strength

        if mode == "XY Grid":
            p.prompt = pos_prompt
            p.negative_prompt = neg_prompt
            p.steps = int(steps)
            p.cfg_scale = float(cfg_scale)
            p.width = int(width)
            p.height = int(height)
        else:
            p.prompt = pos_prompt_b
            p.negative_prompt = neg_prompt_b
            p.steps = int(steps_b)
            p.cfg_scale = float(cfg_scale_b)
            p.width = int(width_b)
            p.height = int(height_b)

        font_candidates = [
            Path(__file__).resolve().parent / "Barlow-SemiBold.ttf",
            Path(__file__).resolve().parent / "fonts" / "Barlow-SemiBold.ttf",
            Path.cwd() / "Barlow-SemiBold.ttf",
        ]
        font_path = next((str(p_path)
                         for p_path in font_candidates if p_path.exists()), None)
        if not font_path:
            print(
                "⚠️ Barlow-SemiBold.ttf font not found. The default font will be used.")

        if mode == "XY Grid":
            current_samplers = samplers_dropdown
            current_schedulers = schedulers_dropdown
            current_weights = compute_weights(min_w, max_w, step_w)

            orientation = determine_grid_orientation(label_positions_map)

            if orientation == "vertical" and lora_label_pos_xy in ["⬅️ Left", "➡️ Right"]:
                current_weights = list(reversed(current_weights))

            if current_samplers:
                p.sampler_name = current_samplers[0]
            if current_schedulers:
                p.scheduler = current_schedulers[0]

            grid_rows, grid_cols = compute_grid_dimensions(
                current_weights, current_samplers, current_schedulers, orientation
            )

            print(
                f"🧩 XY Grid layout → rows={grid_rows}, cols={grid_cols}, "
                f"samplers={len(current_samplers)}, schedulers={len(current_schedulers)}, weights={len(current_weights)}"
            )

            cells_2d = [[None for _ in range(grid_cols)]
                        for _ in range(grid_rows)]

            total_generations = grid_rows * grid_cols
            current_gen_count = 0

            for w_idx, weight in enumerate(current_weights):
                for s_idx_row, scheduler_label in enumerate(current_schedulers):
                    for s_idx_col, sampler_label in enumerate(current_samplers):

                        if state.interrupted:
                            print("🛑 Interrupted by user.", flush=True)
                            logger.warning(
                                "🛑 XY Grid generation interrupted by user.")
                            break

                        sampler_name = sampler_label
                        scheduler_name = scheduler_label

                        current_gen_count += 1
                        trigger_info = f", Trigger = '{trigger_word}'" if trigger_word else ""
                        print(
                            f"🔄[{current_gen_count}/{total_generations}] "
                            f"Sampler = '{sampler_name}', Scheduler = '{scheduler_name}', "
                            f"LoRA = '{lora_dropdown_xy}', Weight = {weight:.2f}{trigger_info}, "
                            f"Seed = {p.seed}",
                            flush=True
                        )

                        img, seed_used = generate_safe_image(
                            p,
                            sampler=sampler_name,
                            scheduler=scheduler_name,
                            weight=weight,
                            lora_name=lora_dropdown_xy,
                            font_path=font_path,
                            trigger_word=trigger_word,
                            from_pairs=False
                        )

                        if seed_used and seed_used != 0:
                            final_seed = seed_used
                            final_subseed = seed_used
                            final_subseed_strength = 0.0
                            p.extra_generation_params = p.extra_generation_params or {}
                            p.extra_generation_params["seed"] = seed_used

                        row_index, col_index = get_cell_indices(
                            w_idx, s_idx_row, s_idx_col,
                            orientation,
                            len(current_samplers),
                            len(current_schedulers),
                            len(current_weights),
                            lora_label_side=lora_label_pos_xy
                        )

                        cells_2d[row_index][col_index] = img

                        if save_cells and img:
                            cell_filename = (
                                f"{lora_dropdown_xy}_W{weight:.2f}_S{sampler_name}_"
                                f"Sch{scheduler_name}_{row_index}_{col_index}.webp"
                            )
                            save_cell_image(img, save_cells,
                                            cells_dir, cell_filename)

                    if state.interrupted:
                        break
                if state.interrupted:
                    break

            if state.interrupted:
                print("🛑 Generation manually stopped by user.", flush=True)
                logger.warning("🛑 Generation stopped before completion.")
                return safe_processed(
                    p, results, p.seed, p.subseed, p.subseed_strength,
                    "🛑 Generation stopped by user", ""
                )

            all_labels = build_grid_labels(
                weights=current_weights,
                samplers=current_samplers,
                schedulers=current_schedulers,
                orientation=orientation,
                label_pos_map=label_positions_map,
                lora_name=lora_dropdown_xy,
                mode="XY",
                trigger_word=trigger_word,
                show_trigger_word=show_trigger_word
            )

            if not validate_cells_for_grid(cells_2d):
                print("❌ Grid creation aborted due to invalid cells.")

                gc.collect()

                return safe_processed(
                    p, [], p.seed, p.subseed, p.subseed_strength,
                    "Grid creation failed (invalid cells)", ""
                )

            flat_cells = [cell for row in cells_2d for cell in row if cell]

            cell_w = p.width
            cell_h = p.height

            grid = create_grid_image(
                images=flat_cells,
                grid_w=grid_cols,
                grid_h=grid_rows,
                cell_w=cell_w,
                cell_h=cell_h,
                all_labels_info=all_labels,
                show_labels=show_labels,
                font_path=font_path,
                padding=padding,
                bg_color=(255, 255, 255)
            )

            if not grid:
                print("❌ Grid image was not created. Possibly no valid cells.")

                gc.collect()

                return safe_processed(
                    p, [], p.seed, p.subseed, p.subseed_strength,
                    "Grid creation failed", ""
                )

            total_generations = len(flat_cells)
            grid = downscale_if_needed(grid, total_generations)

            grid_idx = get_next_grid_index(output_dir, prefix="xy_grid_")
            for fmt in save_formats:
                path = output_dir / f"xy_grid_{grid_idx}.{fmt.lower()}"
                grid.save(str(path))

            results.append(grid)

            print(f"✅ XY Grid saved: {grid.width}×{grid.height}", flush=True)

            gc.collect()

            return safe_processed(p, [grid], sd, sd, 0.0, "✅ XY Grid complete", "")

        elif mode == "Batch Grid":
            pairs = pair_state
            current_weights_b = compute_weights(min_w_b, max_w_b, step_w_b)

            total_batch_generations = len(current_weights_b) * len(pairs)
            current_batch_gen_count = 0

            orientation = determine_grid_orientation({
                "lora": lora_label_pos_b,
                "sampler": sampler_label_pos_b,
                "scheduler": scheduler_label_pos_b
            })

            if orientation == "horizontal":
                rows = len(pairs)
                cols = len(current_weights_b)
            else:
                rows = len(current_weights_b)
                cols = len(pairs)

            cells_for_grid = [[None for _ in range(cols)] for _ in range(rows)]
            print(f"🧩 Batch Grid layout → rows={rows}, cols={cols}, "
                  f"pairs={len(pairs)}, weights={len(current_weights_b)}")

            final_seed = p.seed
            final_subseed = p.subseed
            final_subseed_strength = p.subseed_strength

            for w_idx, weight in enumerate(current_weights_b):
                for p_idx, pair_str in enumerate(pairs):

                    if state.interrupted:
                        print("🛑 Interrupted by user.", flush=True)
                        logger.warning(
                            "🛑 Batch Grid generation interrupted by user.")
                        break

                    try:
                        raw_sampler, raw_scheduler = [
                            x.strip() for x in pair_str.split(",", 1)
                        ]

                        is_ui_sampler = raw_sampler in SAMPLER_CHOICES
                        is_ui_scheduler = raw_scheduler in SCHEDULER_CHOICES

                        sampler = raw_sampler if is_ui_sampler else safe_lookup(
                            raw_sampler, sampler_lookup, "Sampler", manual=True
                        )
                        scheduler = raw_scheduler if is_ui_scheduler else safe_lookup(
                            raw_scheduler, scheduler_lookup, "Scheduler", manual=True
                        )

                        if not sampler or not scheduler:
                            print(
                                f"⚠️ The pair '{pair_str}' was not recognized, skipped")
                            continue

                    except ValueError:
                        print(f"⚠️ Skipping invalid pair: '{pair_str}'")
                        continue

                    current_batch_gen_count += 1
                    trigger_info = f", Trigger = '{trigger_word_b}'" if trigger_word_b else ""
                    print(
                        f"🔄[{current_batch_gen_count}/{total_batch_generations}] "
                        f"Sampler = '{sampler}', Scheduler = '{scheduler}', "
                        f"LoRA = '{lora_dropdown_b}', Weight = {weight:.2f}{trigger_info}, "
                        f"Seed = {p.seed}",
                        flush=True
                    )

                    img, seed_used = generate_safe_image(
                        p,
                        sampler=sampler,
                        scheduler=scheduler,
                        weight=weight,
                        lora_name=lora_dropdown_b,
                        font_path=font_path,
                        trigger_word=trigger_word_b,
                        from_pairs=True
                    )

                    if seed_used and seed_used != 0:
                        final_seed = seed_used
                        final_subseed = seed_used
                        final_subseed_strength = 0.0
                        p.extra_generation_params = p.extra_generation_params or {}
                        p.extra_generation_params["seed"] = seed_used

                    if orientation == "horizontal":
                        row_index = p_idx
                        col_index = w_idx
                    else:
                        row_index = w_idx
                        col_index = p_idx

                    cells_for_grid[row_index][col_index] = img

                    if save_cells_b and img:
                        cell_filename = (
                            f"{lora_dropdown_b}_W{weight:.2f}_S{sampler}_Sch{scheduler}_"
                            f"{row_index}_{col_index}.webp"
                        )
                        save_cell_image(img, save_cells_b,
                                        cells_dir, cell_filename)

                if state.interrupted:
                    break

            if state.interrupted:
                print("🛑 Generation manually stopped by user.", flush=True)
                logger.warning("🛑 Generation stopped before completion.")
                return safe_processed(
                    p, results, p.seed, p.subseed, p.subseed_strength,
                    "🛑 Generation stopped by user", ""
                )

            batch_grid_labels = build_grid_labels(
                weights=current_weights_b,
                samplers=None,
                schedulers=None,
                orientation=orientation,
                label_pos_map=label_positions_map_b,
                lora_name=lora_dropdown_b,
                mode="Batch",
                pairs=pairs,
                trigger_word=trigger_word_b,
                show_trigger_word=show_trigger_word_b
            )

            if not validate_cells_for_grid(cells_for_grid):
                print("❌ Grid creation aborted due to invalid cells.")

            flat_cells = [
                cell for row in cells_for_grid for cell in row if cell]

            cell_w = p.width
            cell_h = p.height

            grid = create_grid_image(
                images=flat_cells,
                grid_w=cols,
                grid_h=rows,
                cell_w=cell_w,
                cell_h=cell_h,
                all_labels_info=batch_grid_labels,
                show_labels=show_labels_b,
                font_path=font_path,
                padding=padding_b,
                bg_color=(255, 255, 255)
            )

            if not grid:
                print("❌ Grid image was not created. Possibly no valid cells.")
                gc.collect()
                return safe_processed(
                    p, [], p.seed, p.subseed, p.subseed_strength, "Grid creation failed", ""
                )

            total_generations = len(flat_cells)
            grid = downscale_if_needed(grid, total_generations)

            grid_idx = get_next_grid_index(output_dir, prefix="batch_grid_")
            for fmt in save_formats_b:
                path = output_dir / f"batch_grid_{grid_idx}.{fmt.lower()}"
                grid.save(str(path))

            results.append(grid)

            print(
                f"✅ Batch Grid saved: {grid.width}×{grid.height}", flush=True)

            gc.collect()

            return safe_processed(
                p, [grid], sd, sd, 0.0, "✅ Batch Grid complete", ""
            )
