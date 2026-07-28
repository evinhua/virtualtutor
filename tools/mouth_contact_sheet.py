"""Contact sheets of the mouth region, one labelled cell per frame.

Picking one frame per viseme is a judgement call about mouth shape, so this
renders every candidate frame side by side with its frame number burned in.
Read the sheets, note the frame numbers you want, and feed them to
build_photo_avatar.py via picks.json (which takes seconds: frame / fps).

The head drifts across the clip, so a fixed crop box slides off the mouth
(it lands on an eye on some frames). Each cell is therefore positioned by
tracking the same rigid anchor build_photo_avatar.py aligns to -- eyes and nose
bridge -- with normalised cross-correlation at reduced resolution. That also
reports how far the head travels, which is what the aligner has to undo.

ffmpeg here is built without drawtext, so labels are drawn with a tiny 3x5
bitmap font in numpy -- no new dependencies.

Usage:
    ./.venv/bin/python tools/mouth_contact_sheet.py VIDEO \
        --box 270,555,250,150 --step 3 --grid 8x6 --out /tmp/vtavatar/sheets
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_photo_avatar import ANCHOR, decode_all, ncc_map, probe, save_png  # noqa: E402

# 3x5 bitmap digits, scaled up when drawn.
DIGITS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("110", "010", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def to_gray(frames: np.ndarray) -> torch.Tensor:
    g = (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2])
    return torch.from_numpy(g.astype(np.float32))


def downscale(gray: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return gray
    return F.avg_pool2d(gray.unsqueeze(1), factor).squeeze(1)


def track_offsets(gray_small, anchor, factor, radius, ref: int = 0):
    """Per-frame (dx, dy, score) of the anchor patch relative to frame `ref`.

    Searched at reduced resolution inside a window of +/- `radius` full-res
    pixels, which is both faster and immune to matching a similar patch
    elsewhere in the room.
    """
    ax0, ay0 = anchor["x0"] // factor, anchor["y0"] // factor
    ax1, ay1 = anchor["x1"] // factor, anchor["y1"] // factor
    r = max(1, radius // factor)
    template = gray_small[ref, ay0:ay1, ax0:ax1]

    n, h, w = gray_small.shape
    sy0, sy1 = max(0, ay0 - r), min(h, ay1 + r)
    sx0, sx1 = max(0, ax0 - r), min(w, ax1 + r)
    out = []
    for i in range(n):
        region = gray_small[i, sy0:sy1, sx0:sx1]
        m = ncc_map(region, template)
        idx = int(torch.argmax(m))
        py, px = divmod(idx, m.shape[1])
        dx = (sx0 + px - ax0) * factor
        dy = (sy0 + py - ay0) * factor
        out.append((dx, dy, float(m[py, px])))
    return out


def draw_label(cell: np.ndarray, text: str, scale: int = 2):
    """Stamp `text` into the top-left of an (H,W,3) uint8 cell."""
    gw, gh = 3 * scale, 5 * scale
    pad = scale
    box_h, box_w = gh + 2 * pad, len(text) * (gw + pad) + pad
    cell[:box_h, :box_w] = (0, 0, 0)
    x = pad
    for ch in text:
        rows = DIGITS.get(ch)
        if rows:
            for r, row in enumerate(rows):
                for c, bit in enumerate(row):
                    if bit == "1":
                        y0, x0 = pad + r * scale, x + c * scale
                        cell[y0:y0 + scale, x0:x0 + scale] = (255, 235, 0)
        x += gw + pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--box", required=True, help="mouth region on frame 0 as x0,y0,w,h")
    ap.add_argument("--step", type=int, default=3, help="frame stride")
    ap.add_argument("--frames", default=None,
                    help="explicit comma-separated frame numbers (ignores --step)")
    ap.add_argument("--grid", default="8x6", help="cells per sheet, COLSxROWS")
    ap.add_argument("--cell", type=int, default=150, help="cell width in px")
    ap.add_argument("--out", default="/tmp/vtavatar/sheets")
    ap.add_argument("--no-track", action="store_true",
                    help="use a fixed box instead of following the head")
    ap.add_argument("--search", type=int, default=110,
                    help="tracking search radius in full-res pixels")
    ap.add_argument("--downscale", type=int, default=2,
                    help="resolution divisor used while tracking")
    ap.add_argument("--ref", type=int, default=0,
                    help="reference frame whose head pose counts as canonical")
    ap.add_argument("--near", type=int, default=None,
                    help="keep only frames whose head is within this many px "
                         "of the reference pose")
    ap.add_argument("--min-ncc", type=float, default=None,
                    help="keep only frames whose anchor match is at least this "
                         "good (a proxy for similar rotation and scale)")
    args = ap.parse_args()

    bx, by, bw, bh = (int(v) for v in args.box.split(","))
    cols, rows = (int(v) for v in args.grid.lower().split("x"))
    w, h, nb, dur = probe(args.video)
    frames = decode_all(args.video, w, h)
    n = frames.shape[0]
    fps = n / dur
    print(f"{n} frames, {fps:.3f} fps, mouth box=({bx},{by}) {bw}x{bh}")

    idx = ([int(v) for v in args.frames.split(",")] if args.frames
           else list(range(0, n, args.step)))

    if args.no_track:
        offsets = {i: (0, 0, 1.0) for i in idx}
    else:
        gray_small = downscale(to_gray(frames), args.downscale)
        tracked = track_offsets(gray_small, ANCHOR, args.downscale, args.search,
                                ref=args.ref)
        dxs = [o[0] for o in tracked]
        dys = [o[1] for o in tracked]
        scores = [o[2] for o in tracked]
        print(f"reference pose: frame {args.ref}")
        print(f"head travel over the clip: dx {min(dxs):+d}..{max(dxs):+d} px, "
              f"dy {min(dys):+d}..{max(dys):+d} px")
        print(f"anchor ncc: min {min(scores):.3f}, mean {np.mean(scores):.3f}")
        still = [i for i, (dx, dy, _) in enumerate(tracked)
                 if abs(dx) <= 8 and abs(dy) <= 8]
        print(f"frames within 8 px of the reference pose: {len(still)}/{n}")

        if args.near is not None or args.min_ncc is not None:
            near = args.near if args.near is not None else 10 ** 6
            min_ncc = args.min_ncc if args.min_ncc is not None else -1.0
            kept = [i for i in idx
                    if abs(tracked[i][0]) <= near and abs(tracked[i][1]) <= near
                    and tracked[i][2] >= min_ncc]
            print(f"near-canonical filter (<= {near} px, ncc >= {min_ncc}): "
                  f"{len(kept)}/{len(idx)} candidate frames kept")
            idx = kept
        offsets = {i: tracked[i] for i in idx}

        if args.frames:
            # Explicit picks: show how far each one is from the canonical pose,
            # since a pick close to it needs the least warping to line up.
            print("pose deviation of the listed frames:")
            for i in idx:
                dx, dy, sc = tracked[i]
                print(f"  frame {i:4}  dx={dx:+4d} dy={dy:+4d} ncc={sc:.3f}")

    if not idx:
        print("no frames left after filtering")
        return

    cells = []
    for i in idx:
        dx, dy, _ = offsets[i]
        y0 = int(np.clip(by + dy, 0, h - bh))
        x0 = int(np.clip(bx + dx, 0, w - bw))
        cells.append(frames[i, y0:y0 + bh, x0:x0 + bw])
    batch = torch.from_numpy(np.stack(cells).astype(np.float32)).permute(0, 3, 1, 2)
    cell_w = args.cell
    cell_h = int(round(cell_w * bh / bw))
    small = F.interpolate(batch, size=(cell_h, cell_w), mode="area")
    rendered = small.clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = cols * rows
    written = []
    for s in range(0, len(idx), per_sheet):
        chunk = idx[s:s + per_sheet]
        sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for k, frame_no in enumerate(chunk):
            cell = rendered[s + k].copy()
            draw_label(cell, str(frame_no))
            r, c = divmod(k, cols)
            sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell
        path = out_dir / f"sheet{s // per_sheet:02d}.png"
        save_png(sheet, path)
        written.append((path.name, chunk[0], chunk[-1]))

    for name, first, last in written:
        print(f"  {name}: frames {first}..{last}")
    print(f"\n{len(written)} sheets in {out_dir} (t = frame / {fps:.3f})")


if __name__ == "__main__":
    main()
