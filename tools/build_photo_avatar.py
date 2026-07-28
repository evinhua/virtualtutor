"""Build a photo-sprite avatar: one aligned image per viseme, from a video clip.

The clip is ordinary speech recorded hand-held, so the head changes position,
scale and tilt between frames. Swapping such frames directly makes the head
jump, which looks far worse than an imprecise mouth. So every exported frame is
warped to one canonical head pose first.

Alignment uses no face-landmark library. Instead it template-matches a patch
that speech does not deform -- the eyes and nose bridge -- searching over
rotation and scale, and normalised cross-correlation for translation. The
resulting similarity transform is applied with torch's grid_sample. torch is
already a dependency (Silero VAD) and ffmpeg is already required, so this adds
nothing to requirements.txt.

Alignment fixes the head, but not the rest of the face: the eyes blink and the
gaze wanders between frames, so a sprite set of whole faces blinks and glances
around as it speaks. By default only the mouth region is taken from each pick
and blended into one base face through a feathered ellipse, so eyes, hair and
background are identical in every sprite and just the mouth moves. Pass
--no-composite to export whole aligned faces instead.

Usage:
    ./.venv/bin/python tools/build_photo_avatar.py VIDEO --picks picks.json \
        --out web/avatar
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --- Geometry, measured from the source clip (720x1280 coordinates) ---------
# Anchor: eyes + nose bridge. Rigid under speech, unlike the mouth and jaw.
ANCHOR = dict(x0=240, y0=430, x1=490, y1=610)
# Square output crop centred between the eyes and the mouth. Stops well above
# the burned-in subtitles, which begin around y=930.
CROP = dict(x0=100, y0=250, size=520)
# Mouth region inside CROP, in CROP-relative pixels: the ellipse that gets
# transplanted from each pick onto the base face. Wide enough to include the
# nasolabial folds and chin movement, which move with the mouth.
MOUTH = dict(cx=270, cy=390, rx=175, ry=125, feather=38)

# Pose search grid. Both ranges include the identity (1.0 / 0 deg) exactly, so
# a frame already in the canonical pose aligns to itself, and both are wide
# enough that the winner is an interior point rather than the grid edge -- an
# edge win means the real pose is outside the grid and the crop will be off.
SCALES = np.linspace(0.80, 1.20, 17)      # step 0.025
ANGLES_DEG = np.linspace(-10.0, 10.0, 11)  # step 2 deg


def probe(video: str):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-show_entries", "format=duration", "-of", "json", video],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    st = info["streams"][0]
    return int(st["width"]), int(st["height"]), int(st["nb_frames"]), float(info["format"]["duration"])


def decode_all(video: str, w: int, h: int) -> np.ndarray:
    """Decode the whole clip to a uint8 array (N, H, W, 3)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = frames.size // (w * h * 3)
    return frames[: n * w * h * 3].reshape(n, h, w, 3)


def to_gray(frames: np.ndarray) -> torch.Tensor:
    g = (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2])
    return torch.from_numpy(g.astype(np.float32))


def gray_frame(frame: np.ndarray) -> torch.Tensor:
    """Grayscale one (H,W,3) uint8 frame.

    Converting the whole clip at once would cost another 2 GB of float32 for a
    22 s 720p clip, and only a handful of frames are ever aligned.
    """
    return to_gray(frame[None])[0]


def similarity_grid(shape, s: float, deg: float, tx: float, ty: float, out_hw):
    """Sampling grid that reads a source image at the given similarity transform.

    Maps output (canonical) pixel coordinates to source coordinates:
        src = R(deg) * (dst - c) / s + c + t
    Returned in the normalised [-1, 1] form grid_sample expects.
    """
    H, W = shape
    OH, OW = out_hw
    cy, cx = H / 2.0, W / 2.0
    ys, xs = torch.meshgrid(torch.arange(OH, dtype=torch.float32),
                            torch.arange(OW, dtype=torch.float32), indexing="ij")
    th = math.radians(deg)
    cos, sin = math.cos(th), math.sin(th)
    dx, dy = xs - cx, ys - cy
    sx = (cos * dx - sin * dy) / s + cx + tx
    sy = (sin * dx + cos * dy) / s + cy + ty
    gx = sx / (W - 1) * 2 - 1
    gy = sy / (H - 1) * 2 - 1
    return torch.stack([gx, gy], dim=-1).unsqueeze(0)


def warp(img: torch.Tensor, s, deg, tx, ty, out_hw=None):
    """img: (C,H,W) float -> warped (C,OH,OW)."""
    C, H, W = img.shape
    out_hw = out_hw or (H, W)
    grid = similarity_grid((H, W), s, deg, tx, ty, out_hw)
    return F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                         padding_mode="border", align_corners=True)[0]


def ncc_map(image: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Normalised cross-correlation of `template` over `image` (both 2-D)."""
    t = template - template.mean()
    t = t / (t.norm() + 1e-8)
    th, tw = t.shape
    img = image.unsqueeze(0).unsqueeze(0)
    ones = torch.ones(1, 1, th, tw)
    n = th * tw
    local_sum = F.conv2d(img, ones)
    local_sq = F.conv2d(img * img, ones)
    local_mean = local_sum / n
    var = local_sq - n * local_mean * local_mean
    denom = torch.sqrt(torch.clamp(var, min=1e-8))
    corr = F.conv2d(img, t.unsqueeze(0).unsqueeze(0))
    return ((corr - local_mean * t.sum()) / denom)[0, 0]


def ncc_best_near(image, template, top_left, radius):
    """Best NCC position for `template` within `radius` px of `top_left`.

    Correlating a 250x180 template over a whole 720x1280 frame needs tens of
    gigabytes once conv2d expands it, and it is pointless work: the head never
    leaves a small neighbourhood. Searching a window keeps memory bounded and
    the match honest.

    Returns (score, dx, dy) where dx, dy are offsets from `top_left`.
    """
    h, w = image.shape
    th, tw = template.shape
    ax, ay = top_left
    y0, y1 = max(0, ay - radius), min(h, ay + th + radius)
    x0, x1 = max(0, ax - radius), min(w, ax + tw + radius)
    m = ncc_map(image[y0:y1, x0:x1], template)
    idx = int(torch.argmax(m))
    py, px = divmod(idx, m.shape[1])
    return float(m[py, px]), (x0 + px) - ax, (y0 + py) - ay


def pyramid_down(gray: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return gray
    return F.avg_pool2d(gray.unsqueeze(0).unsqueeze(0), factor)[0, 0]


def estimate_pose(gray: torch.Tensor, template: torch.Tensor, anchor_xy,
                  coarse: int = 4, radius: int = 150):
    """Best (score, s, deg, tx, ty) aligning `gray` to the reference template.

    Two passes, because the scale/rotation grid is 56 candidates and a full-res
    correlation each would be both slow and memory-hungry:
      1. coarse -- every (scale, angle) on a `coarse`-times downscaled image,
         translation searched in a window around the anchor;
      2. fine -- the winning (scale, angle) re-searched at full resolution in a
         small window around the coarse translation.
    """
    ax, ay = anchor_xy
    small = pyramid_down(gray, coarse)
    t_small = pyramid_down(template, coarse)
    anchor_small = (ax // coarse, ay // coarse)

    best = (-1e9, 1.0, 0.0, 0.0, 0.0)
    for s in SCALES:
        for deg in ANGLES_DEG:
            # Undo the candidate rotation/scale, then search translation.
            cand = warp(small.unsqueeze(0), float(s), float(deg), 0.0, 0.0)[0]
            score, dx, dy = ncc_best_near(cand, t_small, anchor_small,
                                          max(1, radius // coarse))
            if score > best[0]:
                best = (score, float(s), float(deg), dx * coarse, dy * coarse)

    _, s, deg, tx0, ty0 = best
    cand = warp(gray.unsqueeze(0), s, deg, 0.0, 0.0)[0]
    score, dx, dy = ncc_best_near(cand, template,
                                  (ax + int(tx0), ay + int(ty0)), coarse * 2)
    return score, s, deg, float(tx0 + dx), float(ty0 + dy)


def refine_translation(gray, template, anchor_xy, s, deg, tx, ty, radius=6):
    """Re-search translation at full resolution around a known offset."""
    ax, ay = anchor_xy
    cand = warp(gray.unsqueeze(0), s, deg, 0.0, 0.0)[0]
    score, dx, dy = ncc_best_near(cand, template,
                                  (ax + int(round(tx)), ay + int(round(ty))), radius)
    return float(tx + dx), float(ty + dy), score



def save_png(rgb: np.ndarray, path: Path):
    save_image(rgb, path)


def save_image(rgb: np.ndarray, path: Path, quality: int = 3):
    """Write an (H,W,3) uint8 array to `path`; format follows the suffix.

    JPEG matters here: the sprites are photographic, and a set of 14 PNG faces
    is ~2.3 MB where the same set as JPEG patches is a few hundred KB.
    """
    h, w, _ = rgb.shape
    cmd = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-i", "-"]
    if path.suffix.lower() in (".jpg", ".jpeg"):
        cmd += ["-q:v", str(quality)]
    cmd += ["-frames:v", "1", str(path), "-y"]
    subprocess.run(cmd, input=np.ascontiguousarray(rgb).tobytes(), check=True)


def resolve_pick(value, fps: float, n: int) -> int:
    """Frame index for a picks.json value.

    Accepts seconds (a number) or an exact frame index as the string "#123".
    Frame indices are what the contact sheets print, and they survive the
    round-trip through fps without rounding surprises.
    """
    if isinstance(value, str) and value.startswith("#"):
        idx = int(value[1:])
    else:
        idx = int(round(float(value) * fps))
    return min(n - 1, max(0, idx))


def mouth_mask(size: int, mouth=MOUTH) -> torch.Tensor:
    """Feathered elliptical mask (1 inside the mouth region, 0 outside).

    The feather is a smooth ramp over the ellipse boundary, so the transplanted
    mouth fades into the base face instead of showing a visible seam.
    """
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                            torch.arange(size, dtype=torch.float32), indexing="ij")
    # Radial distance in ellipse units: 1.0 exactly on the boundary.
    d = torch.sqrt(((xs - mouth["cx"]) / mouth["rx"]) ** 2
                   + ((ys - mouth["cy"]) / mouth["ry"]) ** 2)
    # Convert the feather width in pixels to ellipse units via the smaller radius.
    ramp = max(1e-6, mouth["feather"] / min(mouth["rx"], mouth["ry"]))
    m = torch.clamp((1.0 + ramp - d) / ramp, 0.0, 1.0)
    # Smoothstep for a soft, banding-free edge.
    return m * m * (3.0 - 2.0 * m)


def aligned_crop(frames, idx, template, anchor_xy):
    """Warp frame `idx` to the canonical pose and return (crop, stats).

    crop is a float (3, CROP.size, CROP.size) tensor at source resolution.
    """
    gray = gray_frame(frames[idx])
    score, s, deg, tx, ty = estimate_pose(gray, template, anchor_xy)
    tx, ty, score = refine_translation(gray, template, anchor_xy, s, deg, tx, ty)
    rgb = torch.from_numpy(frames[idx].astype(np.float32)).permute(2, 0, 1)
    aligned = warp(rgb, s, deg, tx, ty)
    c = CROP
    crop = aligned[:, c["y0"]:c["y0"] + c["size"], c["x0"]:c["x0"] + c["size"]]
    return crop, (score, s, deg, tx, ty)


def to_uint8(crop: torch.Tensor, size: int) -> np.ndarray:
    """Resize a (3,S,S) float crop to size x size and return (H,W,3) uint8."""
    img = F.interpolate(crop.unsqueeze(0), size=(size, size),
                        mode="bilinear", align_corners=False)[0]
    return img.clamp(0, 255).byte().permute(1, 2, 0).contiguous().numpy()


def patch_box(mouth, size: int):
    """Bounding box of the feathered mouth ellipse, clipped to the image."""
    x0 = max(0, int(math.floor(mouth["cx"] - mouth["rx"] - mouth["feather"])))
    y0 = max(0, int(math.floor(mouth["cy"] - mouth["ry"] - mouth["feather"])))
    x1 = min(size, int(math.ceil(mouth["cx"] + mouth["rx"] + mouth["feather"])))
    y1 = min(size, int(math.ceil(mouth["cy"] + mouth["ry"] + mouth["feather"])))
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--picks", required=True,
                    help='JSON: {viseme: seconds} or {viseme: "#frame"}')
    ap.add_argument("--out", default="web/avatar")
    ap.add_argument("--reference", default=None,
                    help='frame defining the canonical pose and the base face: '
                         'seconds or "#frame" (default: the "sil" pick)')
    ap.add_argument("--size", type=int, default=384, help="exported image size")
    ap.add_argument("--format", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--quality", type=int, default=3,
                    help="JPEG quality, 1 (best) to 31")
    ap.add_argument("--whole-face", action="store_true",
                    help="export 14 complete faces instead of one base face "
                         "plus mouth patches (larger, no client-side masking)")
    args = ap.parse_args()

    picks = json.load(open(args.picks))
    w, h, nb, dur = probe(args.video)
    print(f"video {w}x{h}, {nb} frames, {dur:.2f}s")
    frames = decode_all(args.video, w, h)
    n = frames.shape[0]
    fps = n / dur
    print(f"decoded {n} frames ({fps:.3f} fps effective)")

    ref = args.reference if args.reference is not None else picks.get(
        "sil", next(iter(picks.values())))
    ref_idx = resolve_pick(ref, fps, n)
    print(f"canonical pose + base face from frame {ref_idx} (t={ref_idx / fps:.2f}s)")

    template = gray_frame(frames[ref_idx])[
        ANCHOR["y0"]:ANCHOR["y1"], ANCHOR["x0"]:ANCHOR["x1"]].clone()
    anchor_xy = (ANCHOR["x0"], ANCHOR["y0"])

    base, base_stats = aligned_crop(frames, ref_idx, template, anchor_xy)
    base_img = to_uint8(base, args.size)

    scale = args.size / CROP["size"]
    mouth_out = {k: round(v * scale, 1) for k, v in MOUTH.items()}
    box = patch_box(mouth_out, args.size)
    mask = mouth_mask(CROP["size"]).unsqueeze(0) if args.whole_face else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format
    manifest, report = {}, []

    if not args.whole_face:
        save_image(base_img, out_dir / f"base.{ext}", args.quality)

    for viseme, pick in picks.items():
        idx = resolve_pick(pick, fps, n)
        crop, (score, s, deg, tx, ty) = aligned_crop(frames, idx, template, anchor_xy)
        if args.whole_face:
            # Everything but the mouth comes from the base face, so eyes, hair
            # and background are identical in every sprite.
            out = to_uint8(base * (1.0 - mask) + crop * mask, args.size)
        else:
            # Only the mouth rectangle travels; the browser feathers it in.
            full = to_uint8(crop, args.size)
            out = full[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]
        path = out_dir / f"{viseme}.{ext}"
        save_image(out, path, args.quality)

        manifest[viseme] = path.name
        report.append((viseme, idx, score, s, deg, tx, ty))
        edge = ("  <- at search-grid edge, pose may be outside the grid"
                if s in (SCALES[0], SCALES[-1]) or deg in (ANGLES_DEG[0], ANGLES_DEG[-1])
                else "")
        print(f"  {viseme:4} frame {idx:4} t={idx / fps:5.2f} ncc={score:.3f} "
              f"scale={s:.3f} rot={deg:+.1f} shift=({tx:+.0f},{ty:+.0f}){edge}")

    (out_dir / "manifest.json").write_text(json.dumps({
        "size": args.size,
        "source": Path(args.video).name,
        "mode": "whole-face" if args.whole_face else "patch",
        "base": None if args.whole_face else f"base.{ext}",
        "base_frame": ref_idx,
        # Mouth geometry and patch placement in exported-image pixels. The
        # renderer needs both to feather a patch onto the base face.
        "mouth": mouth_out,
        "patch": None if args.whole_face else box,
        "frames": {v: r[1] for v, r in zip(manifest, report)},
        "images": manifest,
    }, indent=1) + "\n")

    total = sum(p.stat().st_size for p in out_dir.iterdir())
    print(f"\nwrote {len(manifest)} images + manifest.json to {out_dir} "
          f"({total / 1024:.0f} KB total)")
    if not args.whole_face:
        print(f"patch {box['w']}x{box['h']} px at ({box['x']},{box['y']}) "
              f"of a {args.size}x{args.size} base face")

    print(f"alignment: mean ncc {np.mean([r[2] for r in report]):.3f} "
          f"(min {min(r[2] for r in report):.3f}), "
          f"scale range {min(r[3] for r in report):.3f}-{max(r[3] for r in report):.3f}, "
          f"rotation range {min(r[4] for r in report):+.1f}..{max(r[4] for r in report):+.1f} deg, "
          f"shift up to ({max(abs(r[5]) for r in report):.0f},"
          f"{max(abs(r[6]) for r in report):.0f}) px")
    print(f"base frame residual: ncc={base_stats[0]:.3f} "
          f"(1.000 means the reference matched itself exactly)")


if __name__ == "__main__":
    main()
