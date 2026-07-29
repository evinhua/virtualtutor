"""Tests for the photo avatar sprite set in web/avatar.

The bundled sprites are committed, but they are still generated output
(tools/build_photo_avatar.py) and a user may delete them to get the drawn face
back, so every test skips when they are absent. What is worth checking is the
contract between the generator and web/app.js: the viseme names must be exactly
the ones the timeline can emit, the patch must sit inside the base image, and
every declared image must exist at the size the renderer assumes.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import visemes

AVATAR = Path(__file__).resolve().parents[1] / "web" / "avatar"
MANIFEST = AVATAR / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason="no photo avatar built (run tools/build_photo_avatar.py)")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def image_size(path: Path):
    """(width, height) via ffprobe, which the build already depends on."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def test_manifest_declares_patch_mode(manifest):
    assert manifest["mode"] == "patch"
    assert manifest["base"], "patch mode needs a base face"
    assert (AVATAR / manifest["base"]).is_file()


def test_visemes_cover_every_shape_the_timeline_can_emit(manifest):
    produced = set(visemes.PHONEME_TO_VISEME.values()) | {visemes.NEUTRAL}
    for parts in visemes.DIPHTHONG_PARTS.values():
        produced.update(parts)
    missing = produced - set(manifest["images"])
    assert not missing, f"no sprite for {sorted(missing)}"


def test_no_sprite_for_an_unknown_viseme(manifest):
    produced = set(visemes.PHONEME_TO_VISEME.values()) | {visemes.NEUTRAL}
    for parts in visemes.DIPHTHONG_PARTS.values():
        produced.update(parts)
    assert not set(manifest["images"]) - produced


def test_every_declared_image_exists(manifest):
    for viseme, name in manifest["images"].items():
        assert (AVATAR / name).is_file(), f"{viseme} -> {name} missing"


def test_patch_lies_inside_the_base_image(manifest):
    p, size = manifest["patch"], manifest["size"]
    assert p["x"] >= 0 and p["y"] >= 0
    assert p["x"] + p["w"] <= size
    assert p["y"] + p["h"] <= size


def test_patch_is_the_mouth_ellipse_clipped_to_the_image(manifest):
    """app.js masks inside the patch, so the patch must cover the feathered
    ellipse -- except where the ellipse runs past the image edge, which it does
    at the bottom (the mask is still partly opaque at the chin, and there is no
    seam to hide because that is the boundary of the picture).
    """
    m, p, size = manifest["mouth"], manifest["patch"], manifest["size"]
    import math
    want = {
        "x": max(0, math.floor(m["cx"] - m["rx"] - m["feather"])),
        "y": max(0, math.floor(m["cy"] - m["ry"] - m["feather"])),
    }
    want["w"] = min(size, math.ceil(m["cx"] + m["rx"] + m["feather"])) - want["x"]
    want["h"] = min(size, math.ceil(m["cy"] + m["ry"] + m["feather"])) - want["y"]
    assert p == want


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe not installed")
def test_image_dimensions_match_the_manifest(manifest):
    size = manifest["size"]
    assert image_size(AVATAR / manifest["base"]) == (size, size)
    expected = (manifest["patch"]["w"], manifest["patch"]["h"])
    for name in manifest["images"].values():
        assert image_size(AVATAR / name) == expected, name


def test_sprites_stay_small_enough_to_load_instantly(manifest):
    """A patch set is the whole point: whole faces came to ~2.3 MB."""
    total = sum(p.stat().st_size for p in AVATAR.iterdir())
    assert total < 600_000, f"{total / 1024:.0f} KB of avatar assets"


def test_base_frame_is_recorded_for_reproducibility(manifest):
    assert isinstance(manifest["base_frame"], int)
    assert manifest["frames"]["sil"] == manifest["base_frame"]
