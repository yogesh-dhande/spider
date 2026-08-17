from PIL import Image

from spider.prepare import bbox_to_xyxy_pixels, canonical_domain, domain_split, resize_screenshot


def test_domain_is_derived_from_url() -> None:
    metadata = {"website": "fallback", "url": "https://www.Example.com/page"}
    assert canonical_domain(metadata) == "example.com"


def test_domain_split_is_stable() -> None:
    ratios = {"train": 75, "validation": 10, "test": 15}
    first = domain_split("example.com", 17, ratios)
    assert first == domain_split("example.com", 17, ratios)
    assert first in ratios


def test_resize_preserves_aspect_and_does_not_upscale() -> None:
    wide = Image.new("RGB", (1920, 1080))
    assert resize_screenshot(wide, 1280, 720).size == (1280, 720)
    tall = Image.new("RGB", (1000, 1000))
    assert resize_screenshot(tall, 1280, 720).size == (720, 720)
    small = Image.new("RGB", (640, 360))
    assert resize_screenshot(small, 1280, 720).size == (640, 360)


def test_screenspot_normalized_bbox_becomes_pixels() -> None:
    assert bbox_to_xyxy_pixels([0.25, 0.1, 0.75, 0.2], 1280, 720, "xyxy", "normalized_0_1") == [
        320.0,
        72.0,
        960.0,
        144.0,
    ]
