"""
Download SkyFinder camera images and/or metadata.

Camera zips:    https://cs.valdosta.edu/~rpmihail/skyfinder/images/<CamId>.zip
Metadata CSV:   https://cs.valdosta.edu/~rpmihail/skyfinder/analysis/complete_table_with_mcr.csv

Usage:
    # Download metadata only
    python scripts/download_images.py --metadata_only

    # Download all camera images
    python scripts/download_images.py --images_dir data/images

    # Download + resize (short side = 256 px, JPEG q90, overwrites in place)
    python scripts/download_images.py --images_dir data/images --resize 256
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

METADATA_URL = ("https://cs.valdosta.edu/~rpmihail/skyfinder/"
                "analysis/complete_table_with_mcr.csv")
IMAGE_BASE_URL = "https://cs.valdosta.edu/~rpmihail/skyfinder/images"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def parse_args():
    p = argparse.ArgumentParser(
        description="Download SkyFinder metadata and/or camera images.")
    p.add_argument("--metadata", default="data/complete_table_with_mcr.csv",
                   help="Path to save / find the metadata CSV")
    p.add_argument("--metadata_only", action="store_true",
                   help="Download only the metadata CSV, skip images")
    p.add_argument("--images_dir", default="data/images",
                   help="Root directory for extracted camera images")
    p.add_argument("--resize", type=int, default=None,
                   help="After extracting, resize so short side = N px "
                        "(JPEG q90, overwrite in place)")
    return p.parse_args()


def download_file(url, dest, retries=MAX_RETRIES):
    """Download url to dest with retries. Returns True on success."""
    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading {url} (attempt {attempt}/{retries})...")
            urlretrieve(url, dest)
            return True
        except (URLError, HTTPError, OSError) as e:
            print(f"  Failed: {e}")
            if attempt < retries:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
    return False


def download_metadata(metadata_path):
    """Download metadata CSV if not already present."""
    if os.path.isfile(metadata_path):
        print(f"Metadata already exists: {metadata_path}")
        return True
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    print(f"Downloading metadata to {metadata_path}...")
    return download_file(METADATA_URL, metadata_path)


def get_camera_ids(metadata_path):
    """Read unique CamId values from the metadata CSV."""
    import pandas as pd
    df = pd.read_csv(metadata_path, usecols=["CamId"])
    return sorted(df["CamId"].unique())


def resize_directory(cam_dir, short_side):
    """Resize all JPEG images in cam_dir so the short side = short_side px."""
    from PIL import Image as PILImage
    count = 0
    for fname in os.listdir(cam_dir):
        if not fname.lower().endswith((".jpg", ".jpeg")):
            continue
        fpath = os.path.join(cam_dir, fname)
        try:
            img = PILImage.open(fpath)
            w, h = img.size
            if min(w, h) <= short_side:
                continue
            if w < h:
                new_w = short_side
                new_h = int(h * short_side / w)
            else:
                new_h = short_side
                new_w = int(w * short_side / h)
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
            img.save(fpath, "JPEG", quality=90)
            count += 1
        except Exception as e:
            print(f"    Warning: could not resize {fpath}: {e}")
    return count


def main():
    args = parse_args()

    # Metadata
    if not download_metadata(args.metadata):
        print("ERROR: could not download metadata.", file=sys.stderr)
        sys.exit(1)

    if args.metadata_only:
        print("Done (metadata only).")
        return

    # Camera images
    cam_ids = get_camera_ids(args.metadata)
    print(f"\n{len(cam_ids)} cameras found in metadata.")
    os.makedirs(args.images_dir, exist_ok=True)

    import pandas as pd
    meta_df = pd.read_csv(args.metadata, usecols=["CamId", "Filename"])
    meta_counts = meta_df.groupby("CamId").size().to_dict()

    results = []  # (cam_id, status, images_on_disk, metadata_rows)

    for cam_id in cam_ids:
        cam_dir = os.path.join(args.images_dir, str(cam_id))
        meta_rows = meta_counts.get(cam_id, 0)

        # Skip if already extracted
        if os.path.isdir(cam_dir) and len(os.listdir(cam_dir)) > 0:
            n_files = len(os.listdir(cam_dir))
            print(f"  Camera {cam_id}: already present ({n_files} files)")
            results.append((cam_id, "present", n_files, meta_rows))
            continue

        # Download zip
        zip_url = f"{IMAGE_BASE_URL}/{cam_id}.zip"
        zip_path = os.path.join(args.images_dir, f"{cam_id}.zip")

        if not download_file(zip_url, zip_path):
            print(f"  Camera {cam_id}: FAILED to download")
            results.append((cam_id, "FAILED", 0, meta_rows))
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue

        # Extract
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Zips may contain files at top level or inside a folder
                members = zf.namelist()
                os.makedirs(cam_dir, exist_ok=True)
                for member in members:
                    basename = os.path.basename(member)
                    if not basename or basename.startswith("."):
                        continue
                    with zf.open(member) as src:
                        dest = os.path.join(cam_dir, basename)
                        with open(dest, "wb") as dst:
                            dst.write(src.read())
            os.remove(zip_path)
            n_files = len(os.listdir(cam_dir))
            print(f"  Camera {cam_id}: extracted {n_files} files")

            # Resize if requested
            if args.resize:
                n_resized = resize_directory(cam_dir, args.resize)
                if n_resized:
                    print(f"    Resized {n_resized} images "
                          f"(short side -> {args.resize} px)")

            results.append((cam_id, "downloaded", n_files, meta_rows))

        except zipfile.BadZipFile:
            print(f"  Camera {cam_id}: bad zip file, skipping")
            results.append((cam_id, "FAILED", 0, meta_rows))
            if os.path.exists(zip_path):
                os.remove(zip_path)

    # Summary table
    print(f"\n{'='*55}")
    print(f"{'CamId':>8}  {'Status':>12}  {'On Disk':>8}  {'Metadata':>8}")
    print(f"{'-'*55}")
    total_disk, total_meta, n_failed = 0, 0, 0
    for cam_id, status, n_disk, n_meta in results:
        print(f"{cam_id:>8}  {status:>12}  {n_disk:>8}  {n_meta:>8}")
        total_disk += n_disk
        total_meta += n_meta
        if status == "FAILED":
            n_failed += 1
    print(f"{'-'*55}")
    print(f"{'TOTAL':>8}  {'':>12}  {total_disk:>8}  {total_meta:>8}")

    if n_failed:
        failed = [str(c) for c, s, _, _ in results if s == "FAILED"]
        print(f"\nWARNING: {n_failed} camera(s) could not be downloaded: "
              f"{', '.join(failed)}")
    else:
        print(f"\nAll {len(results)} cameras downloaded successfully.")


if __name__ == "__main__":
    main()
