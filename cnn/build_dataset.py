import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

from cnn.patch_extractor import extract_patches, save_patches
from cnn.label_patches import label_patches, save_labelled_dataset


def build(data_root: str, scores_csv: str, output_path: str,
          patch_size: int = 64, smart: bool = True) -> None:
    patient_folders = sorted(
        [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"Found {len(patient_folders)} patient folders in {data_root}\n")

    # Translates raw DICOM files into isolated 64x64 pixel image patches containing potential calcium scores.
    all_patches = []
    for i, pid in enumerate(patient_folders, 1):
        folder = os.path.join(data_root, pid)
        print(f"[{i:>3}/{len(patient_folders)}] Patient {pid} ... ", end="", flush=True)
        try:
            patches = extract_patches(folder, patch_size=patch_size)
            all_patches.extend(patches)
            print(f"{len(patches)} patches")
        except Exception as e:
            print(f"SKIPPED ({e})")

    print(f"\nTotal raw patches: {len(all_patches)}\n")

    # Labels the extracted patches against external Ground Truth databases to tell the network what is a true lesion and what is noise.
    if smart:
        print("Labelling mode: SMART (5-rule ground-truth-aware strategy)\n")
        all_patches = label_patches(all_patches, scores_csv, data_root)
    else:
        # Legacy heuristic path kept for backwards compatibility
        print("Labelling mode: HEURISTIC (legacy area+HU thresholds)\n")
        from cnn.label_patches import _heuristic_label
        for p in all_patches:
            p["label"] = _heuristic_label(p["area_mm2"], p["peak_hu"])
        n_pos = sum(1 for p in all_patches if p["label"] == 1)
        n_neg = len(all_patches) - n_pos
        print(f"  Label distribution: {n_pos} CAC (1)  |  {n_neg} FP (0)  |  total {len(all_patches)}")

    # Compresses all labelled patches mathematically into a permanent archive file for neural network ingestion.
    save_labelled_dataset(all_patches, output_path)
    print(f"\nDone. Dataset saved to: {output_path}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build labelled patch dataset from raw DICOM folders")
    parser.add_argument("--data_root",  type=str, default="data/raw",
                        help="Folder containing patient subfolders (default: data/raw)")
    parser.add_argument("--scores_csv", type=str, default="data/scores.csv",
                        help="Ground truth Agatston scores CSV (default: data/scores.csv)")
    parser.add_argument("--output",     type=str, default="data/patches_labelled_smart",
                        help="Output .npz path without extension (default: data/patches_labelled_smart)")
    parser.add_argument("--patch_size", type=int, default=64,
                        help="Patch size in pixels (default: 64)")
    parser.add_argument("--heuristic",  action="store_true",
                        help="Use legacy heuristic labelling instead of smart GT-aware strategy")
    args = parser.parse_args()

    build(args.data_root, args.scores_csv, args.output, args.patch_size, smart=not args.heuristic)
