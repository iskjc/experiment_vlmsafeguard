"""
Main pipeline: runs all experimental steps end-to-end.

Usage:
    python run_pipeline.py                      # uses config.yaml
    python run_pipeline.py --config my.yaml
    python run_pipeline.py --text-only          # skip image step (no GPU needed for analysis)
    python run_pipeline.py --pooling mean       # override pooling strategy
    python run_pipeline.py --layers 14 15 16 17 # only extract these layers
    python run_pipeline.py --skip-extract       # load cached .npy, skip model inference
"""

from __future__ import annotations
import argparse
import yaml
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--pooling", choices=["last", "mean"], default=None)
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--skip-extract", action="store_true",
                   help="Load cached .npy files instead of re-running the model")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_states(states: dict[int, np.ndarray], labels: np.ndarray, prefix: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{prefix}_labels.npy", labels)
    for layer_idx, arr in states.items():
        np.save(cache_dir / f"{prefix}_layer{layer_idx}.npy", arr)
    print(f"[Cache] Saved {prefix} states → {cache_dir} ({len(states)} layers)")


def load_states(prefix: str, cache_dir: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    labels = np.load(cache_dir / f"{prefix}_labels.npy")
    states = {}
    for f in sorted(cache_dir.glob(f"{prefix}_layer*.npy")):
        layer_idx = int(f.stem.replace(f"{prefix}_layer", ""))
        states[layer_idx] = np.load(f)
    print(f"[Cache] Loaded {prefix} states from {cache_dir} ({len(states)} layers)")
    return states, labels


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.pooling:
        cfg["extraction"]["pooling"] = args.pooling
    if args.layers:
        cfg["extraction"]["layers"] = args.layers

    out_dir = Path(cfg["output"]["dir"])
    cache_dir = out_dir / "cache"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load model (only if we need to run inference)
    # ------------------------------------------------------------------ #
    need_model = not args.skip_extract or (
        not (cache_dir / "text_labels.npy").exists()
    )

    extractor = None
    if need_model:
        from extract.hidden_states import load_model, HiddenStateExtractor
        model, processor = load_model(
            cfg["model"]["name"],
            device=cfg["model"]["device"],
            dtype=cfg["model"]["dtype"],
        )
        extractor = HiddenStateExtractor(
            model=model,
            processor=processor,
            device=cfg["model"]["device"],
            pooling=cfg["extraction"]["pooling"],
            batch_size=cfg["extraction"]["batch_size"],
            layers=cfg["extraction"]["layers"],
        )

    # ------------------------------------------------------------------ #
    # STEP 1 — Extract text hidden states
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 1  Extract text hidden states  [PKU-SafeRLHF]")
    print("=" * 60)

    if args.skip_extract and (cache_dir / "text_labels.npy").exists():
        text_states, text_labels = load_states("text", cache_dir)
    else:
        from data.pku_saferlhf import load_pku_saferlhf
        texts, text_labels = load_pku_saferlhf(
            cfg["data"]["pku_saferlhf"]["path"],
            n_samples=cfg["data"]["pku_saferlhf"]["n_samples"],
            text_col=cfg["data"]["pku_saferlhf"]["text_col"],
            response_col=cfg["data"]["pku_saferlhf"]["response_col"],
            label_col=cfg["data"]["pku_saferlhf"]["label_col"],
        )
        text_states = extractor.extract_text(texts, text_labels)
        if cfg["output"]["save_states"]:
            save_states(text_states, text_labels, "text", cache_dir)

    # ------------------------------------------------------------------ #
    # STEP 2 — Locate safety-active layers via linear probe
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 2  Locate safety layers  [linear probe per layer]")
    print("=" * 60)

    from analysis.layer_locator import LayerLocator
    locator = LayerLocator(n_safety_layers=cfg["analysis"]["n_safety_layers"])
    locator.fit(text_states, text_labels)

    from visualize.plots import plot_layer_accuracy
    plot_layer_accuracy(locator.layer_accuracy_, locator.top_layers_, str(plot_dir))

    text_states_top = locator.get_top_layer_states(text_states)

    # ------------------------------------------------------------------ #
    # STEP 3 — Fit PLS direction on text data only
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 3  Fit PLS safe/harmful direction  [text only]")
    print("=" * 60)

    from analysis.pls_direction import PLSDirectionFinder
    finder = PLSDirectionFinder(n_components=cfg["analysis"]["n_pls_components"])
    finder.fit_text(text_states_top, text_labels)

    # ------------------------------------------------------------------ #
    # STEP 4 — Extract image hidden states + project onto text direction
    # ------------------------------------------------------------------ #
    if not args.text_only:
        print("\n" + "=" * 60)
        print("STEP 4  Project image hidden states onto text safety direction")
        print("=" * 60)

        if args.skip_extract and (cache_dir / "image_labels.npy").exists():
            image_states_all, image_labels = load_states("image", cache_dir)
        else:
            from data.visual_harm import load_visual_harm
            images, prompts, image_labels = load_visual_harm(
                cfg["data"]["visual_harm"]["path"],
                n_samples=cfg["data"]["visual_harm"]["n_samples"],
                image_col=cfg["data"]["visual_harm"]["image_col"],
                label_col=cfg["data"]["visual_harm"]["label_col"],
                prompt_template=cfg["data"]["visual_harm"]["prompt_template"],
            )
            # Only extract safety layers to save time
            extractor.layers = locator.top_layers_
            image_states_all = extractor.extract_image(images, prompts, image_labels)
            if cfg["output"]["save_states"]:
                save_states(image_states_all, image_labels, "image", cache_dir)

        image_states_top = {
            k: image_states_all[k]
            for k in locator.top_layers_
            if k in image_states_all
        }
        finder.project_images(image_states_top, image_labels)

    # ------------------------------------------------------------------ #
    # STEP 5 — Visualize
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("STEP 5  Visualize")
    print("=" * 60)

    from visualize.plots import plot_projection, plot_boundary_analysis, plot_summary_heatmap

    summary_stats = {}
    for layer_idx, result in finder.results_.items():
        img_scores = result.image_scores if not args.text_only else None
        img_labels = result.image_labels if not args.text_only else None

        plot_projection(
            layer_idx=layer_idx,
            text_scores=result.text_scores,
            text_labels=result.text_labels,
            image_scores=img_scores,
            image_labels=img_labels,
            save_dir=str(plot_dir),
            pooling=cfg["extraction"]["pooling"],
        )
        plot_boundary_analysis(
            layer_idx=layer_idx,
            text_scores=result.text_scores,
            text_labels=result.text_labels,
            image_scores=img_scores,
            image_labels=img_labels,
            threshold=cfg["analysis"]["boundary_threshold"],
            save_dir=str(plot_dir),
        )

        s = result.stats.copy()
        if img_scores is not None and img_labels is not None:
            harm_img = img_scores[img_labels == 1]
            safe_img = img_scores[img_labels == 0]
            s["img_harm_mean"] = float(harm_img.mean()) if len(harm_img) else float("nan")
            s["img_safe_mean"] = float(safe_img.mean()) if len(safe_img) else float("nan")
        summary_stats[layer_idx] = s

    plot_summary_heatmap(summary_stats, str(plot_dir))

    # ------------------------------------------------------------------ #
    # Print interpretation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for layer_idx, result in finder.results_.items():
        s = result.stats
        print(f"\nLayer {layer_idx}:")
        print(f"  Text safe mean    : {s.get('safe_mean', float('nan')):+.3f}")
        print(f"  Text harmful mean : {s.get('harm_mean', float('nan')):+.3f}")
        print(f"  Separation (d)    : {s.get('separation', float('nan')):.3f}")
        if result.image_scores is not None:
            ih = s.get("img_harm_mean", float("nan"))
            print(f"  Image harmful mean: {ih:+.3f}  ← key finding")
            verdict = "SAFE side" if ih > 0 else "HARMFUL side"
            print(f"  → Harmful images land on {verdict} of text safety boundary")


if __name__ == "__main__":
    main()
