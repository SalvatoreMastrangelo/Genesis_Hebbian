"""Difference of the final Hebbian last-layer ΔW between two champion flights.

Works purely from the ``trajectory.pkl`` files written by the champion flight
videos (``<run>/plots/champion_videos/<label>/trajectory.pkl``): no Genesis
runtime, no re-rendering. ``eval_visual.run_and_record`` stores
``hebbian_weight_delta_history`` with shape ``(T, num_actions, hidden_dim)``,
each entry being ``W(t) - W_checkpoint`` of the plastic last layer after the
step's Hebbian update, so the last entry is the final ΔW of the flight.

Outputs (default ``<run>/plots/champion_videos/deltaw/``):

* ``deltaw_diff.png/.pdf``           — heatmap of ``ΔW_A[-1] - ΔW_B[-1]``.
* ``deltaw_final_panels.png/.pdf``   — ``ΔW_A[-1]``, ``ΔW_B[-1]`` and their
  difference on ONE symmetric colour scale.
* ``deltaw_diff_matched_t.png/.pdf`` — same subtraction but with the longer
  flight sampled at the time the shorter one ended (only when lengths differ).
* ``*_T.png/.pdf``                 — the same three figures transposed
  (head-neuron rows × actuator columns, tall and narrow, actuator labels on
  the x axis), the layout of the ``overlay_alternative`` videos.
* ``deltaw_final.npz`` (all matrices) and ``deltaw_summary.csv`` (norms,
  correlation, flight lengths).

Usage::

    python -m WP2_Outer_Loop.champion_deltaw_diff RUN_DIR \
        [--a progress_champion] [--b cot_champion_gated80m] [--out DIR]

``RUN_DIR`` is the timestamped run folder or its synced ``outer_<exp>_rX``
wrapper (the one holding ``plots/champion_videos``).
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Same order/labels as the overlay heatmap in eval_visual.create_overlay_video.
ACTUATOR_NAMES = ["throttle", "sweep_L", "sweep_R", "twist_L", "twist_R", "elevator", "rudder"]
CMAP = "seismic"  # blue <-> red, white at zero — matches the overlay videos
LABELS = {"progress_champion": "progress champion", "cot_champion_gated80m": "CoT champion"}


def _resolve_videos_dir(run_dir: Path, videos_dir: str) -> Path:
    """Accept the timestamped run folder or its ``outer_<exp>_rX`` wrapper."""
    cands = [run_dir / videos_dir, run_dir / "plots" / videos_dir]
    cands += sorted(run_dir.glob(f"*/plots/{videos_dir}"))
    for c in cands:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"no '{videos_dir}' folder under {run_dir}")


def load_delta_history(label_dir: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return ``(ΔW history (T, A, H), time_steps (T,), end_reason)``."""
    with open(label_dir / "trajectory.pkl", "rb") as f:
        traj = pickle.load(f)["traj"]
    hist = traj.get("hebbian_weight_delta_history")
    if hist is None or hist.size == 0:
        raise ValueError(f"{label_dir}: trajectory has no hebbian_weight_delta_history")
    return (
        np.asarray(hist, dtype=np.float64),
        np.asarray(traj["time_steps"], dtype=np.float64),
        str(traj.get("end_reason", "")),
    )


def _pretty(label: str) -> str:
    return LABELS.get(label, label.replace("_", " "))


def _heatmap(ax, mat: np.ndarray, vmax: float, title: str, transposed: bool = False):
    """Draw one ΔW matrix (actuators × head neurons).

    ``transposed`` draws head neurons as rows and actuators as columns, the
    tall layout of the ``overlay_alternative`` videos.
    """
    n_act, n_hid = mat.shape
    act_labels = ACTUATOR_NAMES if n_act == len(ACTUATOR_NAMES) else [str(i) for i in range(n_act)]
    hid_ticks = list(range(0, n_hid, 4))
    shown = mat.T if transposed else mat
    im = ax.imshow(shown, aspect="auto", cmap=CMAP, vmin=-vmax, vmax=vmax, interpolation="nearest")
    if transposed:
        ax.set_xticks(range(n_act))
        ax.set_xticklabels(act_labels, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(hid_ticks)
        ax.set_yticklabels([str(i) for i in hid_ticks], fontsize=9)
        ax.set_xticks(np.arange(-0.5, n_act, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_hid, 1), minor=True)
    else:
        ax.set_yticks(range(n_act))
        ax.set_yticklabels(act_labels, fontsize=9)
        ax.set_xticks(hid_ticks)
        ax.set_xticklabels([str(i) for i in hid_ticks], fontsize=9)
        ax.set_xticks(np.arange(-0.5, n_hid, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_act, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2)
    ax.set_title(title, fontsize=11)
    return im


def _save(fig, out_stem: Path):
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")


def plot_diff(diff: np.ndarray, title: str, out_stem: Path, transposed: bool = False):
    import matplotlib.pyplot as plt

    vmax = float(np.abs(diff).max()) or 1e-8
    fig, ax = plt.subplots(figsize=(4.2, 9.0) if transposed else (12, 3.6))
    im = _heatmap(ax, diff, vmax, title, transposed=transposed)
    ax.set_xlabel("Actuator" if transposed else "Head neuron")
    ax.set_ylabel("Head neuron" if transposed else "Actuator")
    cb = fig.colorbar(im, ax=ax, pad=0.03 if transposed else 0.015, fraction=0.08 if transposed else 0.03)
    cb.set_label("ΔW difference")
    _save(fig, out_stem)
    plt.close(fig)


def plot_panels(mats: Dict[str, np.ndarray], out_stem: Path, transposed: bool = False):
    import matplotlib.pyplot as plt

    vmax = max(float(np.abs(m).max()) for m in mats.values()) or 1e-8
    n = len(mats)
    if transposed:
        fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.8, 9.0), sharey=True)
    else:
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.0 * n), sharex=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, (title, mat) in zip(axes, mats.items()):
        im = _heatmap(ax, mat, vmax, title, transposed=transposed)
        if transposed:
            ax.set_xlabel("Actuator")
        else:
            ax.set_ylabel("Actuator")
    if transposed:
        axes[0].set_ylabel("Head neuron")
    else:
        axes[-1].set_xlabel("Head neuron")
    cb = fig.colorbar(im, ax=list(axes), pad=0.02 if transposed else 0.015, fraction=0.03)
    cb.set_label("ΔW")
    _save(fig, out_stem)
    plt.close(fig)


def run(run_dir: Path, label_a: str, label_b: str, videos_dir: str, out: Path | None) -> Path:
    vids = _resolve_videos_dir(run_dir, videos_dir)
    out = out or (vids / "deltaw")
    out.mkdir(parents=True, exist_ok=True)

    hist_a, t_a, end_a = load_delta_history(vids / label_a)
    hist_b, t_b, end_b = load_delta_history(vids / label_b)
    if hist_a.shape[1:] != hist_b.shape[1:]:
        raise ValueError(f"ΔW shapes differ: {hist_a.shape[1:]} vs {hist_b.shape[1:]}")
    name_a, name_b = _pretty(label_a), _pretty(label_b)

    final_a, final_b = hist_a[-1], hist_b[-1]
    diff = final_a - final_b
    title_diff = f"Final Hebbian ΔW: {name_a} − {name_b}"
    panels = {
        f"Final Hebbian ΔW, {name_a}": final_a,
        f"Final Hebbian ΔW, {name_b}": final_b,
        f"{name_a} − {name_b}": diff,
    }
    plot_diff(diff, title_diff, out / "deltaw_diff")
    plot_diff(diff, title_diff.replace(": ", ":\n"), out / "deltaw_diff_T", transposed=True)
    plot_panels(panels, out / "deltaw_final_panels")
    plot_panels(
        {k.replace(", ", ",\n").replace(" − ", " −\n"): v for k, v in panels.items()},
        out / "deltaw_final_panels_T",
        transposed=True,
    )

    # Matched-time variant: sample the longer flight where the shorter one ended.
    k = min(len(hist_a), len(hist_b)) - 1
    matched_a, matched_b = hist_a[k], hist_b[k]
    diff_matched = matched_a - matched_b
    if len(hist_a) != len(hist_b):
        title_m = f"Hebbian ΔW at t = {t_a[k]:.1f} s: {name_a} − {name_b}"
        plot_diff(diff_matched, title_m, out / "deltaw_diff_matched_t")
        plot_diff(diff_matched, title_m.replace(": ", ":\n"), out / "deltaw_diff_matched_t_T", transposed=True)

    np.savez(
        out / "deltaw_final.npz",
        final_a=final_a, final_b=final_b, diff=diff,
        matched_a=matched_a, matched_b=matched_b, diff_matched=diff_matched,
        t_final_a=t_a[-1], t_final_b=t_b[-1], t_matched=t_a[k],
        label_a=label_a, label_b=label_b, actuators=np.array(ACTUATOR_NAMES),
    )
    rows = [
        ("label_a", label_a), ("label_b", label_b),
        ("steps_a", len(hist_a)), ("steps_b", len(hist_b)),
        ("t_final_a_s", f"{t_a[-1]:.2f}"), ("t_final_b_s", f"{t_b[-1]:.2f}"),
        ("end_reason_a", end_a), ("end_reason_b", end_b),
        ("max_abs_final_a", f"{np.abs(final_a).max():.4f}"), ("max_abs_final_b", f"{np.abs(final_b).max():.4f}"),
        ("fro_final_a", f"{np.linalg.norm(final_a):.4f}"), ("fro_final_b", f"{np.linalg.norm(final_b):.4f}"),
        ("max_abs_diff", f"{np.abs(diff).max():.4f}"), ("fro_diff", f"{np.linalg.norm(diff):.4f}"),
        ("corr_final", f"{np.corrcoef(final_a.ravel(), final_b.ravel())[0, 1]:.4f}"),
        ("t_matched_s", f"{t_a[k]:.2f}"),
        ("max_abs_diff_matched", f"{np.abs(diff_matched).max():.4f}"), ("fro_diff_matched", f"{np.linalg.norm(diff_matched):.4f}"),
        ("corr_matched", f"{np.corrcoef(matched_a.ravel(), matched_b.ravel())[0, 1]:.4f}"),
    ]
    with open(out / "deltaw_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)
    for k_, v_ in rows:
        print(f"{k_:>22}: {v_}")
    print(f"outputs -> {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path, metavar="RUN_DIR")
    ap.add_argument("--a", default="progress_champion", help="minuend label dir (default: progress_champion)")
    ap.add_argument("--b", default="cot_champion_gated80m", help="subtrahend label dir (default: cot_champion_gated80m)")
    ap.add_argument("--videos-dir", default="champion_videos")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: <videos-dir>/deltaw)")
    args = ap.parse_args(argv)
    run(args.run_dir, args.a, args.b, args.videos_dir, args.out)


if __name__ == "__main__":
    main()
