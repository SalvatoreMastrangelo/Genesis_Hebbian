"""
Common Random Numbers (CRN) for WP2 multi-individual evaluation.
================================================================

WP2 evaluates a CMA-ES population by packing ``E = P * F`` env slots into each
scene, laid out as ``slot e = p*F + f`` (individual ``p``, forest/scenario
``f``), so that ``forest_ids[e] == f``.  CMA-ES ranks individuals, so what
matters for a clean ranking is not the *absolute* noise on one individual's
fitness (``sigma_ind``) but the noise on the *difference* between two
individuals (``sigma_rank``).  Domain-randomization draws that are independent
per slot inflate ``sigma_rank``; draws that are *shared* across individuals
flying the same scenario cancel in the comparison.

``crn_share_`` is the variance-reduction primitive: it rewrites a per-slot
buffer in place so that all slots sharing a scenario id hold the same value.
The *random draw* (epsilon) is made common; any downstream state-dependent
scaling (e.g. depth-proportional sensor noise, force-proportional aero noise)
is applied afterwards and remains per-slot — which is correct, because only the
stochastic part should be shared, not the deterministic response to a diverging
trajectory.

This preserves full domain randomization: across the ``F`` scenarios the draws
still span the DR distribution (and are refreshed each generation), so an
individual is still selected for robustness *across* randomized dynamics — every
competitor is simply measured on the same draws.
"""

from __future__ import annotations

from typing import Optional

import torch


def crn_share_(buf: torch.Tensor, ids: Optional[torch.Tensor]) -> None:
    """In-place common-random-numbers remap: ``buf[e] <- buf[ids[e]]``.

    After the call, any two rows ``e1, e2`` with ``ids[e1] == ids[e2]`` hold the
    same value, so a per-slot random draw becomes COMMON across all slots that
    share a scenario id (e.g. individuals flying the same forest).

    Parameters
    ----------
    buf : Tensor, shape ``(N, ...)``
        Per-slot buffer to share in place (e.g. a freshly drawn ``N(0, 1)``
        tensor).  Only dim 0 is indexed; any trailing dims are preserved.
    ids : Tensor of long, shape ``(N,)``, or None
        Per-slot scenario ids with values in ``[0, N)``.  ``None`` is a no-op
        (CRN disabled, e.g. during training).

    Notes
    -----
    * No-op when ``ids is None``.
    * Skips silently when ``buf.shape[0] != ids.shape[0]`` — this happens for a
      *partial* reset (a subset of envs auto-resetting mid-episode).  Those
      slots are post-termination and excluded from the fitness reduction, so
      sharing them is unnecessary; and a partial buffer is not row-aligned with
      the full ``ids`` vector, so remapping it would be incorrect.
    * ``buf[ids]`` allocates a temporary (advanced indexing copies), so there is
      no aliasing hazard with the in-place ``copy_``.
    """
    if ids is None:
        return
    if buf.shape[0] != ids.shape[0]:
        return
    buf.copy_(buf[ids])
