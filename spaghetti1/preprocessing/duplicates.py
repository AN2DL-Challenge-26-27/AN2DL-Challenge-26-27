"""Near-duplicate detection based on the star field around the central galaxy.

Documentation: ``docs/03_PREPROCESSING.md`` (section "Near-duplicates").

The dataset contains the same sky cutout several times with a different noise realisation,
brightness or colour. Whole-image similarity is high for any two centred galaxies, so the
fingerprint ignores the centre: it is a band-pass (difference-of-Gaussians) image restricted
to an outer ring, where the background stars form a unique pattern. Matching is done over
the 8 rotations and reflections of the square (D8).
"""

import cv2
import numpy as np
import pandas as pd
import torch

FINGERPRINT_SIZE = 128
RING_INNER_RADIUS = 32
BAND_PASS_SIGMAS = (1.0, 4.0)
# Thresholds were set from the orientation of the best match: true copies match without
# rotation, while chance matches spread evenly over the 8 orientations. Rotated (chance)
# matches reach 0.72 within training (4k x 4k images) and 0.78 between training and test
# (4k x 14.7k, more chances), so the cross-split threshold is higher.
DUPLICATE_MIN_CORRELATION = 0.75
CROSS_SPLIT_MIN_CORRELATION = 0.80

_yy, _xx = np.mgrid[:FINGERPRINT_SIZE, :FINGERPRINT_SIZE]
_centre = (FINGERPRINT_SIZE - 1) / 2
RING_MASK = ((_yy - _centre) ** 2 + (_xx - _centre) ** 2) > RING_INNER_RADIUS**2


def star_field_fingerprint(pixels: np.ndarray) -> np.ndarray:
    """Return a band-pass grayscale image (float16) used for duplicate matching."""
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (FINGERPRINT_SIZE, FINGERPRINT_SIZE), interpolation=cv2.INTER_AREA)
    fine, coarse = (cv2.GaussianBlur(gray, (0, 0), sigma) for sigma in BAND_PASS_SIGMAS)
    return (fine - coarse).astype(np.float16)


def _ring_vectors(fingerprints: torch.Tensor) -> torch.Tensor:
    """Flatten the ring, centre it, and scale it to unit length (Pearson correlation via dot product)."""
    ring = fingerprints[:, torch.from_numpy(RING_MASK)].float()
    ring = ring - ring.mean(dim=1, keepdim=True)
    return torch.nn.functional.normalize(ring, dim=1)


def _dihedral(fingerprints: torch.Tensor, index: int) -> torch.Tensor:
    if index >= 4:
        fingerprints = fingerprints.transpose(1, 2)
    return torch.rot90(fingerprints, index % 4, dims=(1, 2))


def find_duplicate_pairs(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    threshold: float = DUPLICATE_MIN_CORRELATION,
    same_set: bool = False,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Return index pairs (query, reference) whose star-field correlation >= threshold.

    ``query`` and ``reference`` are stacks of fingerprints (N, S, S). With ``same_set=True``
    each unordered pair is reported once and self-matches are skipped.
    """
    # The ring mask is symmetric under D8, so transforming the (smaller) query set is
    # equivalent to transforming the references and keeps memory low.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    query_tensor = torch.from_numpy(np.ascontiguousarray(query))
    queries = torch.stack([_ring_vectors(_dihedral(query_tensor, index)) for index in range(8)])
    queries = queries.to(device, dtype)  # (8, Q, D)
    reference_tensor = torch.from_numpy(np.ascontiguousarray(reference))

    rows = []
    for start in range(0, len(reference_tensor), batch_size):
        chunk = _ring_vectors(reference_tensor[start : start + batch_size]).to(device, dtype)
        best, transform = (queries @ chunk.T).float().max(dim=0)  # (Q, B)
        if same_set:
            query_index = torch.arange(best.shape[0], device=device)[:, None]
            reference_index = torch.arange(start, start + len(chunk), device=device)[None, :]
            best = best.masked_fill(reference_index <= query_index, -1.0)
        hits = (best >= threshold).nonzero()
        for q, r in hits.tolist():
            rows.append((q, start + r, float(best[q, r]), int(transform[q, r])))
    return pd.DataFrame(rows, columns=["query_index", "reference_index", "correlation", "dihedral"])


def connected_groups(count: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Union-find over index pairs; returns a group id per item (the smallest member index)."""
    parent = np.arange(count)

    def root(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([root(item) for item in range(count)])
