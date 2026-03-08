"""
Dataset discovery: find paired low/normal images by numeric id in filenames.
Proposal §2: 80–120 paired images from LOL-v2 Real (data/low, data/normal).
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple


def discover_pairs(data_root: str) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Find paired low/normal images by matching numeric id in filename.
    Expects: data_root/low/<name>.png and data_root/normal/<name>.png
    with names like low00001.png / normal00001.png (same numeric part).
    Returns:
        pairs: list of (path_low, path_normal)
        low_only: ids that have low but no normal
        normal_only: ids that have normal but no low
    """
    data_root = Path(data_root)
    low_dir = data_root / "low"
    normal_dir = data_root / "normal"

    def numeric_id(path: Path) -> Optional[str]:
        m = re.search(r"(\d+)$", path.stem)
        return m.group(1) if m else None

    low_files = {numeric_id(p): p for p in low_dir.glob("*.png") if numeric_id(p)}
    normal_files = {numeric_id(p): p for p in normal_dir.glob("*.png") if numeric_id(p)}
    common_ids = set(low_files) & set(normal_files)
    pairs = [(str(low_files[i]), str(normal_files[i])) for i in sorted(common_ids, key=int)]
    low_only = sorted(set(low_files) - set(normal_files), key=int)
    normal_only = sorted(set(normal_files) - set(low_files), key=int)
    return pairs, low_only, normal_only
