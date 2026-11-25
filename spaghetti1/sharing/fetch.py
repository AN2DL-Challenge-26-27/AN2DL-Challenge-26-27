"""Get the cleaned dataset package onto the local disk, from Google Drive or a local copy.

Command: ``python -m spaghetti1 fetch --source <link | path>``. In a notebook::

    from spaghetti1.sharing.fetch import prepare_dataset
    data_dir = prepare_dataset("https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing")

``source`` may be any of:

* a Google Drive share link or bare file id of ``spaghetti1_clean_v1.zip``
  (downloaded with ``gdown``; pre-installed on Colab, otherwise ``pip install gdown``);
* a path to the zip, for example on a mounted Drive:
  ``/content/drive/MyDrive/AN2DL/spaghetti1_clean_v1.zip`` (copied to local disk first,
  because reading thousands of small files from the Drive mount is very slow);
* a folder that is already extracted (it must contain ``dataset_info.json``).

The function returns the dataset folder. Pass it as ``data_dir`` to ``GalaxyDataset`` and
``data_dir / "manifests"`` as the manifest folder. Running it again reuses a verified copy.
"""

import json
from pathlib import Path
import re
import shutil
import zipfile

from .export import INFO_FILE

DRIVE_ID_PATTERNS = (r"/d/([A-Za-z0-9_-]{20,})", r"[?&]id=([A-Za-z0-9_-]{20,})", r"^([A-Za-z0-9_-]{20,})$")


def default_target_dir() -> Path:
    """``/content/spaghetti1_data`` on Colab (fast local disk), else ``data/clean`` in the project."""
    colab = Path("/content")
    if colab.is_dir():
        return colab / "spaghetti1_data"
    from ..config import PROJECT_DIR

    return PROJECT_DIR / "data" / "clean"


def drive_file_id(source: str) -> str | None:
    """Extract the file id from a Google Drive link, or return None if ``source`` is not one."""
    for pattern in DRIVE_ID_PATTERNS:
        match = re.search(pattern, source.strip())
        if match:
            return match.group(1)
    return None


def verify_package(root: Path) -> dict:
    """Check that the file counts match ``dataset_info.json``; return the info dictionary."""
    info = json.loads((root / INFO_FILE).read_text(encoding="utf-8"))
    found = {"train_png": len(list((root / "train").glob("*.png"))), "test_png": len(list((root / "test").glob("*.png")))}
    for key, count in found.items():
        if count != info["counts"][key]:
            raise RuntimeError(f"{root}: expected {info['counts'][key]:,} files for {key}, found {count:,}")
    for name in ("train_manifest.csv", "test_manifest.csv"):
        if not (root / "manifests" / name).is_file():
            raise RuntimeError(f"{root}: missing manifests/{name}")
    return info


def find_package_root(folder: Path, info_file: str = INFO_FILE) -> Path | None:
    """The folder holding ``info_file``: ``folder`` itself or one level below it."""
    if (folder / info_file).is_file():
        return folder
    candidates = sorted(path.parent for path in folder.glob(f"*/{info_file}"))
    return candidates[-1] if candidates else None


def _extract(archive: Path, target_dir: Path, info_file: str) -> Path:
    print(f"Extracting {archive.name} to {target_dir} ...", flush=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target_dir)
    root = find_package_root(target_dir, info_file)
    if root is None:
        raise RuntimeError(f"{archive} does not contain {info_file}")
    return root


def obtain_package(source: str | Path, target_dir: Path, *, info_file: str, verify, zip_name: str,
                   keep_zip: bool = False) -> Path:
    """Shared logic of every package (dataset, model weights): reuse, copy, or download, then verify.

    ``verify(root)`` must raise ``RuntimeError`` for an incomplete package and return its info.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(str(source))

    if source_path.is_dir():  # already extracted somewhere
        root = find_package_root(source_path, info_file)
        if root is None:
            raise FileNotFoundError(f"{source_path} has no {info_file}; point to the extracted package folder")
        verify(root)
        return root

    existing = find_package_root(target_dir, info_file)
    if existing is not None:
        try:
            info = verify(existing)
            print(f"Using existing {info['name']} {info['version']} at {existing}")
            return existing
        except RuntimeError as error:
            print(f"Existing copy is incomplete ({error}); fetching again.")
            shutil.rmtree(existing)

    if source_path.suffix == ".zip" and source_path.is_file():
        archive = target_dir / source_path.name
        if source_path.resolve() != archive.resolve():
            print(f"Copying {source_path} to local disk ...", flush=True)
            shutil.copyfile(source_path, archive)
    else:
        file_id = drive_file_id(str(source))
        if file_id is None:
            raise ValueError(f"Not a folder, zip file, or Google Drive link: {source}")
        try:
            import gdown
        except ImportError as error:
            raise ImportError("Downloading from Google Drive needs gdown: pip install gdown") from error
        archive = target_dir / zip_name
        print(f"Downloading Google Drive file {file_id} ...", flush=True)
        if gdown.download(id=file_id, output=str(archive), quiet=False) is None:
            raise RuntimeError("Download failed. Share the file as 'Anyone with the link' (Viewer) and retry.")

    root = _extract(archive, target_dir, info_file)
    if not keep_zip and archive.resolve() != source_path.resolve():
        archive.unlink()
    verify(root)
    return root


def prepare_dataset(source: str | Path, target_dir: Path | None = None, *, keep_zip: bool = False) -> Path:
    """Make the cleaned dataset available locally and return its root folder."""
    root = obtain_package(source, Path(target_dir) if target_dir else default_target_dir(), info_file=INFO_FILE,
                          verify=verify_package, zip_name="spaghetti1_clean.zip", keep_zip=keep_zip)
    info = verify_package(root)
    print(f"Ready: {info['name']} {info['version']} at {root} "
          f"({info['counts']['train_png']:,} train / {info['counts']['test_png']:,} test images)")
    return root
