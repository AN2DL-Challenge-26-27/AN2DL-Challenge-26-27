from pathlib import Path
from tempfile import TemporaryDirectory

from ..config import COMPETITION
from .dataset import load_manifest, validate_layout


def download_dataset(destination: Path) -> Path:
    import kagglehub
    from kagglehub.exceptions import KaggleApiHTTPError, UnauthenticatedError

    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        validate_layout(destination, load_manifest(destination))
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".kaggle-", dir=destination.parent) as staging:
        try:
            data_dir = Path(kagglehub.competition_download(COMPETITION, output_dir=staging))
        except UnauthenticatedError as exc:
            raise RuntimeError(
                "Kaggle authentication is missing. Set up ~/.kaggle/access_token "
                "or KAGGLE_API_TOKEN and accept the competition rules with that account."
            ) from exc
        except KaggleApiHTTPError as exc:
            raise RuntimeError(str(exc)) from exc

        validate_layout(data_dir, load_manifest(data_dir))
        if data_dir.resolve() != Path(staging).resolve():
            return data_dir
        if destination.exists():
            destination.rmdir()
        data_dir.rename(destination)
    return destination
