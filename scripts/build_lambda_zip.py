"""Build the Lambda deployment zip: Linux dependencies + src/ + handler."""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "build" / "deps"
STAGE = ROOT / "build" / "lambda_package"
ZIP_PATH = ROOT / "build" / "lambda_package.zip"
LIMIT_MB = 250


def main():
    if not DEPS.exists():
        raise SystemExit("build/deps not found: install the Linux dependencies first")
    if STAGE.exists():
        shutil.rmtree(STAGE)

    shutil.copytree(DEPS, STAGE)
    shutil.copytree(ROOT / "src", STAGE / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "lambda_fn" / "handler.py", STAGE / "handler.py")

    for folder in list(STAGE.rglob("__pycache__")) + list(STAGE.rglob("tests")):
        shutil.rmtree(folder, ignore_errors=True)

    unzipped = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file())
    print(f"Unzipped size: {unzipped / 1024**2:.0f} MB (limit {LIMIT_MB} MB)")
    if unzipped > LIMIT_MB * 1024**2:
        raise SystemExit("Package is over the Lambda limit")

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGE).as_posix())
    print(f"Zip written: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()