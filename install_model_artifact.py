"""Install a notebook-exported artifact into the backend's model bundle."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = BACKEND_ROOT / "app" / "ml_models"
ALLOWED_BUNDLES = {"severity", "stay_duration", "transfer", "discharge", "forecasting"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a trained artifact to the location declared by its model contract.",
    )
    parser.add_argument("bundle", choices=sorted(ALLOWED_BUNDLES))
    parser.add_argument("source", type=Path, help="Artifact exported by the notebook")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional companion metadata artifact declared by model_contract.json",
    )
    parser.add_argument("--version", help="Model version written into model_contract.json")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Artifact does not exist: {source}")

    bundle_dir = (MODEL_ROOT / args.bundle).resolve()
    contract_path = bundle_dir / "model_contract.json"
    if not contract_path.is_file():
        parser.error(f"Model contract does not exist: {contract_path}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    artifact_name = contract.get("artifact")
    if not artifact_name or Path(str(artifact_name)).name != artifact_name:
        parser.error("The model contract contains an invalid artifact filename.")

    destination = (bundle_dir / artifact_name).resolve()
    if destination.parent != bundle_dir:
        parser.error("The artifact destination escapes the model bundle directory.")

    shutil.copy2(source, destination)

    if args.metadata:
        metadata_source = args.metadata.expanduser().resolve()
        if not metadata_source.is_file():
            parser.error(f"Metadata artifact does not exist: {metadata_source}")
        metadata_name = contract.get("metadata_artifact")
        if not metadata_name or Path(str(metadata_name)).name != metadata_name:
            parser.error("The model contract does not declare a valid metadata artifact.")
        metadata_destination = (bundle_dir / metadata_name).resolve()
        if metadata_destination.parent != bundle_dir:
            parser.error("The metadata destination escapes the model bundle directory.")
        shutil.copy2(metadata_source, metadata_destination)
        print(f"Installed metadata: {metadata_destination}")

    if args.version:
        contract["version"] = args.version
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    print(f"Installed {args.bundle}: {destination}")
    print("Restart FastAPI, then check GET /health/models.")


if __name__ == "__main__":
    main()
