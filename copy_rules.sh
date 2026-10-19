#!/bin/bash
# Install the outputs of Notebooks/Association_Rule_Mining.ipynb (run from Notebooks/)
# where the Government API reads them.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p app/ml_models/association_rules
cp Notebooks/artifacts/association_rules_readable.json app/ml_models/association_rules/
cp Notebooks/artifacts/association_rules.csv app/ml_models/association_rules/
# The notebook names its summary model_summary.json; pass that file explicitly,
# since other notebooks write a file of the same name.
if [ "${1:-}" != "" ]; then
  cp "$1" app/ml_models/association_rules/association_rules_summary.json
fi
echo "Installed the association rules in app/ml_models/association_rules/"
