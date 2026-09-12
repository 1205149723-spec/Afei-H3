#!/usr/bin/env bash
set -euo pipefail
ROOT="${H3_PACKAGE_ROOT:-/root/Afei-H3}"
if [[ ! -x "$ROOT/start_autodl.sh" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
exec bash "$ROOT/start_autodl.sh"
