#!/usr/bin/env bash
# safe-label.sh — label wrapper with a denylist.
#
# Usage: safe-label.sh <number> <add|remove> <label>
#
# Blocks the "ready-for-fix" label so agent workflows cannot apply it and
# thereby trigger the bug-fix PR workflow. All other labels pass through.
# The denylist is intentionally small; this is the structural boundary that
# makes fix-PR creation a human-only decision.
#
# GITHUB_REPOSITORY is provided automatically by GitHub Actions.

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <number> <add|remove> <label>" >&2
    exit 2
fi

NUMBER="$1"
ACTION="$2"
LABEL="$3"

# Deny ADD of the ready-for-fix label. REMOVE is allowed so the fix
# workflow can clear the label after opening its PR.
case "$ACTION:$LABEL" in
    add:ready-for-fix)
        echo "safe-label.sh: refusing to add label 'ready-for-fix' — reserved for maintainers" >&2
        exit 1
        ;;
esac

REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

case "$ACTION" in
    add)
        exec gh issue edit "$NUMBER" --repo "$REPO" --add-label "$LABEL"
        ;;
    remove)
        exec gh issue edit "$NUMBER" --repo "$REPO" --remove-label "$LABEL"
        ;;
    *)
        echo "safe-label.sh: unknown action '$ACTION' (expected add|remove)" >&2
        exit 2
        ;;
esac
