#!/usr/bin/env bash
# safe-pr-review.sh — PR review wrapper that hardcodes --comment.
#
# Usage: safe-pr-review.sh <pr-number> <body-file>
#
# Forces --comment so the agent cannot approve or request changes on
# a PR. Reads the review body from a file to avoid shell-quoting pitfalls
# with attacker-influenced content.
#
# GITHUB_REPOSITORY is provided automatically by GitHub Actions.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <pr-number> <body-file>" >&2
    exit 2
fi

NUMBER="$1"
BODY_FILE="$2"

if [[ ! -f "$BODY_FILE" ]]; then
    echo "safe-pr-review.sh: body file '$BODY_FILE' does not exist" >&2
    exit 2
fi

REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

exec gh pr review "$NUMBER" --repo "$REPO" --comment --body-file "$BODY_FILE"
