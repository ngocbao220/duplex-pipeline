#!/usr/bin/env bash
# dialogue-split.sh wrapper for dialogue_split.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_ROOT/dialogue_split.sh" "$@"
