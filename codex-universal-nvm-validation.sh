#!/usr/bin/env bash
set -euo pipefail

original="$CANDIDATE/setup_universal.sh"
project="$RUNNER_TEMP/nvm-project"
mkdir -p "$project"
cd "$project"

run_setup() {
    CODEX_ENV_NODE_VERSION=20 bash -c '
        . "$NVM_DIR/nvm.sh" --no-use
        nvm use 22
        . "$1"
        [[ "$(node --version)" == v20.* ]]
    ' _ "$original"
}

printf '18\n' > .nvmrc
run_setup
test "$(cat .nvmrc)" != 18
echo 'Confirmed upstream overwrites an existing workspace .nvmrc.'
rm .nvmrc
run_setup
test -f .nvmrc
echo 'Confirmed upstream creates a workspace .nvmrc when absent.'

git -C "$CANDIDATE" apply --check "$GITHUB_WORKSPACE/harness/codex-universal-nvm.patch"
git -C "$CANDIDATE" apply "$GITHUB_WORKSPACE/harness/codex-universal-nvm.patch"
bash -n "$CANDIDATE/setup_universal.sh"
bash -n "$CANDIDATE/verify.sh"

printf '18\n' > .nvmrc
run_setup
test "$(cat .nvmrc)" = 18
rm .nvmrc
run_setup
test ! -e .nvmrc

fresh_node=$(env -u NVM_BIN -u NVM_INC PATH=/usr/bin:/bin bash -c '. "$NVM_DIR/nvm.sh"; node --version')
[[ "$fresh_node" == v20.* ]]
echo 'Patched setup preserves both workspace states and selects Node 20 in a fresh shell.'
