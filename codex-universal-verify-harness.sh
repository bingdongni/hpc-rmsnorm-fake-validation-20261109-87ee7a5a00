#!/usr/bin/env bash
set -euo pipefail

# Keep Node/nvm real; unrelated language runtimes are fixtures on this runner.
fixture="$RUNNER_TEMP/codex-runtime-fixtures"
mkdir -p "$fixture"
cat > "$fixture/runtime" <<'EOF'
#!/usr/bin/env bash
case "$(basename "$0")" in
  ruby) echo 'ruby 1.0.0p0' ;;
  rustc) echo 'rustc 1.0' ;;
  go) echo 'go version go1.0 linux/amd64' ;;
  swift) echo 'Swift version 1.0' ;;
  java) echo 'java version "1.0.0"' ;;
  php) echo '1.0' ;;
  *) echo 'runtime fixture' ;;
esac
EOF
chmod +x "$fixture/runtime"
for tool in pyenv ruby rustc rustup go swift swiftly mise php phpenv java javac gradle mvn pnpm yarn bun composer elixir erl; do
    ln -s runtime "$fixture/$tool"
done

sudo install -d /opt/codex
printf 'export PATH="%s:$PATH"\n[ -s "${NVM_DIR:-}/nvm.sh" ] && . "$NVM_DIR/nvm.sh"\n' "$fixture" |
    sudo tee /etc/profile.d/codex-nvm-validation.sh > /dev/null
export PATH="$fixture:$PATH"
. "$NVM_DIR/nvm.sh" --no-use
nvm alias default 22
nvm use 22
export PYTHON_VERSIONS=3.12 NODE_VERSIONS='22 20' RUST_VERSIONS=1.0 GO_VERSIONS=1.0
export SWIFT_VERSIONS=1.0 RUBY_VERSIONS=1.0 PHP_VERSIONS=1.0 JAVA_VERSIONS=1

git -C "$CANDIDATE" show HEAD:setup_universal.sh > "$RUNNER_TEMP/upstream-setup.sh"
sudo install -m 755 "$RUNNER_TEMP/upstream-setup.sh" /opt/codex/setup_universal.sh
if bash --login "$CANDIDATE/verify.sh" > "$RUNNER_TEMP/verify-upstream.log" 2>&1; then
    echo 'Regression test unexpectedly passed against upstream.' >&2
    exit 1
fi
grep -F 'setup_universal.sh modified the workspace .nvmrc' "$RUNNER_TEMP/verify-upstream.log"

sudo install -m 755 "$CANDIDATE/setup_universal.sh" /opt/codex/setup_universal.sh
if ! bash --login "$CANDIDATE/verify.sh" > "$RUNNER_TEMP/verify-patched.log" 2>&1; then
    cat "$RUNNER_TEMP/verify-patched.log"
    exit 1
fi
grep -F 'All language runtimes detected successfully.' "$RUNNER_TEMP/verify-patched.log"
echo 'The image verification script detects the regression and passes with the patch.'
