#!/bin/sh
# Manga Translator installer for Linux and macOS:
#   curl -LsSf https://raw.githubusercontent.com/tuantran00541-spec/manga-translator/main/install.sh | sh
# Run from a clone (./install.sh), it installs that clone in place.
set -eu

REPO="tuantran00541-spec/manga-translator"
REF="${MANGA_REF:-main}"
UV_VERSION="0.12.20"

say() { printf '\n==> %s\n' "$1"; }
fail() { printf '\nLỖI: %s\n' "$1" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "cần lệnh $1"; }

need curl
need tar
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) TRIPLE=x86_64-unknown-linux-gnu; SHA=6590717592ace991ff83a63fef799e3ad9d33ecc8f96c5d6bdd732496e79337f ;;
  Linux-aarch64) TRIPLE=aarch64-unknown-linux-gnu; SHA=8a7aad7bc76a2fae5151566ff3e43eacce0b2a113d5e4de3e4afe3e58fa2441e ;;
  Darwin-arm64) TRIPLE=aarch64-apple-darwin; SHA=848fdeb602ff1a1baacd4f6c8b7bdc6cf1ad026a6d9cf59475fda17c179743ca ;;
  Darwin-x86_64) TRIPLE=x86_64-apple-darwin; SHA=ac54283d211fd77cdc152b67606dbaf6406ff4ab03f3af4ae99468fa8e887141 ;;
  *) fail "chưa hỗ trợ $(uname -s) $(uname -m)" ;;
esac
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

MANGA_HOME="${MANGA_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/manga-translator}"
mkdir -p "$MANGA_HOME"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK:?}"' EXIT

say "Chuẩn bị uv $UV_VERSION (trình cài Python)"
UV="$MANGA_HOME/uv/uv"
if ! "$UV" --version 2>/dev/null | grep -q "$UV_VERSION"; then
  curl -LsSf -o "$WORK/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$TRIPLE.tar.gz"
  [ "$(sha256 "$WORK/uv.tar.gz")" = "$SHA" ] || fail "tải uv bị lỗi (sai mã kiểm tra), hãy chạy lại"
  mkdir -p "$MANGA_HOME/uv"
  tar xzf "$WORK/uv.tar.gz" -C "$MANGA_HOME/uv" --strip-components 1
fi

HERE="$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)"
case "$0" in
  *install.sh) [ -f "$HERE/run.py" ] && [ -f "$HERE/scripts/install.py" ] && SOURCE="$HERE" ;;
esac
if [ -n "${SOURCE:-}" ]; then
  TARGET="$SOURCE"
else
  say "Tải mã nguồn ($REF)"
  curl -LsSf -o "$WORK/source.tar.gz" "https://codeload.github.com/$REPO/tar.gz/refs/heads/$REF"
  mkdir -p "$WORK/source"
  tar xzf "$WORK/source.tar.gz" -C "$WORK/source" --strip-components 1
  SOURCE="$WORK/source"
  TARGET="$MANGA_HOME/app"
fi

UV_PYTHON_INSTALL_DIR="$MANGA_HOME/python" UV_PYTHON_PREFERENCE=only-managed \
  MANGA_UV="$UV" MANGA_HOME="$MANGA_HOME" MANGA_REF="$REF" \
  "$UV" run --no-project --python 3.12 "$SOURCE/scripts/install.py" --target "$TARGET" \
  || fail "cài đặt chưa xong, xem lỗi ở trên rồi chạy lại lệnh cài"
say "Xong. Mở terminal mới và gõ:  manga"
