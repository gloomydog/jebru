#!/bin/sh
# Register jebru in the desktop application menu (current user).
set -e
BIN="$(command -v jebru || true)"
[ -z "$BIN" ] && BIN="$HOME/.venvs/jebru/bin/jebru"
[ -x "$BIN" ] || { echo "jebru not found: $BIN"; exit 1; }
mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/jebru.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=jebru
Comment=NMR analysis (JEOL / Bruker)
Exec=$BIN %f
Terminal=false
Categories=Science;Chemistry;Education;
MimeType=application/octet-stream;
EOF
echo "Installed: ~/.local/share/applications/jebru.desktop"
