#!/bin/bash
# Build a double-clickable macOS launcher (Oeuvre.app) with the app icon.
# The app just runs `uv run oeuvre` from this repo. Drag Oeuvre.app to
# /Applications or the Desktop. Re-run this script after moving the repo.
set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ICON_PNG="$REPO/oeuvre/assets/icon.png"
ICNS="$REPO/oeuvre/assets/oeuvre.icns"
APP="$REPO/Oeuvre.app"

[ -f "$ICON_PNG" ] || { echo "Missing $ICON_PNG"; exit 1; }

# 1. PNG -> .icns (Apple iconset sizes)
ICONSET="$(mktemp -d)/oeuvre.iconset"
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
    sips -z "$s" "$s" "$ICON_PNG" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z "$d" "$d" "$ICON_PNG" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$ICNS"

# 2. .app bundle
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ICNS" "$APP/Contents/Resources/oeuvre.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Oeuvre</string>
  <key>CFBundleDisplayName</key><string>Oeuvre</string>
  <key>CFBundleExecutable</key><string>oeuvre</string>
  <key>CFBundleIconFile</key><string>oeuvre</string>
  <key>CFBundleIdentifier</key><string>co.jothflee.oeuvre</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST

cat > "$APP/Contents/MacOS/oeuvre" <<LAUNCH
#!/bin/bash
export PATH="\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
cd "$REPO" || exit 1
exec uv run oeuvre
LAUNCH
chmod +x "$APP/Contents/MacOS/oeuvre"

echo "Built $APP"
echo "Drag it to /Applications or your Desktop to launch with the icon."
