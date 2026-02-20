.PHONY: sidecar dev build clean install

# Detect architecture for Tauri sidecar naming convention
ARCH := $(shell uname -m)
ifeq ($(ARCH),arm64)
    TARGET_TRIPLE := aarch64-apple-darwin
else
    TARGET_TRIPLE := x86_64-apple-darwin
endif

# Install all dependencies (Python + Node)
install:
	uv sync
	npm install

# Build the PyInstaller sidecar binary
sidecar:
	uv run pyinstaller knwler-server.spec --noconfirm
	mkdir -p src-tauri/binaries
	cp dist/knwler-server src-tauri/binaries/knwler-server-$(TARGET_TRIPLE)
	chmod +x src-tauri/binaries/knwler-server-$(TARGET_TRIPLE)

# Run in dev mode (sidecar must be built first)
dev: sidecar
	npm run dev

# Build the production .app / .dmg
build: sidecar
	npm run build

# Run just the FastAPI server for development (without Tauri)
server:
	uv run python server.py --port 8765

# Clean build artifacts
clean:
	rm -rf dist build __pycache__
	rm -rf src-tauri/target
	rm -rf src-tauri/binaries/knwler-server-*
	rm -rf node_modules
