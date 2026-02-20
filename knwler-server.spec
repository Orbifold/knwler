# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Knwler FastAPI sidecar server."""

a = Analysis(
    ["server.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("languages.json", "."),
        ("templates", "templates"),
    ],
    hiddenimports=[
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "tiktoken_ext.openai_public",
        "tiktoken_ext",
        "multipart",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="knwler-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
