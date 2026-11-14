# PyInstaller spec for the Llama RPC Farm web app (source + frozen builds).
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
datas = [
    (os.path.join(ROOT, "ui"), "ui"),
    (os.path.join(ROOT, "config", "settings.example.yaml"), "config"),
]
agent_wheels = os.path.join(ROOT, "agent-wheel")
if os.path.isdir(agent_wheels):
    datas.append((agent_wheels, "agent-wheel"))

binaries = []
hiddenimports = []
for pkg in ("uvicorn", "fastapi", "starlette", "anyio", "pydantic"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass
# embedded :8090 gateway lives in src/swap_gateway.py; import it eagerly
# so PyInstaller's analysis can never miss the dynamic import chain.
hiddenimports.append("swap_gateway")
# hf_xet is a native Rust extension imported dynamically by
# huggingface_hub for Xet-backed repos (required for huge GGUF downloads).
try:
    _hx_datas, _hx_binaries, _hx_hidden = collect_all("hf_xet")
    datas += _hx_datas
    binaries += _hx_binaries
    hiddenimports += _hx_hidden
except Exception:
    hiddenimports.append("hf_xet")

a = Analysis(
    [os.path.join(ROOT, "packaging", "app_entry.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="llama-rpc-farm",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
