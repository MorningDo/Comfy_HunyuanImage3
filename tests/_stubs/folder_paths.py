"""
Minimal stand-in for ComfyUI's real `folder_paths` module.

Used ONLY as a fallback when a real ComfyUI checkout isn't available (see
conftest.py's `_ensure_comfyui_importable` — a real ComfyUI on sys.path via
--comfyui-dir always takes priority over this). Two files in this package
(hunyuan_quantized_nodes.py, hunyuan_full_bf16_nodes.py) hard-import
`folder_paths` at module scope, and this package's own __init__.py
transitively imports both — so without *something* importable named
`folder_paths`, nothing in this repo can be imported at all, including for
purely-static tests that never touch a real model directory.

This provides just enough of the real API surface (verified by grepping
the whole repo for every `folder_paths.*` call — only these two functions
and one attribute are ever used) for imports to succeed. It is NOT a
functional replacement: model dropdowns built from this will be empty
unless a test explicitly registers a folder via add_model_folder_path().
"""
import os
import tempfile

# Real ComfyUI points this at <ComfyUI>/models. Any writable directory works
# for import-time purposes; nothing in this package writes to it during
# import.
models_dir = os.path.join(tempfile.gettempdir(), "hunyuan_test_stub_models")

_folder_names_and_paths: dict = {}


def add_model_folder_path(folder_name: str, full_folder_path: str, is_default: bool = False) -> None:
    paths = _folder_names_and_paths.setdefault(folder_name, [])
    if full_folder_path in paths:
        return
    if is_default:
        paths.insert(0, full_folder_path)
    else:
        paths.append(full_folder_path)


def get_folder_paths(folder_name: str) -> list:
    return list(_folder_names_and_paths.get(folder_name, []))
