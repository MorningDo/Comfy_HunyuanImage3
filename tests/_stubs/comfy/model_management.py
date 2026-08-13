"""Minimal stand-in for comfy.model_management — see
tests/_stubs/folder_paths.py for why this exists. Every call site in this
repo guards these with hasattr() before calling, so no-ops are sufficient;
provided anyway so `COMFYUI_AVAILABLE`-style guarded imports elsewhere in
this repo succeed rather than silently degrading."""


def unload_all_models():
    pass


def soft_empty_cache():
    pass


def cleanup_models():
    pass
