"""Minimal stand-in for comfy.utils — see tests/_stubs/folder_paths.py for
why this exists. ProgressBar is only ever used at runtime inside generate()
methods (never at import time), so a permissive no-op is sufficient."""


class ProgressBar:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def update_absolute(self, *args, **kwargs):
        pass
