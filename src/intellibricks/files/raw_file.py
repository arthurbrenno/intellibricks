from __future__ import annotations

from architecture.data.files import RawFile as _RawFile


class RawFile(_RawFile, frozen=True, gc=False):
    pass
