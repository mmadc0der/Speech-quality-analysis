from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pronunciation_backend.models import ReferencePayload


@dataclass
class ReferenceAudioService:
    manifest_path: Path

    def __post_init__(self) -> None:
        self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def get_reference(self, audio_id: str, ipa: str) -> ReferencePayload:
        item = self._manifest.get(audio_id, {})
        return ReferencePayload(
            ipa=ipa,
            audio_id=audio_id,
            asset_path=item.get("asset_path"),
        )
