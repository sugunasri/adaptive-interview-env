"""SessionStore — persists DomainSkillProfile across episodes (V2).

Teammate 1 owns this file.
"""
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SessionStore:
    """Persists DomainSkillProfile to a JSON file keyed by session_id.

    Storage format:
    {
      "session_abc123": {
        "profile": {"algorithms": {"correctness": 0.7, ...}, ...},
        "total_episodes_completed": 5,
        "last_updated": "2024-01-01T00:00:00Z"
      }
    }
    """

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._data: dict = {}
        self._load_store()

    def _load_store(self) -> None:
        """Load existing store from disk, or start fresh."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path) as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.warning(f"SessionStore: failed to load {self.store_path}: {e}. Starting fresh.")
                self._data = {}
        else:
            self._data = {}

    def _flush(self) -> None:
        """Write current store to disk."""
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def exists(self, session_id: str) -> bool:
        return session_id in self._data

    def load(self, session_id: str):
        """Return DomainSkillProfile for session_id, or None if not found."""
        # TODO (Teammate 1): deserialize profile dict into DomainSkillProfile
        if session_id not in self._data:
            return None
        from .skill_profile import DomainSkillProfile
        profile_data = self._data[session_id].get("profile", {})
        return DomainSkillProfile.from_dict(profile_data)

    def save(self, session_id: str, profile, episodes_completed: int) -> None:
        """Persist DomainSkillProfile for session_id."""
        # TODO (Teammate 1): serialize DomainSkillProfile to dict
        self._data[session_id] = {
            "profile": profile.to_dict(),
            "total_episodes_completed": episodes_completed,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self._flush()

    def get_episodes_completed(self, session_id: str) -> int:
        """Return total episodes completed for a session."""
        if session_id not in self._data:
            return 0
        return self._data[session_id].get("total_episodes_completed", 0)
