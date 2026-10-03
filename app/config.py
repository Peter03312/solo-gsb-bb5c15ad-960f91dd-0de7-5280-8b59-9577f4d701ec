"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

import math
import os


class Settings:
    def __init__(self) -> None:
        # SQLAlchemy URL, e.g. postgresql+psycopg2://radio:radio@db:5432/radio
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://radio:radio@localhost:5432/radio",
        )
        # Non-negative guard band, in the same unit as the frequency numbers
        # (the API itself is unit-agnostic; use MHz, kHz, ... consistently).
        # This is only the INITIAL value seeded into a brand-new database;
        # the live setting is stored in the database and changed online via
        # /api/config/guard-band, so restarts must not overwrite it.
        guard_raw = os.getenv("GUARD_BAND", "0")
        self.guard_band: float = float(guard_raw)
        if not math.isfinite(self.guard_band) or self.guard_band < 0:
            raise ValueError(
                f"GUARD_BAND must be a finite, non-negative number, got {guard_raw}"
            )

        self.app_host: str = os.getenv("APP_HOST", "0.0.0.0")
        self.app_port: int = int(os.getenv("APP_PORT", "8000"))

        # How often the in-process due-approval worker scans for scheduled
        # appointments whose UTC instant has arrived (seconds).  Due
        # appointments are also processed once synchronously at startup
        # (restart catch-up), so a larger interval only delays the steady
        # state.  Must be a positive finite number.
        poll_raw = os.getenv("SCHEDULED_APPROVAL_POLL_SECONDS", "0.25")
        self.scheduled_poll_seconds: float = float(poll_raw)
        if (
            not math.isfinite(self.scheduled_poll_seconds)
            or self.scheduled_poll_seconds <= 0
        ):
            raise ValueError(
                "SCHEDULED_APPROVAL_POLL_SECONDS must be a finite, positive "
                f"number, got {poll_raw}"
            )


settings = Settings()
