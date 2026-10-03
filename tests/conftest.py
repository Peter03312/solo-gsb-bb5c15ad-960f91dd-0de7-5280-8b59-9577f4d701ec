"""Test bootstrap: environment must be configured before app modules import."""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="radio-tests-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("GUARD_BAND", "2.0")
# Keep the live due-approval poller from racing the deterministic tests (they
# call process_due_scheduled_approvals() directly); the single background
# end-to-end test shortens the running worker's interval itself.
os.environ.setdefault("SCHEDULED_APPROVAL_POLL_SECONDS", "3600")

import pytest
from fastapi.testclient import TestClient

from app.db import engine, init_db
from app.main import app
from app.models import Base


@pytest.fixture()
def client():
    # Fresh schema for every test.
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as test_client:
        yield test_client
