"""Public telemetry must not mistake local originals for approved previews."""

import pytest

from tests.test_account_jobs import platform, keys
import app


@pytest.mark.parametrize("status", ["queued", "completed", "failed"])
@pytest.mark.parametrize("files_exist", [False, True])
def test_legacy_feed_never_advertises_unapproved_originals(platform, status, files_exist):
    harness, _, _ = platform
    job_id = harness._create_image_job()
    if files_exist:
        (app.OUTPUTS_DIR / f"{job_id}.png").write_bytes(b"unclassified-original")
        (app.OUTPUTS_DIR / "videos").mkdir(exist_ok=True)
        (app.OUTPUTS_DIR / "videos" / f"{job_id}.mp4").write_bytes(b"private-video")
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))
        item = next(row for row in app.get_job_summary()["feed"] if row["job_id"] == job_id)
    for key in ("preview_url", "image_url", "video_url", "output_path"):
        assert item[key] is None


def test_account_owned_job_stays_out_of_public_feed(platform):
    harness, _, owner = platform
    job_id = harness._create_image_job()
    with app.app.app_context():
        conn = app.get_db()
        with conn:
            conn.execute("UPDATE jobs SET owner_account_id=? WHERE id=?", (owner, job_id))
        assert all(row["job_id"] != job_id for row in app.get_job_summary()["feed"])
