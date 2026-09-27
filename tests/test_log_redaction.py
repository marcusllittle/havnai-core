import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import app


def test_output_access_log_redacts_static_output_filename():
    raw = '127.0.0.1 - - "GET /static/outputs/audio/job-secret.mp3?probe=1 HTTP/1.1" 404 -'

    redacted = app.redact_output_access_log_text(raw)

    assert "job-secret.mp3" not in redacted
    assert "/static/outputs/[output-redacted]" in redacted
    assert " 404 -" in redacted


def test_output_access_log_filter_redacts_message_args():
    record = logging.LogRecord(
        "werkzeug",
        logging.INFO,
        __file__,
        1,
        '%s "%s" %s',
        ("127.0.0.1", "GET /api/static/outputs/audio/job-secret.wav HTTP/1.1", "404 -"),
        None,
    )

    assert app.OutputAccessLogRedactionFilter().filter(record) is True

    message = record.getMessage()
    assert "job-secret.wav" not in message
    assert "/api/static/outputs/[output-redacted]" in message
    assert "404 -" in message
