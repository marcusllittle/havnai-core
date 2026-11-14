import smtplib
from unittest.mock import MagicMock

import pytest
from server import alert_email


@pytest.fixture
def smtp(monkeypatch):
    monkeypatch.setattr(alert_email.os, "environ", {
        "SMTP_HOST": "smtp.example.test", "SMTP_USER": "operator",
        "SMTP_PASSWORD": "private-password", "HAVNAI_ALERT_FROM": "alerts@example.test",
        "HAVNAI_ALERT_TO": "team@example.test",
    })
    factory = MagicMock()
    factory.return_value.__enter__.return_value = factory.return_value
    factory.return_value.send_message.return_value = {}
    monkeypatch.setattr(alert_email.smtplib, "SMTP", factory)
    monkeypatch.setattr(alert_email.smtplib, "SMTP_SSL", factory)
    return factory


PAYLOAD = {"rules": [{"matched": True, "code": "gpu_vram_exhaustion", "severity": "critical",
                      "details": {"private_path": "/secret/file"}},
                     {"matched": False, "code": "not_matched"}]}


@pytest.mark.parametrize("security", ["starttls", "ssl"])
def test_encrypted_delivery_redacts_details(smtp, security):
    alert_email.os.environ["SMTP_SECURITY"] = security
    result = alert_email.send_alert(PAYLOAD)
    assert result == {"mode": "email", "configured": True, "sent": True, "receipt_scope": "smtp_accepted"}
    client = smtp.return_value
    assert client.starttls.called == (security == "starttls")
    if security == "starttls":
        assert [call[0] for call in client.method_calls].index("starttls") < [call[0] for call in client.method_calls].index("login")
    message = client.send_message.call_args.args[0].as_string()
    assert "gpu_vram_exhaustion" in message
    assert "private-password" not in message and "/secret/file" not in message and "not_matched" not in message


@pytest.mark.parametrize("setting,value", [("SMTP_SECURITY", "plain"), ("SMTP_PORT", "invalid"),
                                         ("HAVNAI_ALERT_TO", "a@example.test\nb@example.test")])
def test_invalid_configuration_never_connects(smtp, setting, value):
    alert_email.os.environ[setting] = value
    assert alert_email.send_alert(PAYLOAD)["reason"] == "invalid_alert_email_configuration"
    smtp.assert_not_called()


def test_missing_configuration_and_no_alerts_never_connect(smtp):
    assert alert_email.send_alert({"rules": []})["reason"] == "no_matched_alerts"
    del alert_email.os.environ["SMTP_PASSWORD"]
    assert alert_email.send_alert(PAYLOAD)["reason"] == "alert_email_not_configured"
    smtp.assert_not_called()


def test_provider_failure_does_not_leak_credentials(smtp):
    smtp.return_value.login.side_effect = smtplib.SMTPAuthenticationError(535, b"private-password")
    result = alert_email.send_alert(PAYLOAD)
    assert result["sent"] is False and result["reason"] == "delivery_failed"
    assert "private-password" not in str(result)


def test_recipient_refusal_is_not_success(smtp):
    smtp.return_value.send_message.return_value = {"team@example.test": (550, b"rejected")}
    assert alert_email.send_alert(PAYLOAD)["sent"] is False
