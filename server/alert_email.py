"""SMTP transport for operator alerts; credentials never enter delivery reports."""
import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import parseaddr


def send_alert(payload):
    result = {"mode": "email", "sent": False, "configured": False}
    names = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "HAVNAI_ALERT_FROM", "HAVNAI_ALERT_TO")
    values = {name: os.environ.get(name, "").strip() for name in names}
    values["SMTP_PASSWORD"] = os.environ.get("SMTP_PASSWORD", "")
    if not all(values.values()):
        return {**result, "reason": "alert_email_not_configured"}
    try:
        port = int(os.environ.get("SMTP_PORT", "587"))
        security = os.environ.get("SMTP_SECURITY", "starttls").strip().lower()
        if not 1 <= port <= 65535 or security not in {"starttls", "ssl"}:
            raise ValueError("invalid SMTP configuration")
        for name in ("HAVNAI_ALERT_FROM", "HAVNAI_ALERT_TO"):
            address = values[name]
            if "\r" in address or "\n" in address or parseaddr(address)[1] != address or "@" not in address:
                raise ValueError("one mailbox required")
        matched = [rule for rule in payload.get("rules", []) if rule.get("matched")]
        if not matched:
            return {**result, "configured": True, "reason": "no_matched_alerts"}
        message = EmailMessage()
        message["From"] = values["HAVNAI_ALERT_FROM"]
        message["To"] = values["HAVNAI_ALERT_TO"]
        message["Subject"] = "HavnAI service alert"
        # Include rule summaries only, not worker details, paths or full snapshots.
        message.set_content("HavnAI monitoring detected:\n\n" + "\n".join(
            f"{rule.get('severity', 'warning')}: {rule.get('code', 'unknown')}"
            for rule in matched
        ))
    except (ValueError, TypeError):
        return {**result, "reason": "invalid_alert_email_configuration"}

    result["configured"] = True
    try:
        context = ssl.create_default_context()
        if security == "ssl":
            client = smtplib.SMTP_SSL(values["SMTP_HOST"], port, timeout=10, context=context)
        else:
            client = smtplib.SMTP(values["SMTP_HOST"], port, timeout=10)
        with client:
            if security == "starttls":
                client.ehlo()
                client.starttls(context=context)
                client.ehlo()
            client.login(values["SMTP_USER"], values["SMTP_PASSWORD"])
            refused = client.send_message(message)
            if refused:
                return {**result, "reason": "delivery_failed"}
    except (OSError, smtplib.SMTPException):
        return {**result, "reason": "delivery_failed"}
    return {**result, "sent": True, "receipt_scope": "smtp_accepted"}
