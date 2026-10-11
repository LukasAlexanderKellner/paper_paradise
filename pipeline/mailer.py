import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid


class MailConfigError(Exception):
    """Required SMTP settings or recipients are missing."""


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "SmtpConfig":
        missing = [v for v in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD") if not os.environ.get(v)]
        if missing:
            raise MailConfigError(f"Fehlende Umgebungsvariablen: {', '.join(missing)}")
        return cls(
            host=os.environ["SMTP_HOST"],
            port=int(os.environ["SMTP_PORT"]),
            user=os.environ["SMTP_USER"],
            password=os.environ["SMTP_PASSWORD"],
        )


def recipients_from_env() -> list[str]:
    """Recipients come only from NEWSLETTER_RECIPIENTS (comma-separated) so they never land in the repo."""
    recipients = [r.strip() for r in os.environ.get("NEWSLETTER_RECIPIENTS", "").split(",") if r.strip()]
    if not recipients:
        raise MailConfigError("NEWSLETTER_RECIPIENTS ist nicht gesetzt.")
    return recipients


def resolve_sender(config_sender: str, smtp: SmtpConfig) -> str:
    return os.environ.get("NEWSLETTER_SENDER") or config_sender or smtp.user


def build_message(subject: str, sender: str, html: str, text: str) -> EmailMessage:
    """Multipart mail with plaintext fallback. Recipients are not put in the headers (sent as Bcc)."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = sender
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender.rsplit("@", 1)[-1] if "@" in sender else None)
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")
    return msg


# Local mail catchers (e.g. Mailpit) speak plain SMTP without TLS or auth
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def send_message(msg: EmailMessage, recipients: list[str], smtp: SmtpConfig) -> None:
    """Send via SSL on port 465, otherwise via STARTTLS. Plain SMTP only for local test servers."""
    local = smtp.host in LOCAL_HOSTS
    if smtp.port == 465 and not local:
        server = smtplib.SMTP_SSL(smtp.host, smtp.port, timeout=30)
    else:
        server = smtplib.SMTP(smtp.host, smtp.port, timeout=30)
    with server:
        if not local:
            if smtp.port != 465:
                server.starttls()
            server.login(smtp.user, smtp.password)
        server.send_message(msg, to_addrs=recipients)
