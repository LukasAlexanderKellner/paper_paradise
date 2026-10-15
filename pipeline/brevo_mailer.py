"""
Delivery via Brevo email campaigns (https://developers.brevo.com).

Instead of sending the mail itself like pipeline/mailer.py, this creates a Brevo campaign for the
configured contact list(s) and lets Brevo send it. Brevo then handles the unsubscribe link,
bounces, blocklists and statistics.

    create campaign (draft) -> sendNow          regular send to the list
    create campaign (draft) -> sendTest -> delete   test send to single addresses
"""
import logging
import os
from dataclasses import dataclass

import requests

from core.config import BrevoSettings

logger = logging.getLogger(__name__)

BREVO_API_URL = "https://api.brevo.com/v3"
REQUEST_TIMEOUT_SECONDS = 30

# Brevo replaces this placeholder with the contact's personal unsubscribe link.
# Put it into the rendered HTML (see pipeline/render.py, unsubscribe_url).
UNSUBSCRIBE_PLACEHOLDER = "{{ unsubscribe }}"


class BrevoConfigError(Exception):
    """BREVO_API_KEY or the [brevo] settings are missing."""


class BrevoError(Exception):
    """The Brevo API rejected a request."""


@dataclass(frozen=True)
class BrevoConfig:
    api_key: str
    list_ids: list[int]
    sender_email: str
    sender_name: str

    @classmethod
    def from_env(cls, settings: BrevoSettings) -> "BrevoConfig":
        missing = []
        if not os.environ.get("BREVO_API_KEY"):
            missing.append("environment variable BREVO_API_KEY")
        if not settings.list_ids:
            missing.append("[brevo] list_ids in config.toml")
        if not settings.sender_email:
            missing.append("[brevo] sender_email in config.toml")
        if missing:
            raise BrevoConfigError(f"Missing Brevo settings: {', '.join(missing)}")
        return cls(
            api_key=os.environ["BREVO_API_KEY"],
            list_ids=settings.list_ids,
            sender_email=settings.sender_email,
            sender_name=settings.sender_name,
        )


class BrevoClient:
    def __init__(self, config: BrevoConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        self.session.headers.update({
            "api-key": config.api_key,
            "accept": "application/json",
            "content-type": "application/json",
        })

    def _request(self, method: str, path: str, expected: int, **kwargs) -> requests.Response:
        response = self.session.request(method, f"{BREVO_API_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
        if response.status_code != expected:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text[:300]
            raise BrevoError(f"{method} {path} -> {response.status_code}: {detail}")
        return response

    def create_campaign(self, name: str, subject: str, html: str, preview_text: str | None = None) -> int:
        """Create a draft campaign for the configured list(s) and return its ID."""
        sender = {"email": self.config.sender_email}
        if self.config.sender_name:
            sender["name"] = self.config.sender_name
        body = {
            "name": name,
            "subject": subject,
            "sender": sender,
            "htmlContent": html,
            "recipients": {"listIds": self.config.list_ids},
            # No "tag": Brevo rejects it with 405 on plans without the tagging feature
        }
        if preview_text:
            body["previewText"] = preview_text[:150]
        return self._request("POST", "/emailCampaigns", 201, json=body).json()["id"]

    def send_now(self, campaign_id: int) -> None:
        self._request("POST", f"/emailCampaigns/{campaign_id}/sendNow", 204)

    def send_test(self, campaign_id: int, emails: list[str]) -> None:
        # Brevo only accepts addresses that exist as contacts in your account; max. 50 test mails per day
        self._request("POST", f"/emailCampaigns/{campaign_id}/sendTest", 204, json={"emailTo": emails})

    def delete_campaign(self, campaign_id: int) -> None:
        self._request("DELETE", f"/emailCampaigns/{campaign_id}", 204)


def send_campaign(client: BrevoClient, name: str, subject: str, html: str, preview_text: str | None = None) -> int:
    """Create a campaign for the configured list(s) and send it immediately. Returns the campaign ID."""
    campaign_id = client.create_campaign(name, subject, html, preview_text)
    try:
        client.send_now(campaign_id)
    except BrevoError:
        logger.error("Campaign %s was created but not sent; it stays as a draft in Brevo.", campaign_id)
        raise
    return campaign_id


def send_test(client: BrevoClient, emails: list[str], name: str, subject: str, html: str,
              preview_text: str | None = None) -> None:
    """Send a test of the campaign to single addresses, then delete the temporary draft."""
    campaign_id = client.create_campaign(name, subject, html, preview_text)
    try:
        client.send_test(campaign_id, emails)
    finally:
        try:
            client.delete_campaign(campaign_id)
        except BrevoError as e:
            logger.warning("Could not delete test campaign %s: %s", campaign_id, e)
