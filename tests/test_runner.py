"""End-to-end runner tests with mocked sources and a mocked SMTP server (no network)."""
import json
from types import SimpleNamespace

import pytest

import runner
from core.models import SourceError
from pipeline.llm import Annotations
from tests.conftest import make_item


@pytest.fixture
def env(monkeypatch, tmp_path, sample_items):
    state_file = tmp_path / "state.json"
    config = tmp_path / "config.toml"
    original = (runner.ROOT_DIR / "config.toml").read_text()
    config.write_text(
        original.replace('state_file = "state/sent_items.json"', f'state_file = "{state_file.as_posix()}"')
        .replace('delivery = "brevo"', 'delivery = "smtp"')
    )

    for var, value in {"SMTP_HOST": "smtp.test", "SMTP_PORT": "587", "SMTP_USER": "bot@test",
                       "SMTP_PASSWORD": "x", "NEWSLETTER_RECIPIENTS": "a@test, b@test"}.items():
        monkeypatch.setenv(var, value)
    monkeypatch.delenv("NEWSLETTER_SENDER", raising=False)
    monkeypatch.setattr(runner, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(runner, "annotate", lambda settings, items: Annotations())

    def ok(name):
        return lambda settings, **kw: [i for i in sample_items[name]]

    monkeypatch.setattr(runner.sources, "fetch_huggingface", ok("huggingface"))
    monkeypatch.setattr(runner.sources, "fetch_arxiv", ok("arxiv"))
    monkeypatch.setattr(runner.sources, "fetch_github", ok("github"))
    monkeypatch.setattr(runner.sources, "fetch_blogs", ok("blogs"))

    sent = []

    class FakeSMTP:
        fail = False

        def __init__(self, host, port, timeout):
            self.host, self.port = host, port

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg, to_addrs):
            if FakeSMTP.fail:
                raise OSError("connection refused")
            sent.append((msg, to_addrs))

    monkeypatch.setattr("pipeline.mailer.smtplib.SMTP", FakeSMTP)
    return {"config": config, "state": state_file, "sent": sent, "smtp": FakeSMTP, "tmp": tmp_path}


def run(env, *args):
    return runner.main(["--config", str(env["config"]), *args])


def fail(*a, **k):
    raise SourceError("down")


def test_send_updates_state_and_hides_recipients(env):
    assert run(env) == 0
    assert len(env["sent"]) == 1
    msg, to_addrs = env["sent"][0]
    assert to_addrs == ["a@test", "b@test"]
    assert "a@test" not in str(msg["To"]) and msg["From"] == "bot@test"
    assert msg.get_body(("plain",)) is not None and msg.get_body(("html",)) is not None
    state = json.loads(env["state"].read_text())
    # HF/arXiv duplicate counts once
    assert set(state) == {"arxiv:2610.00001", "arxiv:2610.00002", "https://github.com/a/b", "https://pytorch.org/blog/x"}


def test_second_run_sends_nothing_new(env):
    assert run(env) == 0
    assert run(env) == 0
    assert len(env["sent"]) == 1


def test_send_failure_keeps_state(env):
    env["smtp"].fail = True
    assert run(env) == runner.EXIT_SEND_FAILED
    assert not env["state"].exists()


def test_dry_run_writes_files_and_keeps_state(env, monkeypatch):
    for var in ("SMTP_HOST", "SMTP_PASSWORD", "NEWSLETTER_RECIPIENTS"):
        monkeypatch.delenv(var)  # dry run needs no mail config
    out = env["tmp"] / "out"
    assert run(env, "--dry-run", "--output-dir", str(out)) == 0
    assert env["sent"] == []
    assert not env["state"].exists()
    html = next(out.glob("*.html")).read_text()
    assert "</html>" in html and "Shared Paper" in html


def test_one_source_down_is_named_in_mail(env, monkeypatch):
    monkeypatch.setattr(runner.sources, "fetch_github", fail)
    assert run(env) == 0
    msg, _ = env["sent"][0]
    assert "Not available in this issue: GitHub" in msg.get_body(("plain",)).get_content()


def test_all_sources_down_sends_nothing(env, monkeypatch):
    for name in ("fetch_huggingface", "fetch_arxiv", "fetch_github", "fetch_blogs"):
        monkeypatch.setattr(runner.sources, name, fail)
    assert run(env) == runner.EXIT_ALL_SOURCES_FAILED
    assert env["sent"] == []
    assert not env["state"].exists()


def test_missing_recipients_fails_before_fetching(env, monkeypatch):
    monkeypatch.delenv("NEWSLETTER_RECIPIENTS")
    monkeypatch.setattr(runner, "fetch_all", lambda *a: pytest.fail("should not fetch"))
    assert run(env) == runner.EXIT_CONFIG_ERROR


def test_test_to_sends_only_there_and_keeps_state(env, monkeypatch):
    monkeypatch.delenv("NEWSLETTER_RECIPIENTS")  # not needed for a test send
    assert run(env, "--test-to", "me@test") == 0
    msg, to_addrs = env["sent"][0]
    assert to_addrs == ["me@test"]
    assert msg["Subject"].startswith("[TEST] ")
    assert not env["state"].exists()


def test_dry_run_and_test_to_are_exclusive(env):
    with pytest.raises(SystemExit):
        run(env, "--dry-run", "--test-to", "me@test")



# --- Brevo delivery ---------------------------------------------------------------------------

class FakeBrevoSession:
    """Records Brevo API calls and answers like the real API."""

    def __init__(self, fail_on=None):
        self.headers, self.calls, self.fail_on = {}, [], fail_on

    def request(self, method, url, timeout=None, json=None):
        path = url.removeprefix("https://api.brevo.com/v3")
        self.calls.append((method, path, json))
        if self.fail_on and self.fail_on in path:
            return SimpleNamespace(status_code=400, json=lambda: {"code": "invalid_parameter", "message": "nope"}, text="")
        if method == "POST" and path == "/emailCampaigns":
            return SimpleNamespace(status_code=201, json=lambda: {"id": 42}, text="")
        return SimpleNamespace(status_code=204, json=lambda: {}, text="")


@pytest.fixture
def brevo_env(env, monkeypatch):
    env["config"].write_text(env["config"].read_text().replace('delivery = "smtp"', 'delivery = "brevo"'))
    monkeypatch.setenv("BREVO_API_KEY", "test-key")
    for var in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "NEWSLETTER_RECIPIENTS"):
        monkeypatch.delenv(var)  # Brevo needs none of them
    session = FakeBrevoSession()
    monkeypatch.setattr(runner.brevo_mailer.requests, "Session", lambda: session)
    env["brevo"] = session
    return env


def test_brevo_creates_campaign_for_list_sends_it_and_updates_state(brevo_env):
    assert run(brevo_env) == 0
    calls = brevo_env["brevo"].calls
    assert [(m, p) for m, p, _ in calls] == [("POST", "/emailCampaigns"), ("POST", "/emailCampaigns/42/sendNow")]
    body = calls[0][2]
    brevo = runner.load_settings(brevo_env["config"]).brevo
    assert body["recipients"] == {"listIds": brevo.list_ids}
    assert body["sender"] == {"email": brevo.sender_email, "name": brevo.sender_name}
    assert '<a href="{{ unsubscribe }}"' in body["htmlContent"]
    assert "https://www.ws-productions.de/impressum" in body["htmlContent"]
    assert brevo_env["brevo"].headers["api-key"] == "test-key"
    assert brevo_env["sent"] == []  # SMTP untouched
    assert json.loads(brevo_env["state"].read_text())


def test_brevo_send_failure_keeps_state(brevo_env, monkeypatch):
    session = FakeBrevoSession(fail_on="sendNow")
    monkeypatch.setattr(runner.brevo_mailer.requests, "Session", lambda: session)
    assert run(brevo_env) == runner.EXIT_SEND_FAILED
    assert not brevo_env["state"].exists()


def test_brevo_test_send_deletes_draft_and_keeps_state(brevo_env):
    assert run(brevo_env, "--test-to", "me@test") == 0
    calls = brevo_env["brevo"].calls
    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/emailCampaigns"), ("POST", "/emailCampaigns/42/sendTest"), ("DELETE", "/emailCampaigns/42")]
    assert calls[0][2]["subject"].startswith("[TEST] ")
    assert calls[1][2] == {"emailTo": ["me@test"]}
    assert not brevo_env["state"].exists()


def test_brevo_failed_test_send_still_deletes_draft(brevo_env, monkeypatch):
    session = FakeBrevoSession(fail_on="sendTest")
    monkeypatch.setattr(runner.brevo_mailer.requests, "Session", lambda: session)
    assert run(brevo_env, "--test-to", "unknown@test") == runner.EXIT_SEND_FAILED
    assert ("DELETE", "/emailCampaigns/42") in [(m, p) for m, p, _ in session.calls]


def test_brevo_missing_api_key_fails_before_fetching(brevo_env, monkeypatch):
    monkeypatch.delenv("BREVO_API_KEY")
    monkeypatch.setattr(runner, "fetch_all", lambda *a: pytest.fail("should not fetch"))
    assert run(brevo_env) == runner.EXIT_CONFIG_ERROR


def test_smtp_mail_has_imprint_but_no_unsubscribe_placeholder(env):
    assert run(env) == 0
    msg, _ = env["sent"][0]
    html = msg.get_body(("html",)).get_content()
    assert "https://www.ws-productions.de/impressum" in html
    assert "{{ unsubscribe }}" not in html



def test_blog_teaser_is_replaced_by_gemini_rewrite_and_footer_notice(env, monkeypatch):
    monkeypatch.setattr(runner, "annotate", lambda settings, items: Annotations(
        generated=True, rewrites={"https://pytorch.org/blog/x": "Own words about the PyTorch post."}))
    blog_text_before = "Original teaser that must not appear."
    monkeypatch.setattr(runner.sources, "fetch_blogs", lambda settings, **kw: [
        make_item("blogs", "https://pytorch.org/blog/x", title="Blog X", url="https://pytorch.org/blog/x",
                  summary=blog_text_before, meta={"feed": "Blog - PyTorch"})])
    assert run(env) == 0
    text = env["sent"][0][0].get_body(("plain",)).get_content()
    assert "Own words about the PyTorch post." in text
    assert blog_text_before not in text
    assert "AI-generated (Google Gemini)" in text


def test_without_gemini_no_ai_notice(env):
    assert run(env) == 0
    text = env["sent"][0][0].get_body(("plain",)).get_content()
    assert "AI-generated" not in text
