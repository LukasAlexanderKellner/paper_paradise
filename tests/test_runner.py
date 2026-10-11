"""End-to-end runner tests with mocked sources and a mocked SMTP server (no network)."""
import json

import pytest

import runner
from core.models import SourceError
from pipeline.llm import Annotations


@pytest.fixture
def env(monkeypatch, tmp_path, sample_items):
    state_file = tmp_path / "state.json"
    config = tmp_path / "config.toml"
    original = (runner.ROOT_DIR / "config.toml").read_text()
    config.write_text(original.replace('state_file = "state/sent_items.json"', f'state_file = "{state_file}"'))

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
