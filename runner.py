"""
Weekly newsletter pipeline: fetch all sources, dedupe, filter already-sent items,
optionally annotate with Claude, render and send via SMTP.

    python runner.py --dry-run     # writes out/newsletter-<date>.html/.txt, sends nothing, keeps state
    python runner.py --test-to me@example.com   # real send to one address only, keeps state
    python runner.py               # sends the mail and updates the state file
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from core import sources
from core.config import ROOT_DIR, Settings, load_settings
from core.models import Item
from pipeline.dedupe import dedupe
from pipeline.llm import annotate
from pipeline.mailer import (
    MailConfigError,
    SmtpConfig,
    build_message,
    recipients_from_env,
    resolve_sender,
    send_message,
)
from pipeline.render import NewsletterContext, render_newsletter
from pipeline.state import filter_unsent, load_state, mark_sent, save_state

logger = logging.getLogger("runner")

# Fetch more candidates than needed, so dedupe and the state filter don't empty a section
CANDIDATE_FACTOR = 4

EXIT_OK = 0
EXIT_ALL_SOURCES_FAILED = 2
EXIT_SEND_FAILED = 3
EXIT_CONFIG_ERROR = 4


def fetch_all(settings: Settings, now: datetime) -> tuple[dict[str, list[Item]], list[str]]:
    """Fetch every source for the configured window. Returns (items_by_source, failed_sources)."""
    nl = settings.newsletter
    since = now - timedelta(days=nl.window_days)
    fetchers = {
        "huggingface": lambda k: sources.fetch_huggingface(settings, top_k=k, since=since.date(), until=now.date()),
        "arxiv": lambda k: sources.fetch_arxiv(settings, max_fetch=nl.arxiv_max_fetch, top_k=k, since=since, until=now),
        "github": lambda k: sources.fetch_github(
            settings, top_k=k, days=nl.window_days, min_stars=nl.github_min_stars, max_fetch=nl.github_max_fetch
        ),
        # Feed dates are naive local time (see core.sources.fetch_blogs)
        "blogs": lambda k: sources.fetch_blogs(settings, top_k=k, since=datetime.now() - timedelta(days=nl.window_days)),
    }

    items_by_source: dict[str, list[Item]] = {}
    failed: list[str] = []
    for source, fetch in fetchers.items():
        try:
            items_by_source[source] = fetch(nl.top_k[source] * CANDIDATE_FACTOR)
            logger.info("%s: %d Kandidaten", source, len(items_by_source[source]))
        except Exception as e:
            logger.error("%s fehlgeschlagen: %s: %s", source, type(e).__name__, e)
            failed.append(source)
    return items_by_source, failed


def select(settings: Settings, items_by_source: dict[str, list[Item]], state: dict[str, str]) -> dict[str, list[Item]]:
    unique = dedupe(items_by_source)
    unsent = filter_unsent(unique, state)
    return {source: items[: settings.newsletter.top_k[source]] for source, items in unsent.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wöchentlicher paper_porn-Newsletter")
    parser.add_argument("--dry-run", action="store_true",
                        help="HTML/Text lokal schreiben statt zu versenden, State-Datei nicht verändern")
    parser.add_argument("--test-to", metavar="ADRESSE",
                        help="Testversand nur an diese Adresse, mit [TEST] im Betreff, State-Datei nicht verändern")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "out", help="Zielordner für --dry-run")
    parser.add_argument("--no-llm", action="store_true", help="LLM-Schritt überspringen")
    parser.add_argument("--config", type=Path, default=None, help="Pfad zur config.toml")
    args = parser.parse_args(argv)
    if args.dry_run and args.test_to:
        parser.error("--dry-run und --test-to schließen sich aus")

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("runner", "core", "pipeline"):
        logging.getLogger(name).setLevel(logging.INFO)

    load_dotenv(ROOT_DIR / ".env")
    settings = load_settings(args.config)
    now = datetime.now(timezone.utc)

    # Check mail settings before the slow fetch, so a misconfigured run fails fast
    if not args.dry_run:
        try:
            smtp = SmtpConfig.from_env()
            recipients = [args.test_to] if args.test_to else recipients_from_env()
            sender = resolve_sender(settings.newsletter.sender, smtp)
        except MailConfigError as e:
            logger.error("%s", e)
            return EXIT_CONFIG_ERROR

    items_by_source, failed = fetch_all(settings, now)
    if not items_by_source:
        logger.error("Alle Quellen sind ausgefallen, es wird nichts verschickt.")
        return EXIT_ALL_SOURCES_FAILED

    state = load_state(settings.newsletter.state_file)
    selected = select(settings, items_by_source, state)
    all_items = [item for items in selected.values() for item in items]
    if not all_items:
        logger.info("Keine neuen Einträge seit dem letzten Versand, es wird nichts verschickt.")
        return EXIT_OK

    annotations = annotate(settings, all_items) if not args.no_llm else None
    if annotations:
        for item in all_items:
            item.why = annotations.why.get(item.key)

    html, text = render_newsletter(NewsletterContext(
        sections=selected,
        period_start=now - timedelta(days=settings.newsletter.window_days),
        period_end=now,
        failed_sources=failed,
        editorial=annotations.editorial if annotations else None,
    ))

    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"newsletter-{now:%Y-%m-%d}"
        stem.with_suffix(".html").write_text(html, encoding="utf-8")
        stem.with_suffix(".txt").write_text(text, encoding="utf-8")
        logger.info("Dry-Run: %d Einträge nach %s.html/.txt geschrieben, State unverändert.", len(all_items), stem)
        return EXIT_OK

    subject = f"[TEST] {settings.newsletter.subject}" if args.test_to else settings.newsletter.subject
    msg = build_message(subject, sender, html, text)
    try:
        send_message(msg, recipients, smtp)
    except Exception as e:
        logger.error("Versand fehlgeschlagen, State bleibt unverändert: %s: %s", type(e).__name__, e)
        return EXIT_SEND_FAILED
    logger.info("Newsletter mit %d Einträgen an %d Empfänger verschickt.", len(all_items), len(recipients))

    if args.test_to:
        logger.info("Testversand: State-Datei bleibt unverändert.")
        return EXIT_OK

    # Only after a successful send
    new_state = mark_sent(state, all_items, settings.newsletter.state_retention_days, now)
    save_state(settings.newsletter.state_file, new_state)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
