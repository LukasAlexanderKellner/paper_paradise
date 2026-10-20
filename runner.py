"""
Weekly newsletter pipeline: fetch all sources, dedupe, filter already-sent items,
optionally add a Gemini-written summary, render and send it via SMTP or as a Brevo campaign
(config.toml: [newsletter] delivery = "smtp" | "brevo").

    python runner.py --dry-run     # writes out/newsletter-<date>.html/.txt, sends nothing, keeps state
    python runner.py --test-to me@example.com   # real send to one address only, keeps state
                                                # (Brevo: the address must exist as a contact)
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
from pipeline import brevo_mailer
from pipeline.brevo_mailer import BrevoClient, BrevoConfig, BrevoConfigError
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
            logger.info("%s: %d candidates", source, len(items_by_source[source]))
        except Exception as e:
            logger.error("%s failed: %s: %s", source, type(e).__name__, e)
            failed.append(source)
    return items_by_source, failed


def select(settings: Settings, items_by_source: dict[str, list[Item]], state: dict[str, str]) -> dict[str, list[Item]]:
    unique = dedupe(items_by_source)
    unsent = filter_unsent(unique, state)
    return {source: items[: settings.newsletter.top_k[source]] for source, items in unsent.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weekly paper_paradise newsletter")
    parser.add_argument("--dry-run", action="store_true",
                        help="write HTML/text locally instead of sending, leave the state file unchanged")
    parser.add_argument("--test-to", metavar="ADDRESS",
                        help="test send to this address only, with [TEST] in the subject, leave the state file unchanged")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "out", help="output folder for --dry-run")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM step")
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    args = parser.parse_args(argv)
    if args.dry_run and args.test_to:
        parser.error("--dry-run and --test-to are mutually exclusive")

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("runner", "core", "pipeline"):
        logging.getLogger(name).setLevel(logging.INFO)

    load_dotenv(ROOT_DIR / ".env")
    settings = load_settings(args.config)
    now = datetime.now(timezone.utc)

    delivery = settings.newsletter.delivery
    if delivery not in ("smtp", "brevo"):
        logger.error('Unknown [newsletter] delivery "%s" in config.toml, use "smtp" or "brevo".', delivery)
        return EXIT_CONFIG_ERROR

    # Check mail settings before the slow fetch, so a misconfigured run fails fast
    if not args.dry_run:
        try:
            if delivery == "brevo":
                brevo = BrevoClient(BrevoConfig.from_env(settings.brevo))
            else:
                smtp = SmtpConfig.from_env()
                recipients = [args.test_to] if args.test_to else recipients_from_env()
                sender = resolve_sender(settings.newsletter.sender, smtp)
        except (MailConfigError, BrevoConfigError) as e:
            logger.error("%s", e)
            return EXIT_CONFIG_ERROR

    items_by_source, failed = fetch_all(settings, now)
    if not items_by_source:
        logger.error("All sources failed, nothing will be sent.")
        return EXIT_ALL_SOURCES_FAILED

    state = load_state(settings.newsletter.state_file)
    selected = select(settings, items_by_source, state)
    all_items = [item for items in selected.values() for item in items]
    if not all_items:
        logger.info("No new items since the last send, nothing will be sent.")
        return EXIT_OK

    annotations = annotate(settings, all_items) if not args.no_llm else None
    if annotations:
        for item in all_items:
            item.why = annotations.why.get(item.key)
            # Blog teasers belong to their publishers: show Gemini's own-words version instead
            if item.source == "blogs" and annotations.generated:
                item.summary = annotations.rewrites.get(item.key, "")

    html, text = render_newsletter(NewsletterContext(
        sections=selected,
        period_start=now - timedelta(days=settings.newsletter.window_days),
        period_end=now,
        failed_sources=failed,
        headline=annotations.headline if annotations else None,
        summary=annotations.summary if annotations else None,
        highlights=annotations.highlights if annotations else [],
        glossary=annotations.glossary if annotations else [],
        ai_notice=bool(annotations and annotations.generated),
        imprint_url=settings.newsletter.imprint_url,
        # Brevo fills in each contact's personal unsubscribe link
        unsubscribe_url=brevo_mailer.UNSUBSCRIBE_PLACEHOLDER if delivery == "brevo" else "",
    ))

    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / f"newsletter-{now:%Y-%m-%d}"
        stem.with_suffix(".html").write_text(html, encoding="utf-8")
        stem.with_suffix(".txt").write_text(text, encoding="utf-8")
        logger.info("Dry run: wrote %d items to %s.html/.txt, state unchanged.", len(all_items), stem)
        return EXIT_OK

    subject = f"[TEST] {settings.newsletter.subject}" if args.test_to else settings.newsletter.subject
    try:
        if delivery == "brevo":
            name = f"paper_paradise {now:%Y-%m-%d}" + (" TEST" if args.test_to else "")
            preview = annotations.headline if annotations else None
            if args.test_to:
                brevo_mailer.send_test(brevo, [args.test_to], name, subject, html, preview)
                logger.info("Sent Brevo test of %d items to 1 address.", len(all_items))
            else:
                campaign_id = brevo_mailer.send_campaign(brevo, name, subject, html, preview)
                logger.info("Brevo campaign %s with %d items sent to list(s) %s.",
                            campaign_id, len(all_items), settings.brevo.list_ids)
        else:
            msg = build_message(subject, sender, html, text)
            send_message(msg, recipients, smtp)
            logger.info("Sent newsletter with %d items to %d recipient(s).", len(all_items), len(recipients))
    except Exception as e:
        logger.error("Sending failed, state unchanged: %s: %s", type(e).__name__, e)
        return EXIT_SEND_FAILED

    if args.test_to:
        logger.info("Test send: state file unchanged.")
        return EXIT_OK

    # Only after a successful send
    new_state = mark_sent(state, all_items, settings.newsletter.state_retention_days, now)
    save_state(settings.newsletter.state_file, new_state)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
