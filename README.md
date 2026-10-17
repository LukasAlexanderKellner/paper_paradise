# paper_porn

Collects new content from the AI and ML world and filters it against a personal interest profile: arXiv papers, Hugging Face Daily Papers, new GitHub repositories and engineering blog posts.

The project has two entry points built on the same logic:

- **MCP server** (`paperporn_mcp.py`): Claude Desktop or any other MCP client calls the sources as tools on demand.
- **Newsletter pipeline** (`runner.py`): Once a week, collects all sources over the past 7 days, removes duplicates and items that were already sent, and sends an HTML email, either via SMTP or as a [Brevo](https://www.brevo.com) campaign to a subscriber list. Runs automatically via GitHub Actions every Monday at 9:30 (Europe/Berlin).

## Sources

| Source | Selection | MCP tool |
|---|---|---|
| arXiv | Newest papers in the configured categories, ranked by semantic similarity to the profile | `fetch_filtered_arxiv_papers` |
| Hugging Face Daily Papers | Ranked by community upvotes | `fetch_huggingface_daily_papers` |
| GitHub | New repos with at least `min_stars` stars, ranked by semantic similarity | `fetch_github_trending_ml_repos` |
| Blogs (PyTorch, Hugging Face, AlphaSignal) | Latest posts, alternating across all feeds | `fetch_ai_engineering_blogs` |

Every item includes a link to the full paper, repo or article.

## How it works

1. **Fetch:** Each source fetches more candidates than it returns.
2. **Exclusion filter:** Items whose title or description contains a word from `exclude_keywords` are dropped. Matching is case-insensitive.
3. **Semantic ranking** (arXiv and GitHub): Title and description are embedded with [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and compared to the profile via cosine similarity (shown as `Relevance score`).
4. **Top-k:** Only the best items are returned.

The pipeline adds three more steps:

5. **Duplicates:** The same paper can appear on both Hugging Face and arXiv. It is identified by its arXiv ID (without version number), otherwise by its URL, and kept in the Hugging Face section.
6. **Already sent:** Everything listed in [`state/sent_items.json`](state/sent_items.json) is filtered out. The file is only updated after a successful send. Entries older than 180 days are removed from it.
7. **Optional summary (Google Gemini):** A single Gemini call writes the opening of the newsletter: a headline ("This week's highlight: …"), 5–6 sentences on the most exciting findings, a one- to two-sentence takeaway for each of the 3–5 most exciting items, and a short "why it matters" for every item. The same call creates a **glossary** at the very top of the newsletter: the 10–20 hardest terms of the issue in the form `english term = German term: short German explanation`. Blog teasers from RSS feeds belong to their publishers, so the same call rewrites them in its own words (rewrites that copy 8 or more consecutive words are dropped). Whenever Gemini text is included, the footer says so (EU AI Act, Art. 50). Without `GEMINI_API_KEY`, or if the call fails, the newsletter goes out without this part.

The MCP tools do not use the state file. They always show the current state.

## Project structure

```
paper_porn/
├── config.toml              # profile, filters, sources, newsletter settings
├── .env.example             # template for credentials (locally as .env)
├── paperporn_mcp.py         # MCP server: thin wrapper that formats results as text
├── runner.py                # newsletter pipeline
├── core/                    # fetching and ranking, no MCP dependency
│   ├── config.py            # loads config.toml
│   ├── models.py            # Item, SourceError
│   ├── ranking.py           # embedding model, exclusion filter
│   └── sources.py           # the four sources
├── pipeline/                # newsletter only
│   ├── dedupe.py
│   ├── state.py
│   ├── llm.py
│   ├── render.py
│   ├── mailer.py            # delivery via SMTP
│   ├── brevo_mailer.py      # delivery as a Brevo campaign
│   └── templates/           # Jinja2 templates (HTML and plain text)
├── state/sent_items.json    # items already sent (committed by Actions)
├── tests/
└── .github/workflows/newsletter.yml
```

## Installation

Requires Python 3.11 or newer (developed with 3.14).

```bash
git clone <repo-url> paper_porn
cd paper_porn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # includes requirements.txt and pytest
```

On first start, `sentence-transformers` downloads the embedding model (about 90 MB).

## Configuration

### `config.toml`

All content settings live in [`config.toml`](config.toml). The file is part of the repo, so it must **not contain credentials or recipient addresses**.

| Section | Key | Meaning |
|---|---|---|
| `[profile]` | `description` | Your interests as free text, the basis for ranking |
| | `exclude_keywords` | Terms that exclude an item (applies to all sources) |
| `[sources]` | `arxiv_categories` | arXiv categories, e.g. `cs.LG` |
| | `github_topics` | GitHub topics, each queried separately |
| | `blog_feeds` | RSS feed URLs |
| `[newsletter]` | `delivery` | `"smtp"` or `"brevo"`, see [Delivery via Brevo](#delivery-via-brevo) |
| | `imprint_url` | Link to your legal notice, shown in the footer of every issue (empty: no link) |
| | `window_days` | Time window in days (default: 7) |
| | `subject` | Subject line |
| | `sender` | SMTP only: sender address, empty means `SMTP_USER` |
| | `state_file`, `state_retention_days` | State file and how long entries stay in it |
| `[newsletter.top_k]` | `arxiv`, `huggingface`, `github`, `blogs` | Number of items per section |
| `[newsletter.arxiv]` | `max_fetch` | Upper limit for the weekly query (a week has roughly 2,000–2,500 papers) |
| `[newsletter.github]` | `min_stars`, `max_fetch` | Minimum stars, candidates per topic |
| `[llm]` | `enabled`, `model` | Gemini summary and glossary on/off, Gemini model (default `gemini-3.8-flash`) |
| | `glossary_known_terms` | Terms you already know; they never appear in the glossary. Extend it to fine-tune the glossary |
| `[brevo]` | `list_ids` | Brevo only: IDs of the contact lists that receive the newsletter |
| | `sender_email`, `sender_name` | Brevo only: sender, must be an active sender in Brevo |

**Glossary language:** The glossary translates and explains terms in German. To change that, edit the last sentences of `GLOSSARY_INSTRUCTIONS` in [`pipeline/llm.py`](pipeline/llm.py). Terms the model returns that do not appear in the newsletter text are dropped and logged, so the glossary never explains words you won't read.

Exclusion keywords match substrings, not whole words. A short keyword like `"rl"` would also exclude "world".

### Environment variables

Credentials and recipients come exclusively from environment variables. Locally they live in `.env`, which is not committed:

```bash
cp .env.example .env
```

| Variable | Required | Meaning |
|---|---|---|
| `BREVO_API_KEY` | for `delivery = "brevo"` | Brevo API key (**SMTP & API → API Keys**) |
| `SMTP_HOST`, `SMTP_PORT` | for `delivery = "smtp"` | SMTP server. Port 465 uses SSL, any other port uses STARTTLS (e.g. 587) |
| `SMTP_USER`, `SMTP_PASSWORD` | for `delivery = "smtp"` | Login (for Gmail, an [app password](https://myaccount.google.com/apppasswords)) |
| `NEWSLETTER_RECIPIENTS` | for `delivery = "smtp"` | Recipients, comma-separated. They receive the mail as Bcc and cannot see each other |
| `NEWSLETTER_SENDER` | no | Overrides `sender` from `config.toml` |
| `GEMINI_API_KEY` | no | Enables the Gemini summary and "why it matters" texts ([get a key](https://aistudio.google.com/apikey)) |
| `GITHUB_TOKEN` | no | Higher rate limit for GitHub search (without a token: 10 requests per minute) |

## Testing the newsletter locally

There are two kinds of test run. Both fetch all sources live and take about a minute, most of it spent on the arXiv query. **Neither changes the state file**, so the next real newsletter still includes the tested items.

| | Dry run | Test send |
|---|---|---|
| Command | `python runner.py --dry-run` | `python runner.py --test-to <address>` |
| Sends an email | no | yes, only to `<address>` |
| Needs SMTP settings in `.env` | no | yes |
| Result | Files in `out/` | Email with `[TEST]` in the subject |
| Checks | Content and layout | Also SMTP access, sender and rendering in your mail client |

### 1. Dry run: render the newsletter to a file

```bash
source .venv/bin/activate
python runner.py --dry-run
```

1. The log shows the number of candidates per source and, at the end, the path of the generated files.
2. Open the HTML file in your browser, e.g. on macOS with `open out/newsletter-<date>.html`.
3. `out/newsletter-<date>.txt` contains the plain-text version shown by mail clients without HTML.

If a source fails, it appears in the log as `ERROR` and in the file under "Not available in this issue".

### 2. Test send to a real address

1. Create the `.env` if you haven't yet (`cp .env.example .env`). Fill in at least `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER` and `SMTP_PASSWORD`.
2. Send the newsletter to a test address:

   ```bash
   python runner.py --test-to your-test-address@example.com
   ```

   The mail goes **only** to this address; `NEWSLETTER_RECIPIENTS` is ignored.
3. The log should end with `Sent newsletter with N items to 1 recipient(s).` followed by `Test send: state file unchanged.`
4. The mail arrives with the subject `[TEST] paper_porn Weekly: …`. If you can't find it, check your spam folder.

Common problems with the test send:

- **Exit code 4, "Missing environment variables" / "Missing Brevo settings":** A variable is missing from `.env` (SMTP: `SMTP_*`, Brevo: `BREVO_API_KEY`) or `[brevo]` in `config.toml` is incomplete.
- **Brevo: `400` on `sendTest` with `unexistingEmails` or `withoutListEmails`:** Brevo only sends test mails to addresses that exist as contacts in your account. Add the address as a contact first.
- **`SMTPAuthenticationError`:** Username or password is wrong. Gmail and many other providers require an app password instead of your regular password.
- **Timeout or `ConnectionRefusedError`:** Host or port is wrong. Common ports are 587 (STARTTLS) and 465 (SSL).
- **No summary at the top of the mail:** `GEMINI_API_KEY` is not set, or the Gemini call failed (the log says which). The newsletter still works, just without the summary and "why it matters" texts.

### More options

```bash
python runner.py --dry-run --no-llm          # without the Gemini call
python runner.py --dry-run --output-dir /tmp # different output folder
python runner.py                             # real send to NEWSLETTER_RECIPIENTS, updates state/sent_items.json
```

Only run a real send locally if you actually mean it. It marks all included items as sent, and the automatic Monday newsletter will then skip them.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | Success, or there were no new items |
| 2 | All sources failed, nothing was sent |
| 3 | Sending failed, state unchanged |
| 4 | SMTP/Brevo settings or recipients missing |

If only some sources fail, the newsletter goes out with the remaining ones and lists the missing sources at the end of the mail.

## Delivery via Brevo

With `delivery = "brevo"`, the newsletter is not sent by this code but created as a campaign in your [Brevo](https://www.brevo.com) account and sent by Brevo to your contact list. Brevo handles sign-up forms, double opt-in, unsubscribes, bounces and statistics. The code lives in [`pipeline/brevo_mailer.py`](pipeline/brevo_mailer.py); [`pipeline/mailer.py`](pipeline/mailer.py) stays available for plain SMTP.

What a run does:

1. Create a draft campaign with the rendered HTML for the lists in `[brevo] list_ids` (`POST /v3/emailCampaigns`).
2. Send it right away (`POST /v3/emailCampaigns/{id}/sendNow`).
3. Update the state file, as with SMTP.

`--test-to <address>` creates a temporary campaign, sends it with Brevo's test function (`sendTest`) and deletes the draft again. `--dry-run` works as before; the unsubscribe link in the file is then the literal placeholder `{{ unsubscribe }}`.


## Automatic sending with GitHub Actions

The workflow [`.github/workflows/newsletter.yml`](.github/workflows/newsletter.yml) runs every Monday at 9:30 Berlin time.

- **Daylight saving time:** GitHub cron runs in UTC, so two times are scheduled (7:30 and 8:30 UTC). A preceding job checks Berlin's current UTC offset and only lets the matching run continue. This also works when GitHub starts the run late.
- **Not on the full hour:** GitHub does not guarantee scheduled runs. At `:00` the load is highest and runs are often delayed or dropped, so the schedule uses `:30`. If a Monday run is missing, start it manually via **Run workflow**.
- **Caching:** pip packages and the embedding model are cached between runs. torch is installed as the CPU-only build.
- **State file:** After a successful send, `state/sent_items.json` is committed back by `github-actions[bot]`.

### Setup

1. Under **Settings → Secrets and variables → Actions → New repository secret**, create these secrets:

   | Secret | Required |
   |---|---|
   | `BREVO_API_KEY` | with `delivery = "brevo"` |
   | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` | with `delivery = "smtp"` |
   | `NEWSLETTER_RECIPIENTS` | with `delivery = "smtp"` |
   | `NEWSLETTER_SENDER` | no |
   | `GEMINI_API_KEY` | no |

   GitHub provides `GITHUB_TOKEN` automatically.

2. Under **Settings → Actions → General → Workflow permissions**, **Read and write permissions** must be allowed so the workflow can commit the state file. The workflow requests `contents: write` itself, but this setting can block it at the organization level.

3. Test run without sending: **Actions → Weekly newsletter → Run workflow**, then tick "Dry run: render HTML only (as artifact), send nothing". The preview is attached to the run as the artifact `newsletter-preview`.

4. Without the checkbox, the mail is actually sent.

Note: GitHub disables scheduled workflows in repositories with no activity for 60 days. The weekly state commits count as activity, but weeks without new items produce no commit.

## Connecting the MCP server to Claude Desktop

First run the server once manually so the embedding model gets downloaded:

```bash
.venv/bin/python paperporn_mcp.py
```

Once `Loading semantic embedding model...` appears with no error after it, the server is running and waiting for input on stdin. Stop it with `Ctrl+C`.

1. In Claude Desktop, open **Settings → Developer → Edit Config**. Alternatively, open the file directly:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the server under `mcpServers`. Both paths must be **absolute** and point to the Python inside the virtual environment:

   ```json
   {
     "mcpServers": {
       "paperporn": {
         "command": "/ABSOLUTE/PATH/TO/paper_porn/.venv/bin/python",
         "args": ["/ABSOLUTE/PATH/TO/paper_porn/paperporn_mcp.py"],
         "env": {
           "GITHUB_TOKEN": "optional_your_token"
         }
       }
     }
   }
   ```

   Get the absolute path by running `pwd` in the project directory. On Windows the Python path is `...\\.venv\\Scripts\\python.exe`. You can omit the `env` block if you don't use a token.

3. Quit Claude Desktop completely and restart it.

4. The four tools should now appear under the tools icon in the chat input. A good test prompt:

   > Using the paperporn tools, give me a short overview: the 3 most relevant arXiv papers, the top 3 Hugging Face papers, 3 new GitHub repos and the latest blog posts.

### Troubleshooting

- **Server doesn't show up or shows an error:** Logs are in `~/Library/Logs/Claude/mcp-server-paperporn.log` on macOS and in `%APPDATA%\Claude\logs\` on Windows.
- **Timeout on first start:** Usually the embedding model is still downloading. Run the server once manually, then restart Claude Desktop.
- **`ModuleNotFoundError`:** `command` probably points to the system Python instead of `.venv/bin/python`.
- **GitHub error 403:** The rate limit was hit. Wait a minute or set `GITHUB_TOKEN`.
- **`SSLError` / `CERTIFICATE_VERIFY_FAILED`:** All sources are fetched with `requests`, which uses the certificate bundle from `certifi`, independent of how Python was installed. Behind a corporate proxy that inspects TLS traffic, point it to your company's CA certificate: `export REQUESTS_CA_BUNDLE=/path/to/company-ca.pem`. For the newsletter pipeline you can also put it in `.env`; for the MCP server, add it to the `env` block in `claude_desktop_config.json`.

## Tests

```bash
python -m pytest
```

The tests need no network access. They cover deduplication, the state filter, template rendering, the runner's error handling (one or all sources failing, sending failing), the LLM step with a simulated client, and the output format of the MCP tools.

## License

[MIT](LICENSE)
