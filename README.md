# paper_porn

Sammelt neue Inhalte aus der KI- und ML-Welt und filtert sie nach einem persönlichen Interessenprofil: arXiv-Paper, Hugging Face Daily Papers, neue GitHub-Repositories und Engineering-Blogbeiträge.

Das Projekt hat zwei Einstiegspunkte, die auf derselben Logik aufbauen:

- **MCP-Server** (`paperporn_mcp.py`): Claude Desktop oder ein anderer MCP-Client ruft die Quellen bei Bedarf als Tools auf.
- **Newsletter-Pipeline** (`runner.py`): Sammelt einmal pro Woche alle Quellen über 7 Tage, entfernt Dubletten und bereits verschickte Einträge und verschickt eine HTML-Mail. Läuft automatisch über GitHub Actions, jeden Montag um 7:00 Uhr (Europe/Berlin).

## Quellen

| Quelle | Auswahl | MCP-Tool |
|---|---|---|
| arXiv | Neueste Paper der konfigurierten Kategorien, nach semantischer Ähnlichkeit zum Profil sortiert | `fetch_filtered_arxiv_papers` |
| Hugging Face Daily Papers | Nach Community-Upvotes sortiert | `fetch_huggingface_daily_papers` |
| GitHub | Neue Repos mit mindestens `min_stars` Sternen, nach semantischer Ähnlichkeit sortiert | `fetch_github_trending_ml_repos` |
| Blogs (PyTorch, Hugging Face) | Neueste Beiträge, abwechselnd aus allen Feeds | `fetch_ai_engineering_blogs` |

Jeder Eintrag enthält einen Link zum vollständigen Paper, Repo oder Artikel.

## Wie es funktioniert

1. **Abrufen:** Jede Quelle holt mehr Kandidaten als sie zurückgibt.
2. **Ausschlussfilter:** Einträge, deren Titel oder Beschreibung ein Wort aus `exclude_keywords` enthält, werden verworfen. Groß- und Kleinschreibung spielt dabei keine Rolle.
3. **Semantisches Ranking** (arXiv und GitHub): Titel und Beschreibung werden mit [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) eingebettet und per Kosinus-Ähnlichkeit mit dem Profil verglichen (`Relevanz-Score`).
4. **Top-k:** Nur die besten Einträge werden zurückgegeben.

Die Pipeline ergänzt drei weitere Schritte:

5. **Dubletten:** Dasselbe Paper kann bei Hugging Face und arXiv auftauchen. Erkannt wird es an der arXiv-ID (ohne Versionsnummer), sonst an der URL. Es bleibt im Hugging-Face-Abschnitt.
6. **Bereits verschickt:** Alles, was in [`state/sent_items.json`](state/sent_items.json) steht, wird ausgefiltert. Die Datei wird erst nach erfolgreichem Versand aktualisiert. Einträge, die älter als 180 Tage sind, werden daraus gelöscht.
7. **Optionales Editorial:** Ein einzelner Aufruf der Claude API schreibt eine kurze Einordnung der Woche und pro Eintrag ein bis zwei Sätze, warum er relevant ist. Ohne `ANTHROPIC_API_KEY`, oder wenn der Aufruf fehlschlägt, geht der Newsletter ohne diesen Teil raus.

Die MCP-Tools nutzen die State-Datei nicht. Sie zeigen immer den aktuellen Stand.

## Projektstruktur

```
paper_porn/
├── config.toml              # Profil, Filter, Quellen, Newsletter-Einstellungen
├── .env.example             # Vorlage für Zugangsdaten (lokal als .env)
├── paperporn_mcp.py         # MCP-Server: dünner Wrapper, formatiert Ergebnisse als Text
├── runner.py                # Newsletter-Pipeline
├── core/                    # Abruf und Ranking, ohne MCP-Abhängigkeit
│   ├── config.py            # lädt config.toml
│   ├── models.py            # Item, SourceError
│   ├── ranking.py           # Embedding-Modell, Ausschlussfilter
│   └── sources.py           # die vier Quellen
├── pipeline/                # nur für den Newsletter
│   ├── dedupe.py
│   ├── state.py
│   ├── llm.py
│   ├── render.py
│   ├── mailer.py
│   └── templates/           # Jinja2-Templates (HTML und Text)
├── state/sent_items.json    # bereits verschickte Einträge (wird von Actions committet)
├── tests/
└── .github/workflows/newsletter.yml
```

## Installation

Voraussetzung ist Python 3.11 oder neuer (entwickelt mit 3.14).

```bash
git clone <repo-url> paper_porn
cd paper_porn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # enthält requirements.txt und pytest
```

Beim ersten Start lädt `sentence-transformers` das Embedding-Modell herunter (ca. 90 MB).

## Konfiguration

### `config.toml`

Alle inhaltlichen Einstellungen stehen in [`config.toml`](config.toml). Die Datei liegt im Repo, deshalb gehören **keine Zugangsdaten und keine Mailadressen** hinein.

| Abschnitt | Schlüssel | Bedeutung |
|---|---|---|
| `[profile]` | `description` | Deine Interessen als Freitext, Grundlage für das Ranking |
| | `exclude_keywords` | Begriffe, die einen Eintrag ausschließen (gilt für alle Quellen) |
| `[sources]` | `arxiv_categories` | arXiv-Kategorien, z. B. `cs.LG` |
| | `github_topics` | GitHub-Topics, jedes wird einzeln abgefragt |
| | `blog_feeds` | RSS-Feed-URLs |
| `[newsletter]` | `window_days` | Zeitfenster in Tagen (Standard: 7) |
| | `subject` | Betreffzeile |
| | `sender` | Absenderadresse, leer bedeutet `SMTP_USER` |
| | `state_file`, `state_retention_days` | State-Datei und wie lange Einträge darin bleiben |
| `[newsletter.top_k]` | `arxiv`, `huggingface`, `github`, `blogs` | Anzahl Einträge pro Abschnitt |
| `[newsletter.arxiv]` | `max_fetch` | Obergrenze für die Wochenabfrage (eine Woche hat etwa 2.000–2.500 Paper) |
| `[newsletter.github]` | `min_stars`, `max_fetch` | Mindestanzahl Sterne, Kandidaten pro Topic |
| `[llm]` | `enabled`, `model`, `effort` | Editorial an/aus, Claude-Modell, Aufwand (`low` bis `max`) |

Die Ausschlusswörter werden als Textteile gesucht, nicht als ganze Wörter. Ein kurzes Keyword wie `"rl"` würde also auch „world“ ausschließen.

### Umgebungsvariablen

Zugangsdaten und Empfänger kommen ausschließlich aus Umgebungsvariablen. Lokal stehen sie in `.env`, die Datei wird nicht committet:

```bash
cp .env.example .env
```

| Variable | Pflicht | Bedeutung |
|---|---|---|
| `SMTP_HOST`, `SMTP_PORT` | für den Versand | SMTP-Server. Port 465 nutzt SSL, jeder andere Port STARTTLS (z. B. 587) |
| `SMTP_USER`, `SMTP_PASSWORD` | für den Versand | Login (bei Gmail ein [App-Passwort](https://myaccount.google.com/apppasswords)) |
| `NEWSLETTER_RECIPIENTS` | für den Versand | Empfänger, kommagetrennt. Sie bekommen die Mail als Bcc und sehen sich gegenseitig nicht |
| `NEWSLETTER_SENDER` | nein | Überschreibt `sender` aus `config.toml` |
| `ANTHROPIC_API_KEY` | nein | Aktiviert Editorial und „Warum relevant“-Texte |
| `GITHUB_TOKEN` | nein | Höheres Rate-Limit für die GitHub-Suche (ohne Token: 10 Anfragen pro Minute) |

## Newsletter lokal testen

Es gibt zwei Arten von Testlauf. Beide rufen alle Quellen live ab und dauern etwa eine Minute, den größten Teil davon braucht die arXiv-Abfrage. Bei **beiden bleibt die State-Datei unverändert**, der nächste echte Newsletter enthält die getesteten Einträge also trotzdem.

| | Dry-Run | Testversand |
|---|---|---|
| Befehl | `python runner.py --dry-run` | `python runner.py --test-to <adresse>` |
| Verschickt eine Mail | nein | ja, nur an `<adresse>` |
| Braucht SMTP-Daten in `.env` | nein | ja |
| Ergebnis | Dateien in `out/` | Mail mit `[TEST]` im Betreff |
| Prüft | Inhalt und Layout | zusätzlich SMTP-Zugang, Absender und Darstellung im Mailprogramm |

### 1. Dry-Run: Newsletter als Datei erzeugen

```bash
source .venv/bin/activate
python runner.py --dry-run
```

1. Im Log siehst du pro Quelle die Zahl der Kandidaten und am Ende den Pfad der erzeugten Dateien.
2. Öffne die HTML-Datei im Browser, z. B. auf macOS mit `open out/newsletter-<datum>.html`.
3. `out/newsletter-<datum>.txt` enthält die Textversion, die Mailprogramme ohne HTML anzeigen.

Fällt eine Quelle aus, steht sie im Log als `ERROR` und in der Datei unter „Nicht verfügbar in dieser Ausgabe“.

### 2. Testversand an eine echte Adresse

1. Lege die `.env` an, falls noch nicht geschehen (`cp .env.example .env`). Trage mindestens `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER` und `SMTP_PASSWORD` ein.
2. Schicke den Newsletter an eine Testadresse:

   ```bash
   python runner.py --test-to deine-test-adresse@example.com
   ```

   Die Mail geht **nur** an diese Adresse, `NEWSLETTER_RECIPIENTS` wird ignoriert.
3. Im Log sollte am Ende `Newsletter mit N Einträgen an 1 Empfänger verschickt.` stehen, danach `Testversand: State-Datei bleibt unverändert.`
4. Im Postfach kommt die Mail mit dem Betreff `[TEST] paper_porn Weekly: …` an. Findest du sie nicht, schau in den Spam-Ordner.

Typische Fehler beim Testversand:

- **Exit-Code 4, „Fehlende Umgebungsvariablen“:** In der `.env` fehlt eine SMTP-Variable.
- **`SMTPAuthenticationError`:** Benutzername oder Passwort stimmen nicht. Bei Gmail und vielen anderen Anbietern brauchst du ein App-Passwort statt deines normalen Passworts.
- **Timeout oder `ConnectionRefusedError`:** Host oder Port sind falsch. Üblich sind 587 (STARTTLS) und 465 (SSL).
- **Kein Editorial in der Mail:** `ANTHROPIC_API_KEY` ist nicht gesetzt. Der Newsletter funktioniert trotzdem, nur ohne Einordnung und „Warum relevant“.

### Weitere Optionen

```bash
python runner.py --dry-run --no-llm          # ohne Claude-Aufruf
python runner.py --dry-run --output-dir /tmp # anderer Zielordner
python runner.py                             # echter Versand an NEWSLETTER_RECIPIENTS, aktualisiert state/sent_items.json
```

Starte den echten Versand lokal nur, wenn du das wirklich willst. Er markiert alle enthaltenen Einträge als verschickt, und der automatische Newsletter am Montag lässt sie dann weg.

**Exit-Codes:**

| Code | Bedeutung |
|---|---|
| 0 | Erfolgreich, oder es gab keine neuen Einträge |
| 2 | Alle Quellen ausgefallen, es wurde nichts verschickt |
| 3 | Versand fehlgeschlagen, State unverändert |
| 4 | SMTP-Daten oder Empfänger fehlen |

Fällt nur ein Teil der Quellen aus, geht der Newsletter mit den übrigen raus und nennt die fehlenden am Ende der Mail.

## Automatischer Versand mit GitHub Actions

Der Workflow [`.github/workflows/newsletter.yml`](.github/workflows/newsletter.yml) läuft jeden Montag um 7:00 Uhr Berliner Zeit.

- **Sommer- und Winterzeit:** GitHub-Cron rechnet in UTC. Deshalb sind zwei Zeitpunkte eingetragen (5:00 und 6:00 UTC). Ein vorgeschalteter Job prüft den aktuellen Berliner UTC-Offset und lässt nur den passenden Lauf weiterlaufen. Das funktioniert auch dann, wenn GitHub den Start verzögert.
- **Caching:** pip-Pakete und das Embedding-Modell werden zwischen den Läufen gecacht. torch wird als CPU-Variante installiert.
- **State-Datei:** Nach erfolgreichem Versand wird `state/sent_items.json` als `github-actions[bot]` zurückcommittet.

### Einrichtung

1. Unter **Settings → Secrets and variables → Actions → New repository secret** diese Secrets anlegen:

   | Secret | Pflicht |
   |---|---|
   | `SMTP_HOST` | ja |
   | `SMTP_PORT` | ja |
   | `SMTP_USER` | ja |
   | `SMTP_PASSWORD` | ja |
   | `NEWSLETTER_RECIPIENTS` | ja |
   | `NEWSLETTER_SENDER` | nein |
   | `ANTHROPIC_API_KEY` | nein |

   `GITHUB_TOKEN` stellt GitHub automatisch bereit.

2. Unter **Settings → Actions → General → Workflow permissions** muss **Read and write permissions** erlaubt sein, damit der Workflow die State-Datei committen darf. Der Workflow fordert `contents: write` zwar selbst an, diese Einstellung kann es aber auf Organisationsebene blockieren.

3. Testlauf ohne Versand: **Actions → Weekly newsletter → Run workflow**, Haken bei „Nur HTML erzeugen“ setzen. Die Vorschau liegt danach als Artifact `newsletter-preview` am Lauf.

4. Ohne den Haken wird die Mail echt verschickt.

Hinweis: GitHub deaktiviert geplante Workflows in Repositories, in denen 60 Tage lang nichts passiert. Die wöchentlichen State-Commits zählen als Aktivität. Wochen ohne neue Einträge erzeugen allerdings keinen Commit.

## MCP-Server mit Claude Desktop verbinden

Teste den Server zuerst einmal manuell, damit das Embedding-Modell heruntergeladen wird:

```bash
.venv/bin/python paperporn_mcp.py
```

Sobald `Lade Semantic-Embedding-Modell...` erscheint und keine Fehlermeldung folgt, läuft der Server und wartet auf Eingaben über stdin. Beenden mit `Ctrl+C`.

1. Öffne in Claude Desktop **Einstellungen → Entwickler → Konfiguration bearbeiten**. Alternativ kannst du die Datei direkt öffnen:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Trage den Server unter `mcpServers` ein. Beide Pfade müssen **absolut** sein und auf die Python-Version im virtuellen Environment zeigen:

   ```json
   {
     "mcpServers": {
       "paperporn": {
         "command": "/ABSOLUTER/PFAD/ZU/paper_porn/.venv/bin/python",
         "args": ["/ABSOLUTER/PFAD/ZU/paper_porn/paperporn_mcp.py"],
         "env": {
           "GITHUB_TOKEN": "optional_dein_token"
         }
       }
     }
   }
   ```

   Den absoluten Pfad bekommst du mit `pwd` im Projektverzeichnis. Unter Windows lautet der Python-Pfad `...\\.venv\\Scripts\\python.exe`. Den `env`-Block kannst du weglassen, wenn du keinen Token verwendest.

3. Beende Claude Desktop vollständig und starte es neu.

4. Im Chat sollten die vier Tools jetzt über das Tool-Symbol im Eingabefeld erscheinen. Zum Testen eignet sich zum Beispiel:

   > Stell mir mit den paperporn-Tools einen kurzen Überblick zusammen: die 3 relevantesten arXiv-Paper, die Top-3 Hugging Face Papers, 3 neue GitHub-Repos und die neuesten Blogbeiträge.

### Fehlersuche

- **Server erscheint nicht oder zeigt einen Fehler:** Die Logs findest du unter macOS in `~/Library/Logs/Claude/mcp-server-paperporn.log` und unter Windows in `%APPDATA%\Claude\logs\`.
- **Timeout beim ersten Start:** Meist lädt das Embedding-Modell noch. Starte den Server einmal manuell und dann Claude Desktop neu.
- **`ModuleNotFoundError`:** `command` zeigt wahrscheinlich auf das System-Python statt auf `.venv/bin/python`.
- **GitHub-Fehler 403:** Das Rate-Limit ist erreicht. Warte eine Minute oder setze `GITHUB_TOKEN`.

## Tests

```bash
python -m pytest
```

Die Tests brauchen keinen Netzwerkzugriff. Sie decken Deduplizierung, State-Filter, Template-Rendering, das Fehlerverhalten des Runners (eine oder alle Quellen fallen aus, Versand schlägt fehl), den LLM-Schritt mit einem simulierten Client und das Ausgabeformat der MCP-Tools ab.
