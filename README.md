# paper_porn

Ein MCP-Server, der täglich neue Inhalte aus der KI- und ML-Welt sammelt und nach einem persönlichen Interessenprofil filtert. Er liefert ausgewählte arXiv-Paper, Hugging Face Daily Papers, neue GitHub-Repositories und Engineering-Blogbeiträge. Claude Desktop (oder ein anderer MCP-Client) kann daraus z. B. einen täglichen Newsletter zusammenstellen.

Damit nur relevante Inhalte beim Modell ankommen und Tokens gespart werden, wird bereits im Server gefiltert und sortiert.

## Tools

| Tool | Quelle | Auswahl |
|---|---|---|
| `fetch_filtered_arxiv_papers` | arXiv API | Neueste Paper der konfigurierten Kategorien, nach semantischer Ähnlichkeit zum Profil sortiert |
| `fetch_huggingface_daily_papers` | Hugging Face Daily Papers | Nach Community-Upvotes sortiert |
| `fetch_github_trending_ml_repos` | GitHub Search API | Repos der letzten `days` Tage mit mindestens `min_stars` Sternen, nach semantischer Ähnlichkeit sortiert |
| `fetch_ai_engineering_blogs` | RSS-Feeds (PyTorch, Hugging Face) | Neueste Beiträge, abwechselnd aus allen Feeds |

Jeder Eintrag enthält einen Link zum vollständigen Paper, Repo oder Artikel.

## Wie es funktioniert

1. **Abrufen:** Jedes Tool holt mehr Kandidaten als es zurückgibt, z. B. `max_fetch=30` arXiv-Paper.
2. **Ausschlussfilter:** Einträge, deren Titel oder Beschreibung ein Wort aus `EXCLUDE_KEYWORDS` enthält, werden verworfen. Groß- und Kleinschreibung spielt dabei keine Rolle.
3. **Semantisches Ranking** (arXiv und GitHub): Titel und Beschreibung werden mit dem Modell [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) in Embeddings umgewandelt und per Kosinus-Ähnlichkeit mit `STUDENT_PROFILE` verglichen. Der Wert erscheint im Output als `Relevanz-Score`.
4. **Top-k:** Nur die besten `top_k` Einträge werden als kompakter Text zurückgegeben.

## Konfiguration

Alle inhaltlichen Einstellungen stehen in [`config_module.py`](config_module.py):

| Variable | Bedeutung |
|---|---|
| `STUDENT_PROFILE` | Freitext-Beschreibung deiner Interessen, Grundlage für das semantische Ranking |
| `EXCLUDE_KEYWORDS` | Begriffe, die zum Ausschluss eines Eintrags führen (gilt für alle Quellen) |
| `PORN_CATEGORIES` | arXiv-Kategorien, z. B. `cs.LG`, `cs.AI`, `cs.DC` |
| `GITHUB_TOPICS` | GitHub-Topics, nach denen gesucht wird (jedes Topic wird einzeln abgefragt) |
| `BLOG_FEEDS` | RSS-Feed-URLs der Blogs |

Optional kannst du die Umgebungsvariable `GITHUB_TOKEN` (ein GitHub Personal Access Token ohne besondere Rechte) setzen. Ohne Token erlaubt die GitHub-Suche nur 10 Anfragen pro Minute.

Hinweis: Die Ausschlusswörter werden als Textteile gesucht, nicht als ganze Wörter. Ein kurzes Keyword wie `"rl"` würde also auch „world“ ausschließen.

## Installation

Voraussetzung ist Python 3.10 oder neuer (entwickelt mit 3.14).

```bash
git clone <repo-url> paper_porn
cd paper_porn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Beim ersten Start lädt `sentence-transformers` das Embedding-Modell herunter (ca. 90 MB). Starte den Server deshalb einmal manuell, bevor du ihn mit Claude Desktop verbindest:

```bash
.venv/bin/python paperporn_mcp.py
```

Sobald `Lade Semantic-Embedding-Modell...` erscheint und keine Fehlermeldung folgt, läuft der Server und wartet auf Eingaben über stdin. Beenden mit `Ctrl+C`.

## Verbindung mit Claude Desktop

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

   > Stell mir mit den paperporn-Tools einen kurzen Newsletter zusammen: die 3 relevantesten arXiv-Paper, die Top-3 Hugging Face Papers, 3 neue GitHub-Repos und die neuesten Blogbeiträge.

### Fehlersuche

- **Server erscheint nicht oder zeigt einen Fehler:** Die Logs findest du unter macOS in `~/Library/Logs/Claude/mcp-server-paperporn.log` und unter Windows in `%APPDATA%\Claude\logs\`.
- **Timeout beim ersten Start:** Meist lädt das Embedding-Modell noch. Starte den Server einmal manuell (siehe Installation) und dann Claude Desktop neu.
- **`ModuleNotFoundError`:** `command` zeigt wahrscheinlich auf das System-Python statt auf `.venv/bin/python`.
- **GitHub-Fehler 403:** Das Rate-Limit ist erreicht. Warte eine Minute oder setze `GITHUB_TOKEN`.

## Projektstruktur

```
paper_porn/
├── paperporn_mcp.py    # MCP-Server mit allen Tools
├── config_module.py    # Profil, Filter, Quellen
├── requirements.txt    # Abhängigkeiten
└── README.md
```

## Ausblick

- Automatischer täglicher Newsletter
- Speicher für bereits gesehene Einträge, damit nichts doppelt angezeigt wird
