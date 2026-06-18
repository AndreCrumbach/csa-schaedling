import anthropic
import base64
import httpx
import json
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Schädlingsbestimmung")

# ── Foto-Zähler ───────────────────────────────────────────────────────────────
# AppData/Local ist immer beschreibbar, auch wenn die App in Programme/ installiert ist
_APP_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "CSA-Schaedlingsbekaempfung"
_APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
_COUNTER_FILE = _APP_DATA_DIR / "foto_counter.txt"
_counter_lock = threading.Lock()
_STATS_KEY = "CSA2024"

def _get_count() -> int:
    try:
        return int(_COUNTER_FILE.read_text().strip())
    except Exception:
        return 0

def _increment_count() -> int:
    with _counter_lock:
        count = _get_count() + 1
        _COUNTER_FILE.write_text(str(count))
        return count
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY nicht gesetzt. Bitte in der .env-Datei oder als Umgebungsvariable hinterlegen.",
        )
    # SSL-Zertifikatsprüfung deaktivieren (Workaround für Netzwerke mit
    # blockierter Zertifikatssperrliste)
    http_client = httpx.Client(verify=False)
    return anthropic.Anthropic(api_key=api_key, http_client=http_client)


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/count")
async def count():
    return JSONResponse(content={"count": _get_count() + 1000})


@app.get("/stats")
async def stats(key: str = Query(default="")):
    if key != _STATS_KEY:
        raise HTTPException(status_code=403, detail="Zugriff verweigert.")
    count = _get_count()
    return JSONResponse(content={
        "analysierte_fotos": count,
        "analysierte_fotos_angezeigt": count + 1000,
        "hinweis": "Gesamtanzahl aller Analysen seit Inbetriebnahme (angezeigte Zahl = +1000)"
    })


@app.post("/analyze")
async def analyze_pest(file: UploadFile = File(...)):
    # Validate file type
    content_type = file.content_type or "image/jpeg"
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Nicht unterstütztes Dateiformat: {content_type}. Erlaubt sind JPEG, PNG, GIF, WebP.",
        )

    # Read and validate file size
    image_data = await file.read()
    if len(image_data) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Datei zu groß. Maximale Größe: 20 MB.",
        )

    base64_image = base64.standard_b64encode(image_data).decode("utf-8")
    client = get_client()

    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": content_type,
            "data": base64_image,
        },
    }

    # ── Schritt 1: Diagnosefakten feststellen ─────────────────────────────────
    diag_response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=80,
        messages=[{
            "role": "user",
            "content": [
                image_block,
                {"type": "text", "text": (
                    "Beantworte nur diese drei Fragen zu den Tieren im Bild. "
                    "Antworte exakt in diesem Format, keine weiteren Worte: A=JA|B=3|C=12\n"
                    "A: Sind fadenförmige Antennen EINDEUTIG sichtbar, die mindestens so lang sind wie der GESAMTE Körper? "
                    "Antworte NUR JA wenn du sie wirklich klar erkennst – im Zweifelsfall NEIN.\n"
                    "B: Geschätzte Körperlänge in mm – eher konservativ (zu klein) als zu groß schätzen (nur eine Zahl)\n"
                    "C: Anzahl sichtbarer Tiere (nur eine Zahl)"
                )},
            ],
        }],
    )

    diag_text = next((b.text for b in diag_response.content if b.type == "text"), "")

    antennen = "NEIN"
    groesse_mm = 0.0
    anzahl = 1
    for part in diag_text.replace("\n", "|").split("|"):
        part = part.strip()
        if part.upper().startswith("A="):
            antennen = "JA" if "JA" in part.upper() else "NEIN"
        elif part.upper().startswith("B="):
            try:
                groesse_mm = float(re.sub(r"[^\d.]", "", part[2:]))
            except ValueError:
                pass
        elif part.upper().startswith("C="):
            try:
                anzahl = int(re.sub(r"[^\d]", "", part[2:]))
            except ValueError:
                pass

    schabe_ausgeschlossen = antennen == "NEIN" or groesse_mm < 10 or anzahl > 6
    floh_wahrscheinlich   = groesse_mm <= 3 and anzahl >= 5

    fakten = (
        f"VORAB FESTGESTELLTE FAKTEN (durch separate Bildanalyse, nicht anzuzweifeln):\n"
        f"- Antennen sichtbar: {antennen}\n"
        f"- Geschätzte Körpergröße: {groesse_mm} mm\n"
        f"- Anzahl Tiere im Bild: ca. {anzahl}\n"
    )
    if schabe_ausgeschlossen:
        fakten += "- REGEL: Schabe/Kakerlake ist AUSGESCHLOSSEN (Antennen nicht sichtbar und/oder Größe < 6 mm).\n"
    if floh_wahrscheinlich:
        fakten += "- REGEL: Sehr wahrscheinlich Flöhe (Körper ≤ 3 mm, mehrere Tiere sichtbar).\n"

    # ── Schritt 2: Vollständige Bestimmung ────────────────────────────────────
    prompt = fakten + """
Bestimme nun den Hausschädling. Die obigen Fakten sind bindend – leite deine Bestimmung zwingend daraus ab.

Typische Referenzwerte:
- Schaben/Kakerlaken: 10–25 mm, lange Antennen (≥ Körperlänge), 6 lange abstehende Beine
- Flöhe: 1–3 mm, dunkelbraun, kompakt-rundlich, keine sichtbaren Antennen
- Bettwanzen: 4–7 mm, oval abgeplattet, rotbraun
- Teppichkäfer: 2–5 mm, rundlich, oft gemustert
- Silberfischchen: 10–20 mm, silbrig, 3 Schwanzborsten

Antworte ausschließlich auf Deutsch im folgenden JSON-Format (kein Markdown, kein Codeblock – nur reines JSON):
{
  "ist_schädling": true,
  "ist_nützling": false,
  "konfidenz": 85,
  "merkmale": "Kurze Auflistung der sichtbaren Bestimmungsmerkmale",
  "name": "Deutscher Name",
  "wissenschaftlicher_name": "Lateinischer Name",
  "gefährlichkeit": "gering",
  "gefährlichkeit_beschreibung": "Kurze Erklärung der Gefährlichkeit für Mensch, Tier oder Gebäude",
  "nutzen": "",
  "verbreitung": "Wo das Tier typischerweise vorkommt",
  "beschreibung": "Aussehen, Größe und wichtigste Erkennungsmerkmale",
  "bekämpfung": ["Methode 1", "Methode 2", "Methode 3"],
  "präventionstipps": ["Tipp 1", "Tipp 2"],
  "hinweis": ""
}

Werte für "gefährlichkeit": "gering", "mittel" oder "hoch".
"konfidenz": Ganzzahl 0–100 – bitte konservativ einschätzen: 80–100 nur wenn das Bild sehr klar und alle Merkmale eindeutig erkennbar sind, 60–79 wenn die Art wahrscheinlich zutrifft aber nicht alle Merkmale sicher zu erkennen sind, unter 60 bei unsicherer oder unklarer Bestimmung.

Nützlinge (ist_nützling=true, ist_schädling=false): Spinnen, Marienkäfer, Florfliegen, Schlupfwespen, Laufkäfer, Ohrwürmer, Tausendfüßer u.ä.
Bei Nützlingen: "nutzen" mit konkretem Nutzen füllen (z.B. welche Schädlinge sie fressen), bekämpfung=[], präventionstipps=[].

Falls kein Tier erkennbar: ist_schädling=false, ist_nützling=false, alle Felder leer, hinweis erklären."""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [image_block, {"type": "text", "text": prompt}],
            }
        ],
    )

    result_text = next(
        (block.text for block in response.content if block.type == "text"), ""
    )

    # Parse JSON – strip optional code fence if present
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", result_text.strip(), flags=re.MULTILINE)
    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "ist_schädling": False,
                "hinweis": "Antwort konnte nicht verarbeitet werden.",
                "name": "",
                "wissenschaftlicher_name": "",
                "gefährlichkeit": "gering",
                "gefährlichkeit_beschreibung": "",
                "verbreitung": "",
                "beschreibung": "",
                "bekämpfung": [],
                "präventionstipps": [],
            }

    # Sicherstellen dass konfidenz immer vorhanden ist
    result.setdefault("konfidenz", 40)

    # Schaben-Erkennung immer mit Warnkarte – KI kann Schaben von ähnlichen
    # Kleinstinsekten (Flöhe, Nymphen) ohne Größenreferenz nicht zuverlässig trennen.
    name_lower = result.get("name", "").lower()
    ist_schabe = any(w in name_lower for w in ["schab", "kakerlak"])
    if ist_schabe and result.get("konfidenz", 0) <= 75:
        result["konfidenz"] = 45

    _increment_count()
    return JSONResponse(content=result)
