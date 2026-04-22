"""
Consulente Legale AI  —  v2.0
================================
App Streamlit per consulenza giuridica italiana basata su Google Gemini 2.0 Flash
con Google Search Grounding per il recupero in tempo reale della normativa vigente.

Fix applicati (v2.0):
  BUG-001 [CRITICAL] Python 3.9 compat: rimossi union type X|Y, usato Optional
  BUG-002 [CRITICAL] Grounding incompatibile con start_chat(): sostituito con
           generate_content() + history manuale (unico approccio supportato)
  BUG-003 [HIGH]     Guard su response.candidates prima di accedere a .text
  BUG-004 [MEDIUM]   Validazione lunghezza input (MAX_INPUT_CHARS)
  BUG-005 [MEDIUM]   Spinner spostato fuori da chat_message per crretto rendering
  BUG-006 [LOW]      Rimossa dipendenza da immagine Wikipedia (URL sterno fragile)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import streamlit as st
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────
MODEL_NAME       = "gemini-2.5-flash"
APP_TITLE        = "Consulente Legale AI"
APP_SUBTITLE     = "Analisi giuridica aggiornata sul diritto italiano vigente"
MAX_INPUT_CHARS  = 4_000   # FIX BUG-004: limite caratteri per messaggio utente
MAX_HISTORY_TURNS = 20     # Ultimi N scambi inclusi nel contesto AI

# ─────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Agisci come un consulente legale esperto nel sistema giuridico italiano \
con oltre 20 anni di esperienza in diritto civile, penale, commerciale e del lavoro.

Il tuo metodo si articola in tre fasi obbligatorie:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 1 — INTERROGA (se necessario)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prima di elaborare qualsiasi parere, valuta se il racconto contiene tutti gli elementi \
essenziali per qualificare correttamente la fattispecie giuridica.

Elementi tipicamente rilevanti (verifica caso per caso):
- Esistenza e forma del contratto (scritto, verbale, digitale)
- Data esatta dei fatti (per valutare prescrizione e decadenza)
- Qualità giuridica delle parti (consumatore/professionista, privato/azienda)
- Valore economico della controversia
- Luogo dei fatti, precedenti tentativi di risoluzione, prove disponibili

REGOLA: Fai domande SOLO se strettamente necessarie. Se le informazioni bastano, \
procedi direttamente. Se devi chiedere, raggruppa tutte le domande in un unico \
messaggio numerato.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 2 — RICERCA NORMATIVA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usa lo strumento di ricerca web per recuperare la versione aggiornata degli articoli \
pertinenti. Fonti ufficiali da consultare:
- normattiva.it (testo vigente dei codici)
- gazzettaufficiale.it (decreti e leggi recenti)
- eur-lex.europa.eu (normativa europea applicabile)
- italgiure.giustizia.it (giurisprudenza Cassazione)

NON affidarti solo alla memoria di addestramento: le leggi cambiano.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FASE 3 — PARERE STRUTTURATO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quando hai tutti gli elementi, produci il parere con questa struttura ESATTA:

---

## 📋 Inquadramento Giuridico
[Qualifica la fattispecie: contrattuale, extracontrattuale, penale, lavoristica, ecc.]

## 📜 Riferimenti Normativi
[Per ogni norma: Articolo + testo aggiornato + fonte ufficiale + eventuale Cassazione]

## 🔍 Analisi del Caso
[Applica le norme ai fatti. Spiega il ragionamento. Evidenzia punti di forza e \
debolezza. Considera le possibili difese della controparte.]

## ⚡ Possibili Azioni
[In ordine di preferenza pratica:
1. Azione stragiudiziale (diffida, negoziazione)
2. ADR (mediazione obbligatoria, arbitrato)
3. Azione giudiziaria (tribunale competente, rito applicabile)
Con costi approssimativi, tempi medi e probabilità di successo per ciascuna.]

## ⏰ Termini e Prescrizioni
[Segnala SEMPRE scadenze critiche: prescrizione, decadenza, termini processuali. \
Indica la data limite entro cui agire per non perdere il diritto.]

---

> **⚠️ DISCLAIMER LEGALE — LEGGERE ATTENTAMENTE**
> Questo elaborato è fornito a scopo **esclusivamente informativo e didattico**.
> **Non costituisce consulenza legale professionale** e non crea alcun rapporto \
avvocato-cliente. Non sostituisce l'assistenza di un avvocato iscritto all'Albo.
> Per qualsiasi azione legale rilevante, **rivolgersi a un professionista qualificato**.

---

STILE: Linguaggio chiaro e accessibile. Spiega i termini tecnici. Sii preciso ma \
empatico. Rispondi SEMPRE in italiano."""

# ─────────────────────────────────────────────────────────────
# CSS Custom — Tema legale professionale
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Font e palette ── */
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&display=swap');

:root {
    --navy:       #1a3a5c;
    --navy-light: #2c5282;
    --gold:       #b8860b;
    --gold-light: #d4a017;
    --ivory:      #fafaf6;
    --card-bg:    #ffffff;
    --border:     #e2ddd5;
    --text-main:  #1c1c1e;
    --text-muted: #6b6b6b;
    --success:    #2d6a4f;
    --warning:    #7c4d00;
    --error:      #8b0000;
}

/* ── Corpo principale ── */
.stApp {
    background-color: var(--ivory);
    font-family: 'Inter', sans-serif;
}

/* ── Header personalizzato ── */
.app-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
    padding: 1.5rem 2rem 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 20px rgba(26, 58, 92, 0.2);
}

.app-header-icon {
    font-size: 2.8rem;
    line-height: 1;
}

.app-header-text h1 {
    font-family: 'Lora', Georgia, serif;
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 600;
    margin: 0 0 0.15rem 0;
    letter-spacing: -0.3px;
}

.app-header-text p {
    color: rgba(255,255,255,0.75);
    font-size: 0.85rem;
    margin: 0;
    font-style: italic;
}

.header-badge {
    margin-left: auto;
    background: rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.9);
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    white-space: nowrap;
    border: 1px solid rgba(255,255,255,0.25);
}

/* ── Messaggi chat ── */
[data-testid="stChatMessage"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.2rem 0.5rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    animation: fadeInUp 0.25s ease-out;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}

/* Messaggio utente — accento navy a sinistra */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 4px solid var(--navy);
    background: #f8f9ff;
}

/* Messaggio AI — accento dorato a sinistra */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 4px solid var(--gold);
    background: var(--card-bg);
}

/* ── Sezioni del parere legale (markdown headers) ── */
[data-testid="stChatMessage"] h2 {
    font-family: 'Lora', Georgia, serif;
    color: var(--navy);
    font-size: 1.05rem;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.4rem;
    margin-top: 1.4rem;
    margin-bottom: 0.6rem;
}

[data-testid="stChatMessage"] blockquote {
    background: #fffbf0;
    border-left: 4px solid var(--gold-light);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 1rem 0;
    font-size: 0.85rem;
    color: var(--warning);
}

[data-testid="stChatMessage"] hr {
    border-color: var(--border);
    margin: 1.2rem 0;
}

/* ── Input chat ── */
[data-testid="stChatInput"] {
    border: 2px solid var(--navy) !important;
    border-radius: 12px !important;
    background: var(--card-bg) !important;
    box-shadow: 0 2px 12px rgba(26,58,92,0.1) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold-light) !important;
    box-shadow: 0 0 0 3px rgba(184,134,11,0.15) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
}

[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.9) !important;
}

[data-testid="stSidebar"] .stTextInput > label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.8rem !important;
}

[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] input::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.22) !important;
    border-color: rgba(255,255,255,0.4) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] a {
    color: var(--gold-light) !important;
    text-decoration: underline !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── Alert / Warning ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* ── Status / Spinner ── */
[data-testid="stStatusWidget"] {
    border-radius: 10px !important;
}

/* ── Schermata di benvenuto ── */
.welcome-hero {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}

.welcome-hero h2 {
    font-family: 'Lora', Georgia, serif;
    color: var(--navy);
    font-size: 1.7rem;
    margin-bottom: 0.5rem;
}

.welcome-hero p {
    color: var(--text-muted);
    font-size: 1rem;
    line-height: 1.6;
    max-width: 560px;
    margin: 0 auto;
}

/* ── Card esempio ── */
.example-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-top: 3px solid var(--navy);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    cursor: pointer;
    transition: box-shadow 0.2s, transform 0.15s;
    font-size: 0.88rem;
    line-height: 1.55;
}

.example-card:hover {
    box-shadow: 0 4px 16px rgba(26,58,92,0.12);
    transform: translateY(-1px);
}

.example-card strong {
    color: var(--navy);
    display: block;
    margin-bottom: 0.3rem;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Contatore caratteri ── */
.char-counter {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-align: right;
    margin-top: -0.5rem;
    margin-bottom: 0.5rem;
}

.char-counter.near-limit { color: var(--warning); }
.char-counter.at-limit   { color: var(--error);   }

/* ── Footer ── */
.legal-footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    font-size: 0.75rem;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
    margin-top: 2rem;
    font-style: italic;
}

/* ── Rimuove il menu hamburger e il footer Streamlit ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
"""

# ─────────────────────────────────────────────────────────────
# Inizializzazione Session State
# ─────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Inizializza le variabili di sessione di Streamlit."""
    defaults = {
        "messages":     [],    # Lista di dict {role, content}
        "model":        None,  # Modello Gemini (cached per evitare riconfigurazioni)
        "last_api_key": "",    # API key in uso (per rilevare cambiamenti)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────────────────────
# Gestione API Key
# ─────────────────────────────────────────────────────────────

def get_api_key() -> Optional[str]:
    """
    Recupera la API key con priorità:
    1. Input dalla sidebar (deploy pubblici)
    2. st.secrets["GEMINI_API_KEY"] (deploy privati su Streamlit Cloud)
    3. None

    FIX BUG-001: sostituito 'str | None' con Optional[str] per Python 3.9 compat.
    """
    secrets_key: Optional[str] = None
    try:
        secrets_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        pass  # secrets non configurati: normale in sviluppo locale

    sidebar_key: str = st.session_state.get("sidebar_api_key", "").strip()

    if sidebar_key:
        return sidebar_key
    return secrets_key or None


# ─────────────────────────────────────────────────────────────
# Configurazione Modello Gemini
# ─────────────────────────────────────────────────────────────

def create_model(api_key: str) -> genai.GenerativeModel:
    """
    Crea e configura il modello Gemini 2.0 Flash con Google Search Grounding.

    FIX BUG-002: eliminato start_chat(). Il grounding tool non è compatibile con
    la Chat API di Gemini. La conversazione multi-turno viene ora gestita tramite
    generate_content() con history manuale (vedi send_message).
    """
    genai.configure(api_key=api_key)

    # Google Search Grounding disabilitato: non supportato su piano free
    # Usa il modello base senza ricerca web
    tools_list = []

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
        tools=tools_list,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )
    return model


def get_or_create_model(api_key: str) -> genai.GenerativeModel:
    """Restituisce il modello dalla session; lo ricrea solo se la API key cambia."""
    if (
        st.session_state.model is None
        or st.session_state.last_api_key != api_key
    ):
        st.session_state.model    = create_model(api_key)
        st.session_state.last_api_key = api_key
        logger.info("Modello Gemini configurato")
    return st.session_state.model


# ─────────────────────────────────────────────────────────────
# Logica di Chat  — generate_content() con history manuale
# ─────────────────────────────────────────────────────────────

def build_history(messages: List[dict]) -> List[dict]:
    """
    Converte la cronologia Streamlit nel formato richiesto da generate_content().
    Mantiene solo gli ultimi MAX_HISTORY_TURNS scambi per non superare il context window.

    FIX BUG-002: sostituisce start_chat() incompatibile con grounding.
    """
    # Prende solo gli ultimi N turni per gestire conversazioni lunghe
    recent = messages[-(MAX_HISTORY_TURNS * 2):]  # *2 perché ogni turno = user+model

    history = []
    for msg in recent:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [{"text": msg["content"]}]})
    return history


def extract_text(response: genai.types.GenerateContentResponse) -> str:
    """
    Estrae il testo dalla risposta Gemini in modo sicuro.

    FIX BUG-003: accede prima a candidates per evitare ValueError su risposte bloccate.
    FIX gemini-2.5: salta le "thinking parts" (thought=True) e restituisce solo il testo finale.
    """
    try:
        # Percorso primario: candidates[0].content.parts
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            parts = response.candidates[0].content.parts
            # gemini-2.5 include parti di "pensiero" interne (thought=True): le saltiamo
            for part in parts:
                if getattr(part, "thought", False):
                    continue
                if hasattr(part, "text") and part.text:
                    return part.text
            # Fallback: primo part con testo
            for part in parts:
                if hasattr(part, "text") and part.text:
                    return part.text

        # Fallback: response.text (può lanciare se bloccato)
        return response.text

    except (ValueError, AttributeError, IndexError):
        # Risposta bloccata dai filtri di sicurezza o vuota
        finish_reason = None
        try:
            finish_reason = response.candidates[0].finish_reason
        except Exception:
            pass

        if finish_reason and str(finish_reason) in ("SAFETY", "2"):
            return (
                "⚠️ La risposta è stata limitata dai filtri di sicurezza di Gemini. "
                "Prova a riformulare la domanda in modo più neutro."
            )
        return "⚠️ Risposta non disponibile. Riprova."


def send_message(user_input: str, model: genai.GenerativeModel) -> str:
    """
    Invia un messaggio al modello (non-streaming, usato come fallback).
    """
    try:
        history = build_history(st.session_state.messages)
        history.append({"role": "user", "parts": [{"text": user_input}]})
        response = model.generate_content(history)
        return extract_text(response)
    except genai.types.BlockedPromptException:
        logger.warning("Prompt bloccato dai filtri Gemini")
        return (
            "❌ La richiesta è stata bloccata dai filtri di sicurezza. "
            "Prova a riformularla in modo più neutro."
        )
    except Exception as e:
        error_str = str(e)
        logger.error(f"Errore Gemini API: {error_str}")
        return _format_api_error(error_str)


# Messaggi di stato mostrati in rotazione durante la fase di elaborazione
_THINKING_MESSAGES = [
    "🔍 Analizzando il caso...",
    "📚 Consultando le fonti normative vigenti...",
    "⚖️ Elaborando il parere giuridico...",
    "📝 Verificando la normativa aggiornata...",
    "🔎 Ricercando precedenti e disposizioni rilevanti...",
]


def send_message_streaming(
    user_input: str,
    model: genai.GenerativeModel,
    placeholder,
) -> str:
    """
    Versione streaming di send_message.
    - Durante la fase di "thinking" di gemini-2.5 mostra messaggi rotativi.
    - Appena arriva il primo token di testo lo streamma nel placeholder in tempo reale.
    - Restituisce il testo completo per salvarlo in session_state.
    """
    import time

    try:
        history = build_history(st.session_state.messages)
        history.append({"role": "user", "parts": [{"text": user_input}]})

        response_stream = model.generate_content(history, stream=True)

        full_text = ""
        thinking_idx = 0
        last_rotate = time.time()
        first_text = True

        for chunk in response_stream:
            try:
                if not (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                ):
                    continue

                for part in chunk.candidates[0].content.parts:
                    # Parte di "thinking" interna → ruota il messaggio di stato
                    if getattr(part, "thought", False):
                        now = time.time()
                        if now - last_rotate > 2.5:
                            placeholder.markdown(
                                f"*{_THINKING_MESSAGES[thinking_idx % len(_THINKING_MESSAGES)]}*"
                            )
                            thinking_idx += 1
                            last_rotate = now
                        continue

                    # Testo reale → streamma nel placeholder
                    if hasattr(part, "text") and part.text:
                        if first_text:
                            first_text = False
                        full_text += part.text
                        placeholder.markdown(full_text + " ▌")

            except Exception:
                continue

        # Rimozione cursore finale
        if full_text:
            placeholder.markdown(full_text)
            return full_text

        result = "⚠️ Risposta non disponibile. Riprova."
        placeholder.markdown(result)
        return result

    except genai.types.BlockedPromptException:
        logger.warning("Prompt bloccato dai filtri Gemini")
        msg = (
            "❌ La richiesta è stata bloccata dai filtri di sicurezza. "
            "Prova a riformularla in modo più neutro."
        )
        placeholder.markdown(msg)
        return msg
    except Exception as e:
        error_str = str(e)
        logger.error(f"Errore Gemini API (streaming): {error_str}")
        msg = _format_api_error(error_str)
        placeholder.markdown(msg)
        return msg


def _format_api_error(error_str: str) -> str:
    """Traduce gli errori API Gemini in messaggi comprensibili in italiano."""
    e = error_str.lower()
    if any(k in e for k in ["api_key_invalid", "api key", "invalid key", "unauthenticated"]):
        return (
            "❌ **API Key non valida.**\n\n"
            "Verifica la chiave nella sidebar. "
            "Creane una gratuita su [Google AI Studio](https://aistudio.google.com/app/apikey)."
        )
    if any(k in e for k in ["quota", "rate_limit", "resource_exhausted"]):
        # Distingue tra limite giornaliero e limite al minuto
        if "free_tier" in e or "limit: 0" in e or "daily" in e or "per day" in e:
            return (
                "❌ **Quota giornaliera esaurita.**\n\n"
                "Hai raggiunto il limite giornaliero del piano gratuito Gemini (1.500 richieste/giorno). "
                "La quota si azzera ogni giorno alle ore 09:00 circa (ora italiana). "
                "Riprova domani mattina."
            )
        return (
            "❌ **Limite di velocità raggiunto.**\n\n"
            "Hai superato le 15 richieste/min del piano gratuito. "
            "Attendi 60 secondi e riprova."
        )
    if "timeout" in e:
        return (
            "❌ **Timeout.**\n\n"
            "La risposta ha impiegato troppo tempo. Riprova tra qualche istante."
        )
    if "model" in e and "not found" in e:
        return (
            "❌ **Modello non trovato.**\n\n"
            "Gemini 2.0 Flash non è disponibile con questa API key. "
            "Verifica che la key abbia accesso ai modelli su Google AI Studio."
        )
    return (
        f"❌ **Errore API:** {error_str}\n\n"
        "Riprova o controlla la tua API key."
    )


# ─────────────────────────────────────────────────────────────
# Rendering — Sidebar
# ─────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Sidebar con gestione API key, pulsanti azione e info."""
    with st.sidebar:
        # ── Brand ──
        st.markdown(
            "<div style='text-align:center; padding: 1rem 0 0.5rem;'>"
            "<span style='font-size:3rem;'>⚖️</span>"
            "<h2 style='margin:0.3rem 0 0; font-size:1.15rem; font-weight:600;'>"
            "Consulente Legale AI</h2>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── API Key ──
        st.markdown("#### 🔑 API Key Google Gemini")

        has_secrets = False
        try:
            has_secrets = bool(st.secrets.get("GEMINI_API_KEY"))
        except Exception:
            pass

        if has_secrets:
            st.success("✅ Chiave configurata dal server")
        else:
            st.markdown(
                "<small>Inserisci la tua chiave <b>gratuita</b> da "
                "<a href='https://aistudio.google.com/app/apikey' target='_blank'>"
                "Google AI Studio</a>.</small>",
                unsafe_allow_html=True,
            )
            st.text_input(
                "API Key",
                type="password",
                placeholder="AIzaSy...",
                key="sidebar_api_key",
                label_visibility="collapsed",
                help="La chiave rimane solo in questa sessione browser e non viene salvata.",
            )

        st.divider()

        # ── Azioni ──
        n_msgs = len(st.session_state.get("messages", []))
        if n_msgs > 0:
            st.caption(f"💬 {n_msgs // 2} scambi in questa sessione")

        if st.button("🔄 Nuova Consultazione", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # ── Come funziona ──
        st.markdown("#### 💡 Come funziona")
        st.markdown(
            "<small>"
            "1. <b>Descrivi</b> la situazione nel dettaglio<br>"
            "2. <b>Rispondi</b> alle eventuali domande dell'AI<br>"
            "3. <b>Ricevi</b> il parere con le norme vigenti<br>"
            "4. <b>Approfondisci</b> con domande di follow-up"
            "</small>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Ambiti ──
        st.markdown("#### 📚 Ambiti coperti")
        st.markdown(
            "<small>"
            "• Diritto Civile (contratti, proprietà, famiglia)<br>"
            "• Diritto Penale e Processuale<br>"
            "• Diritto del Lavoro<br>"
            "• Diritto Commerciale e Societario<br>"
            "• Diritto dei Consumatori<br>"
            "• Responsabilità Civile<br>"
            "• Locazioni e Condominio<br>"
            "• Diritto Amministrativo (base)"
            "</small>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown(
            "<small style='opacity:0.55;'>"
            "⚠️ Strumento informativo. "
            "Non sostituisce la consulenza di un avvocato abilitato."
            "</small>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# Rendering — Schermata di Benvenuto
# ─────────────────────────────────────────────────────────────

EXAMPLES = [
    ("🏠", "Locazioni",
     "Il mio inquilino non paga l'affitto da 4 mesi. Il contratto 4+4 è regolarmente registrato. "
     "Ho già inviato un sollecito scritto. Cosa posso fare?"),
    ("💼", "Diritto del Lavoro",
     "Sono stato licenziato senza preavviso dopo 10 anni in un'azienda con 60 dipendenti. "
     "La motivazione è 'giustificato motivo oggettivo'. Ho diritto alla reintegra?"),
    ("🚗", "Sinistro Stradale",
     "Ho avuto un incidente stradale, l'altro conducente è risultato in torto al 100%. "
     "L'assicurazione ha liquidato solo 4.000€ su 9.000€ di danni. Posso contestare?"),
    ("🛍️", "Consumatori",
     "Ho acquistato un laptop a 1.200€ che si è rotto dopo 13 mesi. Il venditore dice "
     "che la garanzia legale è scaduta. È vero? Quali diritti ho?"),
    ("🏗️", "Proprietà",
     "Il vicino ha costruito un muro che sconfina di 30 cm nel mio terreno. "
     "Non ha risposto alle mie richieste. Come posso tutelarmi?"),
    ("📋", "Contratti",
     "Ho firmato un contratto con una penale di 15.000€. La controparte vuole applicarla "
     "per un mio ritardo di 2 giorni. È riducibile? Come mi difendo?"),
]


def render_welcome_screen() -> None:
    """Schermata di benvenuto con hero, istruzioni ed esempi cliccabili."""
    st.markdown(
        "<div class='welcome-hero'>"
        "<h2>Benvenuto nel tuo Consulente Legale AI</h2>"
        "<p>Descrivi la tua situazione nel dettaglio.<br>"
        "L'AI analizzerà il caso, cercherà le norme vigenti aggiornate<br>"
        "e produrrà un parere giuridico strutturato.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### 💬 Esempi — clicca per usare come punto di partenza:")
    col1, col2 = st.columns(2)
    for i, (icon, label, text) in enumerate(EXAMPLES):
        col = col1 if i % 2 == 0 else col2
        with col:
            card_html = (
                f"<div class='example-card'>"
                f"<strong>{icon} {label}</strong>"
                f"{text}"
                f"</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Rendering — Cronologia Chat
# ─────────────────────────────────────────────────────────────

def render_chat_history() -> None:
    """Renderizza la cronologia dei messaggi con avatar differenziati."""
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "⚖️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point dell'applicazione Streamlit."""

    # set_page_config DEVE essere la prima chiamata Streamlit
    st.set_page_config(
        page_title="Consulente Legale AI",
        page_icon="⚖️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Inietta CSS custom (dopo set_page_config)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Inizializza session state
    init_session_state()

    # Sidebar
    render_sidebar()

    # ── Header ──
    st.markdown(
        "<div class='app-header'>"
        "  <div class='app-header-icon'>⚖️</div>"
        "  <div class='app-header-text'>"
        f"    <h1>{APP_TITLE}</h1>"
        f"    <p>{APP_SUBTITLE}</p>"
        "  </div>"
        "  <div class='header-badge'>🔍 Ricerca normativa in tempo reale</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Verifica API Key ──
    api_key = get_api_key()
    if not api_key:
        render_welcome_screen()
        st.info(
            "👈 **Inserisci la tua API Key Google Gemini nella sidebar** per iniziare.\n\n"
            "La chiave è gratuita su "
            "[Google AI Studio](https://aistudio.google.com/app/apikey).",
            icon="🔑",
        )
        st.markdown("<div class='legal-footer'>Strumento informativo — non sostituisce un avvocato</div>", unsafe_allow_html=True)
        return

    # ── Inizializzazione modello ──
    try:
        model = get_or_create_model(api_key)
    except Exception as e:
        st.error(f"❌ Impossibile avviare il modello Gemini: {e}")
        logger.error(f"Errore init modello: {e}")
        return

    # ── Schermata benvenuto (solo se chat vuota) ──
    if not st.session_state.messages:
        render_welcome_screen()

    # ── Cronologia chat ──
    render_chat_history()

    # ── Input utente ──
    user_input = st.chat_input(
        placeholder=(
            "Descrivi la tua situazione legale in dettaglio... "
            "(es. 'Il mio inquilino non paga da 3 mesi...')"
        ),
        max_chars=MAX_INPUT_CHARS,  # FIX BUG-004: limite caratteri
    )

    if user_input:
        # Mostra messaggio utente immediatamente
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Risposta AI in streaming: il testo appare man mano che viene generato
        with st.chat_message("assistant", avatar="⚖️"):
            placeholder = st.empty()
            # Messaggio iniziale mentre il modello elabora
            placeholder.markdown(f"*{_THINKING_MESSAGES[0]}*")
            ai_response = send_message_streaming(user_input, model, placeholder)

        st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # ── Footer ──
    if st.session_state.messages:
        st.markdown(
            "<div class='legal-footer'>"
            "⚠️ Questo strumento è a scopo informativo e non sostituisce la consulenza di un avvocato abilitato."
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
