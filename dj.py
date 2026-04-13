import os
import json
import random
import datetime
import getpass
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# =========================================================
# CONFIG
# =========================================================

SURPRISE_PROBABILITY = 0.18  # ~1 in 5 runs

BANDS = {
    "energy": {
        "very_low": {"safe": (0.05, 0.20), "spicy": (0.05, 0.35)},
        "low":      {"safe": (0.20, 0.35), "spicy": (0.15, 0.45)},
        "medium":   {"safe": (0.45, 0.55), "spicy": (0.35, 0.70)},
        "high":     {"safe": (0.75, 0.85), "spicy": (0.65, 0.95)},
        "very_high":{"safe": (0.90, 1.00), "spicy": (0.80, 1.00)},
    },
    "valence": {
        "negative": {"safe": (0.10, 0.35), "spicy": (0.05, 0.45)},
        "neutral":  {"safe": (0.45, 0.55), "spicy": (0.35, 0.65)},
        "positive": {"safe": (0.70, 0.85), "spicy": (0.55, 0.95)},
    },
    "instrumentalness": {
        "vocal":         {"safe": (0.0, 0.2), "spicy": (0.0, 0.4)},
        "mixed":         {"safe": (0.3, 0.6), "spicy": (0.2, 0.7)},
        "instrumental":  {"safe": (0.85, 0.95), "spicy": (0.65, 1.0)},
    }
}

# =========================================================
# LOGGING
# =========================================================

def log(step, message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{step}] {message}")

# =========================================================
# UTILS
# =========================================================

def sample_band(band_name, band_value, profile):
    low, high = BANDS[band_name][band_value][profile]
    return round(random.uniform(low, high), 2)

def should_create_spicy():
    return random.random() < SURPRISE_PROBABILITY

# =========================================================
# AUTH
# =========================================================

log("INIT", "Initializing agent")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google API Key: ")

SPOTIPY_CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID") or input("Spotify Client ID: ")
SPOTIPY_CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET") or input("Spotify Client Secret: ")
SPOTIPY_REDIRECT_URI = "http://127.0.0.1:8889/callback"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-top-read playlist-modify-public playlist-modify-private"
))

# =========================================================
# TOOLS
# =========================================================

@tool
def get_current_context() -> str:
    """Get the current day of the week and time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %H:%M")

@tool
def get_my_top_artists() -> str:
    """Get the user's top 10 artists from Spotify."""
    results = sp.current_user_top_artists(limit=10, time_range="medium_term")
    artists = [{
        "name": a["name"],
        "id": a["id"],
        "genres": a.get("genres", [])
    } for a in results["items"]]
    return json.dumps(artists)

def create_playlist(
    name: str,
    seed_type: str,
    seed_value: str,
    energy: float,
    valence: float,
    instrumentalness: float
) -> str:
    """Create a Spotify playlist from user's library, filtered by audio features."""
    log("SPOTIFY", f"Creating playlist '{name}'")

    # Collect candidate tracks from user's own library (these endpoints work without Extended Quota)
    candidate_tracks = []
    seen_ids = set()

    def add_tracks(tracks):
        for t in tracks:
            # Handle saved tracks format (has a "track" wrapper)
            track = t.get("track", t) if isinstance(t, dict) and "track" in t else t
            if track and track["id"] not in seen_ids:
                seen_ids.add(track["id"])
                candidate_tracks.append(track)

    # Get top tracks across all time ranges for variety
    for time_range in ["short_term", "medium_term", "long_term"]:
        try:
            results = sp.current_user_top_tracks(limit=50, time_range=time_range)
            add_tracks(results["items"])
        except Exception as e:
            log("WARNING", f"Could not get top tracks ({time_range}): {e}")

    # Also get saved/liked tracks
    try:
        saved = sp.current_user_saved_tracks(limit=50)
        add_tracks(saved["items"])
    except Exception as e:
        log("WARNING", f"Could not get saved tracks: {e}")

    log("SPOTIFY", f"Collected {len(candidate_tracks)} unique candidate tracks from library")

    if not candidate_tracks:
        log("ERROR", "No candidate tracks found in user library")
        return "No tracks found"

    # Try to score by audio features; fall back to random selection if API is restricted
    track_uris = []
    try:
        track_ids = [t["id"] for t in candidate_tracks]
        features_list = sp.audio_features(track_ids)

        # Score tracks by proximity to target features
        scored_tracks = []
        for track, features in zip(candidate_tracks, features_list):
            if features is None:
                continue
            score = (
                abs(features.get("energy", 0.5) - energy) +
                abs(features.get("valence", 0.5) - valence) +
                abs(features.get("instrumentalness", 0.5) - instrumentalness)
            )
            scored_tracks.append((score, track))

        # Sort by score (lower = closer match) and take top 20
        scored_tracks.sort(key=lambda x: x[0])
        selected = scored_tracks[:20]
        track_uris = [t["uri"] for _, t in selected]

        if track_uris:
            log("SPOTIFY", f"Selected {len(track_uris)} tracks (best match score: {selected[0][0]:.2f})")
    except Exception as e:
        log("WARNING", f"Audio features unavailable ({e}), using random selection")

    # Fallback: random selection from candidates
    if not track_uris:
        import random as rnd
        sample = rnd.sample(candidate_tracks, min(20, len(candidate_tracks)))
        track_uris = [t["uri"] for t in sample]
        log("SPOTIFY", f"Randomly selected {len(track_uris)} tracks")

    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(
        user=user_id,
        name=name,
        public=False,
        description=f"AI DJ | Energy {energy} | Valence {valence}"
    )

    sp.playlist_add_items(playlist["id"], track_uris)
    return playlist["external_urls"]["spotify"]

# =========================================================
# LLM + PROMPT
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.3,
    max_output_tokens=2048,
    timeout=30,
)

SYSTEM_PROMPT = """
You are an Elite AI DJ.

Your task is to map human mood into structured musical intent.

You MUST output JSON with:
- mood_name (short, creative)
- energy_band: one of [very_low, low, medium, high, very_high]
- valence_band: one of [negative, neutral, positive]
- instrumentalness_band: one of [vocal, mixed, instrumental]

You MUST also choose TWO seeds:
- safe_seed: aligned with user's listening history
- spicy_seed: exploratory but mood-compatible

Each seed must be:
{
  "type": "artist" | "genre",
  "value": "<spotify_id_or_genre>"
}

Rules:
- If user mentions focus/study/code/read → instrumentalness_band MUST be "instrumental"
- Safe seed prioritizes taste match
- Spicy seed prioritizes novelty without breaking mood
- Do NOT output numeric values
- Do NOT explain reasoning
"""

agent_executor = create_react_agent(
    llm,
    tools=[get_current_context, get_my_top_artists],
    prompt=SYSTEM_PROMPT,
)

# =========================================================
# RUNTIME
# =========================================================

if __name__ == "__main__":
    log("READY", "🎧 AI DJ is live")
    user_input = input(">> How are you feeling or what do you need? ")

    log("LLM", "Analyzing intent")
    result = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})

    # Debug: show all message types
    for i, msg in enumerate(result["messages"]):
        content = msg.content
        if isinstance(content, list):
            content = str(content)[:200]
        elif isinstance(content, str):
            content = content[:200]
        tool_calls = getattr(msg, "tool_calls", [])
        log("DEBUG", f"Message {i} ({type(msg).__name__}): content={content!r}, tool_calls={tool_calls}")

    # Extract JSON from the agent response - search messages in reverse
    import re
    payload = None
    for msg in reversed(result["messages"]):
        content = msg.content
        if isinstance(content, list):
            content = "".join(
                block if isinstance(block, str) else block.get("text", "")
                for block in content
            )
        if not content:
            continue
        # Try to find a JSON object in the content
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
                break
            except json.JSONDecodeError:
                continue

    if payload is None:
        log("ERROR", "Could not extract JSON from LLM response")
        exit(1)

    log("INTENT", json.dumps(payload, indent=2))

    energy_safe = sample_band("energy", payload["energy_band"], "safe")
    valence_safe = sample_band("valence", payload["valence_band"], "safe")
    instr_safe = sample_band("instrumentalness", payload["instrumentalness_band"], "safe")

    safe_url = create_playlist(
        name=f"AI DJ: {payload['mood_name']}",
        seed_type=payload["safe_seed"]["type"],
        seed_value=payload["safe_seed"]["value"],
        energy=energy_safe,
        valence=valence_safe,
        instrumentalness=instr_safe
    )

    log("DONE", f"Safe playlist created → {safe_url}")

    if should_create_spicy():
        log("SURPRISE", "🌶️ Spicy playlist triggered")

        energy_spicy = sample_band("energy", payload["energy_band"], "spicy")
        valence_spicy = sample_band("valence", payload["valence_band"], "spicy")
        instr_spicy = sample_band("instrumentalness", payload["instrumentalness_band"], "spicy")

        spicy_url = create_playlist(
            name=f"AI DJ: {payload['mood_name']} 🌶️",
            seed_type=payload["spicy_seed"]["type"],
            seed_value=payload["spicy_seed"]["value"],
            energy=energy_spicy,
            valence=valence_spicy,
            instrumentalness=instr_spicy
        )

        log("DONE", f"Spicy playlist created → {spicy_url}")
    else:
        log("SURPRISE", "No spicy playlist this time")
