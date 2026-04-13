import os
import json
import random
import datetime
import getpass
import re
import argparse
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain / LangGraph Imports
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# =========================================================
# CONFIGURATION
# =========================================================

SURPRISE_PROBABILITY = 0.18  # ~1 in 5 runs will trigger the spicy playlist

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
        "vocal":         {"safe": (0.0,  0.2),  "spicy": (0.0,  0.4)},
        "mixed":         {"safe": (0.3,  0.6),  "spicy": (0.2,  0.7)},
        "instrumental":  {"safe": (0.85, 0.95), "spicy": (0.65, 1.0)},
    }
}

# Maps mood bands to Spotify search keywords
ENERGY_KEYWORDS = {
    "very_low": "calm ambient",
    "low":      "chill mellow",
    "medium":   "indie alternative",
    "high":     "energetic upbeat",
    "very_high":"intense hype",
}
VALENCE_KEYWORDS = {
    "negative": "sad melancholic",
    "neutral":  "mellow thoughtful",
    "positive": "happy feel good",
}
INSTRUMENTALNESS_KEYWORDS = {
    "vocal":        "",             # default — no extra keyword needed
    "mixed":        "lo-fi",
    "instrumental": "instrumental",
}

# Keys the LLM must include in its JSON response
REQUIRED_PAYLOAD_KEYS = {"energy_band", "valence_band", "instrumentalness_band"}

# =========================================================
# UTILITIES
# =========================================================

def log(step, message):
    """Consistent logging format for the console."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{step}] {message}")

def sample_band(band_name, band_value, profile):
    """Samples a random value within the specified band for the Spotify API."""
    low, high = BANDS[band_name][band_value][profile]
    return round(random.uniform(low, high), 2)

def should_create_spicy():
    """Returns True if a spicy playlist should be created."""
    return random.random() < SURPRISE_PROBABILITY

def clean_json_output(text: str) -> str:
    """
    Cleans up LLM output that might be wrapped in Markdown code fences.
    LLMs often output JSON as: ```json\n{...}\n```
    This function extracts just the JSON part to prevent json.loads() from failing.
    """
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

# =========================================================
# AUTHENTICATION (deferred — call init_clients() from __main__)
# =========================================================

sp = None  # Initialized lazily via init_clients()

def init_clients():
    """
    Initializes the Spotify client. Must be called once from __main__ before
    using any Spotify functionality. Reads credentials from .env or prompts the user.
    """
    global sp

    log("INIT", "Initializing Spotify client...")

    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter Anthropic API Key: ")

    client_id     = os.environ.get("SPOTIPY_CLIENT_ID")     or input("Spotify Client ID: ")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET") or input("Spotify Client Secret: ")

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri="http://127.0.0.1:8889/callback",
        scope="user-top-read playlist-modify-public playlist-modify-private",
        show_dialog=True,  # Force re-consent to ensure all scopes are granted
    ))

    log("INIT", "Spotify client ready.")

# =========================================================
# LANGCHAIN TOOLS
# =========================================================

@tool
def get_current_context() -> str:
    """
    Provides the current day of the week and time.
    Useful for understanding the user's temporal context (e.g., Friday night vs Monday morning).
    """
    now = datetime.datetime.now()
    return now.strftime("%A, %H:%M")

@tool
def get_weather(city: str) -> str:
    """
    Returns the current weather for a given city.
    Useful for enriching mood context (e.g., rainy day -> calmer playlist).
    Requires OPENWEATHERMAP_API_KEY in the .env file.
    """
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "Weather data unavailable (no API key configured)"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        if response.status_code != 200:
            return f"Could not fetch weather for '{city}': {data.get('message', 'unknown error')}"
        weather_desc = data["weather"][0]["description"].capitalize()
        temp         = data["main"]["temp"]
        humidity     = data["main"]["humidity"]
        return f"{weather_desc}, {temp}°C, {humidity}% humidity"
    except Exception as e:
        return f"Weather lookup failed: {e}"

@tool
def search_spotify(query: str) -> str:
    """
    Searches Spotify for artists or tracks matching the query.
    Useful when the user mentions a specific artist or song by name.
    Returns the top 5 results with name, ID, and type (artist/track).
    """
    try:
        results = sp.search(q=query, limit=10, type="artist,track")
        output = []
        for artist in results.get("artists", {}).get("items", []):
            output.append({
                "type":   "artist",
                "name":   artist["name"],
                "id":     artist["id"],
                "genres": artist.get("genres", [])
            })
        for track in results.get("tracks", {}).get("items", []):
            output.append({
                "type":   "track",
                "name":   track["name"],
                "id":     track["id"],
                "artist": track["artists"][0]["name"] if track["artists"] else "Unknown"
            })
        return json.dumps(output)
    except Exception as e:
        return f"Spotify search failed: {e}"

@tool
def get_audio_features(track_id: str) -> str:
    """
    Returns the audio features of a specific Spotify track.
    Useful when the user says 'make a playlist like this song'.
    Returns energy, valence, danceability, tempo, acousticness, and instrumentalness.
    """
    try:
        features = sp.audio_features([track_id])
        if not features or not features[0]:
            return f"No audio features found for track ID: {track_id}"
        f = features[0]
        return json.dumps({
            "energy":           f.get("energy"),
            "valence":          f.get("valence"),
            "danceability":     f.get("danceability"),
            "tempo":            f.get("tempo"),
            "acousticness":     f.get("acousticness"),
            "instrumentalness": f.get("instrumentalness")
        })
    except Exception as e:
        return f"Audio features lookup failed: {e}"

def get_top_artists(limit=5) -> list:
    """
    Fetches the user's top artists (id + genres) from Spotify.
    Returns a list of dicts: [{"id": ..., "name": ..., "genres": [...]}, ...]
    Returns an empty list on failure instead of crashing.
    """
    try:
        results = sp.current_user_top_artists(limit=limit, time_range="medium_term")
        return [
            {"id": a["id"], "name": a["name"], "genres": a.get("genres", [])}
            for a in results.get("items", []) if a.get("id")
        ]
    except Exception as e:
        log("SPOTIFY_ERROR", f"Failed to fetch top artists: {e}")
        return []

def get_spicy_genre_seeds(top_artists: list, limit=2) -> list:
    """
    Derives genre seeds from the user's top artists.
    Picks the least common genres to push toward discovery territory.
    Falls back to ['indie', 'electronic'] if no genre data is available.
    """
    genre_counts = {}
    for artist in top_artists:
        for genre in artist.get("genres", []):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    sorted_genres = sorted(genre_counts, key=lambda g: genre_counts[g])
    spicy = sorted_genres[:limit]
    return spicy if spicy else ["indie", "electronic"]

def build_search_query(
    energy_band: str,
    valence_band: str,
    instrumentalness_band: str,
    genre_hint: str = "",
) -> str:
    """
    Builds a Spotify search query string from mood band labels.
    Combines energy, valence, and instrumentalness keywords with an optional
    genre hint (e.g. a genre from the user's top artists).

    Replaces sp.recommendations() which was deprecated in November 2024.
    """
    parts = [
        VALENCE_KEYWORDS[valence_band],
        ENERGY_KEYWORDS[energy_band],
        INSTRUMENTALNESS_KEYWORDS[instrumentalness_band],
        genre_hint,
    ]
    return " ".join(p for p in parts if p).strip()

def get_tracks_by_mood(
    energy_band: str,
    valence_band: str,
    instrumentalness_band: str,
    genre_hint: str = "",
    limit: int = 20,
) -> list:
    """
    Searches Spotify for tracks matching the mood bands using keyword search.
    Returns a shuffled list of track URIs for variety across repeated runs.

    Replaces sp.recommendations() which was deprecated in November 2024.
    Uses pagination (max 10 per request) due to Spotify API limit changes (Feb 2026).
    """
    query = build_search_query(energy_band, valence_band, instrumentalness_band, genre_hint)
    log("SPOTIFY", f"Search query: '{query}'")

    try:
        # Spotify search limit is now capped at 10, so paginate to collect enough tracks
        tracks = []
        page_limit = 10
        for offset in range(0, limit * 2, page_limit):
            results = sp.search(q=query, type="track", limit=page_limit, offset=offset)
            items = results.get("tracks", {}).get("items", [])
            tracks.extend(t["uri"] for t in items)
            if len(items) < page_limit:
                break  # No more results available
        # Deduplicate, shuffle, and trim
        tracks = list(dict.fromkeys(tracks))  # preserve order but remove dupes
        random.shuffle(tracks)
        return tracks[:limit]
    except Exception as e:
        log("SPOTIFY_ERROR", f"Track search failed: {e}")
        return []

def create_playlist(
    name: str,
    energy_band: str,
    valence_band: str,
    instrumentalness_band: str,
    genre_hint: str = "",
) -> str:
    """
    Creates a Spotify playlist using mood-informed keyword search.
    Uses sp.search() instead of the deprecated sp.recommendations() endpoint.
    """
    if not name or not name.strip():
        name = "Mood DJ Mix"
    log("SPOTIFY", f"Creating playlist '{name}'")

    tracks = get_tracks_by_mood(
        energy_band=energy_band,
        valence_band=valence_band,
        instrumentalness_band=instrumentalness_band,
        genre_hint=genre_hint,
        limit=20,
    )

    if not tracks:
        log("API_FALLBACK", "Mood search returned no tracks. Falling back to generic query...")
        tracks = []
        for offset in range(0, 20, 10):
            results = sp.search(q="feel good pop", type="track", limit=10, offset=offset)
            items = results.get("tracks", {}).get("items", [])
            tracks.extend(t["uri"] for t in items)
            if len(items) < 10:
                break

    playlist = sp._post(
        "me/playlists",
        payload={
            "name": name,
            "public": False,
            "description": f"Mood DJ | {energy_band} energy | {valence_band} valence",
        },
    )

    sp._post(
        f"playlists/{playlist['id']}/items",
        payload={"uris": tracks},
    )
    return playlist["external_urls"]["spotify"]

# =========================================================
# AGENT SETUP
# =========================================================

# 1. Structured LLM — temperature=0.0 for deterministic band classification
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.0,
    max_tokens=150,
)

# 2. Creative LLM — temperature=0.7 for evocative playlist name generation
creative_llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.7,
    max_tokens=50,
)

# 3. System Prompt
SYSTEM_PROMPT = """
You are an Elite Mood DJ. Your ONLY job is to output a JSON object.

You have access to tools:
- get_current_context: returns current day and time
- get_weather: returns weather for a city
- search_spotify: searches artists/tracks by name
- get_audio_features: returns audio features of a track

STRICT RULES — NO EXCEPTIONS:
- Output ONLY a raw JSON object. No text before or after. No markdown. No questions.
- NEVER ask the user for clarification. Make your best inference from what they said.
- If the input is vague (e.g. "so so", "ok", "meh"), map it to neutral/medium defaults.
- Do NOT greet, explain, or comment. JSON only.

Required output format:
{
  "energy_band": <one of: very_low, low, medium, high, very_high>,
  "valence_band": <one of: negative, neutral, positive>,
  "instrumentalness_band": <one of: vocal, mixed, instrumental>
}

Examples:
- "so so" -> {"energy_band": "low", "valence_band": "neutral", "instrumentalness_band": "vocal"}
- "pumped for the gym" -> {"energy_band": "very_high", "valence_band": "positive", "instrumentalness_band": "vocal"}
- "coding late at night" -> {"energy_band": "medium", "valence_band": "neutral", "instrumentalness_band": "instrumental"}
"""

# 4. Bind tools and create the ReAct agent
tools = [get_current_context, get_weather, search_spotify, get_audio_features]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT
)

def generate_mood_name(user_input: str, payload: dict) -> str:
    """
    Generates a short, creative playlist name using the creative LLM (temperature=0.7).
    """
    prompt = (
        f"The user said: \"{user_input}\"\n"
        f"The mood was mapped to: energy={payload['energy_band']}, "
        f"valence={payload['valence_band']}, instrumentalness={payload['instrumentalness_band']}.\n"
        "Give this playlist a short, creative, evocative name (3-5 words max). "
        "Output ONLY the name, nothing else."
    )
    response = creative_llm.invoke(prompt)
    return response.content.strip().strip('"')

# =========================================================
# MAIN RUNTIME
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mood DJ - mood-based Spotify playlist generator")
    parser.add_argument("--debug", action="store_true", help="Print full agent message chain and raw LLM output")
    args = parser.parse_args()

    init_clients()

    log("READY", "🎧 Mood DJ is live")
    user_input = input(">> How are you feeling or what do you need? ")

    log("LLM", "Agent is analyzing intent and invoking tools...")
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

    if args.debug:
        print("\n--- AGENT MESSAGE CHAIN ---")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            if isinstance(msg.content, str):
                content_preview = msg.content[:100] if msg.content else "(empty)"
            elif isinstance(msg.content, list):
                content_preview = str([
                    x.get("text", "") for x in msg.content
                    if isinstance(x, dict) and "text" in x
                ])[:100] or "(tool calls only)"
            else:
                content_preview = "(unknown format)"
            print(f"  [{i}] {msg_type}: {content_preview}")
        print("---------------------------\n")

    raw_output = ""
    for msg in reversed(result["messages"]):
        content = msg.content
        if isinstance(content, list):
            text_parts = [x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text"]
            content = "\n".join(text_parts)
        if isinstance(content, str) and content.strip():
            raw_output = content
            break

    if args.debug:
        print("\n--- LLM RAW OUTPUT ---")
        print(raw_output)
        print("----------------------\n")

    if not raw_output:
        log("ERROR", "Agent did not return any text content. Cannot proceed.")
        exit(1)

    clean_json_str = clean_json_output(raw_output)

    try:
        payload = json.loads(clean_json_str)
    except json.JSONDecodeError as e:
        log("ERROR", f"Failed to parse LLM output as JSON: {e}")
        log("RAW OUTPUT", raw_output)
        exit(1)

    missing_keys = REQUIRED_PAYLOAD_KEYS - payload.keys()
    if missing_keys:
        log("ERROR", f"LLM response is missing required keys: {missing_keys}")
        log("RAW OUTPUT", raw_output)
        exit(1)

    log("INTENT", json.dumps(payload, indent=2))

    mood_name = generate_mood_name(user_input, payload)
    log("MOOD", f"Playlist name: '{mood_name}'")

    # Fetch top artists — used to derive a genre hint for the search query
    log("SPOTIFY", "Fetching top artists for genre hint...")
    top_artists = get_top_artists(limit=5)
    log("SPOTIFY", f"Found {len(top_artists)} top artists")

    # Genre hints personalize the search query without needing the recommendations API
    genre_hints  = get_spicy_genre_seeds(top_artists)
    safe_genre   = genre_hints[0]  if genre_hints else ""
    spicy_genre  = genre_hints[-1] if genre_hints else ""

    # ── Summary before creation ──────────────────────────────────────
    spicy_triggered = should_create_spicy()

    # Compute the audio feature target ranges for display
    profile = "spicy" if spicy_triggered else "safe"
    energy_range   = BANDS["energy"][payload["energy_band"]]["safe"]
    valence_range  = BANDS["valence"][payload["valence_band"]]["safe"]
    instr_range    = BANDS["instrumentalness"][payload["instrumentalness_band"]]["safe"]

    # Preview the search query
    preview_query = build_search_query(
        payload["energy_band"],
        payload["valence_band"],
        payload["instrumentalness_band"],
        safe_genre,
    )

    print("\n" + "=" * 60)
    print("  🎧  Mood DJ — Playlist Summary")
    print("=" * 60)
    print(f"  📝 You said:           \"{user_input}\"")
    print(f"  🎵 Playlist name:      {mood_name}")
    print()
    print("  🧠 Detected Mood Bands:")
    print(f"     • Energy:           {payload['energy_band']}")
    print(f"     • Valence:          {payload['valence_band']}")
    print(f"     • Instrumentalness: {payload['instrumentalness_band']}")
    print()
    print("  🎚️  Audio Feature Targets (Spotify scale 0.0–1.0):")
    print(f"     • Energy:           {energy_range[0]:.2f} – {energy_range[1]:.2f}")
    print(f"     • Valence:          {valence_range[0]:.2f} – {valence_range[1]:.2f}")
    print(f"     • Instrumentalness: {instr_range[0]:.2f} – {instr_range[1]:.2f}")
    print()
    print("  🔍 Search query:       \"" + preview_query + "\"")
    print()
    print("  👤 Top artists used for genre hints:")
    for a in top_artists:
        genres_str = ", ".join(a["genres"][:3]) if a["genres"] else "no genres"
        print(f"     • {a['name']}  ({genres_str})")
    print(f"  🏷️  Genre hint (safe):  {safe_genre or '(none)'}")
    if spicy_triggered:
        print(f"  🌶️  Genre hint (spicy): {spicy_genre or '(none)'}")
        print(f"  🎲 Spicy playlist:     YES — surprise triggered!")
    else:
        print(f"  🎲 Spicy playlist:     No (not triggered this run)")
    print("=" * 60)

    confirm = input("\n>> Ready to create? Type 'yes' to proceed: ").strip().lower()
    if confirm not in ("yes", "y"):
        log("ABORT", "Playlist creation cancelled by user.")
        exit(0)

    print()

    # --- Safe playlist ---
    safe_url = create_playlist(
        name=f"Mood DJ: {mood_name}",
        energy_band=payload["energy_band"],
        valence_band=payload["valence_band"],
        instrumentalness_band=payload["instrumentalness_band"],
        genre_hint=safe_genre,
    )
    log("DONE", f"Safe playlist created -> {safe_url}")

    # --- Spicy playlist (occasional surprise) ---
    if spicy_triggered:
        log("SURPRISE", "🌶️  Spicy playlist triggered")
        log("SURPRISE", f"Spicy genre hint: '{spicy_genre}'")

        spicy_url = create_playlist(
            name=f"Mood DJ: {mood_name} 🌶️",
            energy_band=payload["energy_band"],
            valence_band=payload["valence_band"],
            instrumentalness_band=payload["instrumentalness_band"],
            genre_hint=spicy_genre,
        )
        log("DONE", f"Spicy playlist created -> {spicy_url}")
    else:
        log("SURPRISE", "No spicy playlist this time")