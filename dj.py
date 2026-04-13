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


# Comprehensive genre whitelist — the LLM must pick from this list.
# Each entry should return good results when used as a Spotify search term.
GENRE_WHITELIST = {
    # Pop & Mainstream
    "pop", "indie pop", "synth pop", "electropop", "dream pop",
    "k-pop", "j-pop", "latin pop", "dance pop", "art pop",
    # Rock
    "rock", "indie rock", "alternative rock", "classic rock", "punk rock",
    "hard rock", "grunge", "post-punk", "shoegaze", "emo",
    "progressive rock", "psychedelic rock", "garage rock", "soft rock",
    # Metal
    "metal", "heavy metal", "death metal", "black metal", "metalcore",
    # Electronic
    "electronic", "edm", "house", "deep house", "tech house",
    "techno", "trance", "dubstep", "drum and bass", "ambient",
    "synthwave", "chillwave", "downtempo", "future bass",
    "lo-fi", "lo-fi hip hop", "chillhop",
    # Hip-Hop & R&B
    "hip-hop", "rap", "trap", "r&b", "neo-soul", "soul", "funk",
    "boom bap", "conscious hip-hop", "drill",
    # Latin
    "reggaeton", "latin", "salsa", "bachata", "merengue",
    "samba", "bossa nova", "mpb", "forró", "sertanejo", "pagode",
    "cumbia", "tango", "vallenato", "reggaeton romantico",
    # Jazz & Blues
    "jazz", "smooth jazz", "blues", "swing", "bebop", "jazz fusion",
    # Classical & Instrumental
    "classical", "piano", "orchestral", "soundtrack", "new age",
    "chamber music", "opera",
    # Country & Folk
    "country", "folk", "bluegrass", "americana", "singer-songwriter",
    "celtic", "indie folk",
    # Reggae & Caribbean
    "reggae", "dancehall", "ska", "dub",
    # African & World
    "afrobeat", "afropop", "world music", "highlife",
    # Other
    "gospel", "disco", "new wave", "post-rock", "math rock",
    "noise", "experimental", "vaporwave", "phonk",
}

# Single mood keyword per valence — used to flavor genre-based queries
MOOD_KEYWORDS = {
    "negative": "sad",
    "neutral":  "chill",
    "positive": "upbeat",
}

# Keys the LLM must include in its JSON response
REQUIRED_PAYLOAD_KEYS = {"energy_band", "valence_band", "instrumentalness_band", "wants_twist", "skip_top_artists", "genres"}

# =========================================================
# UTILITIES
# =========================================================

def log(step, message):
    """Consistent logging format for the console."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{step}] {message}")


def validate_genres(genres: list) -> list:
    """Validates LLM-picked genres against the whitelist. Returns only valid ones."""
    valid = [g for g in genres if g.lower() in GENRE_WHITELIST]
    return valid if valid else ["pop"]  # safe fallback


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

def get_twist_genre_seeds(top_artists: list, limit=2) -> list:
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
    twist = sorted_genres[:limit]
    return twist if twist else ["indie", "electronic"]

def build_search_queries(
    genres: list,
    valence_band: str,
    genre_hint: str = "",
) -> list:
    """
    Builds multiple focused Spotify search queries from validated genres
    and a mood keyword. Returns a list of query strings.

    Strategy:
      - Query per genre (pure genre match)
      - Query per genre + mood keyword (mood-flavored)
      - Optional genre_hint from top artists for personalization
    """
    mood_word = MOOD_KEYWORDS.get(valence_band, "")
    queries = []

    for genre in genres:  # use all validated genres
        queries.append(genre)                               # pure genre
        if mood_word:
            queries.append(f"{genre} {mood_word}")          # genre + mood

    # Add a genre_hint-flavored query if available
    if genre_hint and genre_hint not in genres:
        queries.append(f"{genre_hint} {mood_word}".strip())

    return queries

def get_tracks_by_search(
    genres: list,
    valence_band: str,
    genre_hint: str = "",
    limit: int = 20,
) -> list:
    """
    Searches Spotify using multiple focused queries for better results.
    Runs each query separately and merges results for variety.

    Uses pagination (max 10 per request) due to Spotify API limit changes (Feb 2026).
    """
    queries = build_search_queries(genres, valence_band, genre_hint)
    all_tracks = []

    for query in queries:
        log("SPOTIFY", f"Search query: '{query}'")
        try:
            page_limit = 10
            for offset in range(0, 20, page_limit):
                results = sp.search(q=query, type="track", limit=page_limit, offset=offset)
                items = results.get("tracks", {}).get("items", [])
                all_tracks.extend(t["uri"] for t in items)
                if len(items) < page_limit:
                    break
        except Exception as e:
            log("SPOTIFY_ERROR", f"Search failed for '{query}': {e}")

    # Deduplicate, shuffle, and trim
    all_tracks = list(dict.fromkeys(all_tracks))
    random.shuffle(all_tracks)
    log("SPOTIFY", f"Collected {len(all_tracks)} unique tracks from {len(queries)} queries")
    return all_tracks[:limit]

def create_playlist(
    name: str,
    genres: list,
    valence_band: str,
    genre_hint: str = "",
) -> str:
    """
    Creates a Spotify playlist using genre-based keyword search.
    Runs multiple focused queries (genre + mood) for better results.
    """
    if not name or not name.strip():
        name = "Mood DJ Mix"
    log("SPOTIFY", f"Creating playlist '{name}'")

    tracks = get_tracks_by_search(
        genres=genres,
        valence_band=valence_band,
        genre_hint=genre_hint,
        limit=20,
    )

    if not tracks:
        log("API_FALLBACK", "Genre search returned no tracks. Falling back to generic query...")
        tracks = []
        for offset in range(0, 20, 10):
            results = sp.search(q="popular music", type="track", limit=10, offset=offset)
            items = results.get("tracks", {}).get("items", [])
            tracks.extend(t["uri"] for t in items)
            if len(items) < 10:
                break

    playlist = sp._post(
        "me/playlists",
        payload={
            "name": name,
            "public": False,
            "description": f"Mood DJ | genres: {', '.join(genres)} | mood: {valence_band}",
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
    max_tokens=350,
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
  "instrumentalness_band": <one of: vocal, mixed, instrumental>,
  "wants_twist": <true or false>,
  "skip_top_artists": <true or false>,
  "genres": [<1 to 3 genres from the allowed list>]
}

Field rules:
- "wants_twist": set to true ONLY when the user explicitly asks to be surprised, wants a twist,
  wants something unexpected, or asks for a wildcard. Default is false.
- "skip_top_artists": set to true ONLY when the user explicitly asks to hear something different
  from their usual taste, wants to discover new music, or says they are tired of the same artists.
  Default is false.
- "genres": pick genres that best match the user's intent from the allowed list below.
  IMPORTANT RULE:
  - If the user asks for a SPECIFIC genre or style (e.g. "samba", "jazz", "metal"), pick only 1–2
    closely related genres that match what they asked for. Stay focused.
  - If the user describes a VAGUE mood or feeling (e.g. "I'm happy", "feeling chill", "pumped"),
    pick up to 5 diverse genres that fit that mood for maximum variety.
  You MUST choose from this list:
  pop, indie pop, synth pop, electropop, dream pop, k-pop, j-pop, latin pop, dance pop, art pop,
  rock, indie rock, alternative rock, classic rock, punk rock, hard rock, grunge, post-punk,
  shoegaze, emo, progressive rock, psychedelic rock, garage rock, soft rock,
  metal, heavy metal, death metal, black metal, metalcore,
  electronic, edm, house, deep house, tech house, techno, trance, dubstep, drum and bass,
  ambient, synthwave, chillwave, downtempo, future bass, lo-fi, lo-fi hip hop, chillhop,
  hip-hop, rap, trap, r&b, neo-soul, soul, funk, boom bap, conscious hip-hop, drill,
  reggaeton, latin, salsa, bachata, merengue, samba, bossa nova, mpb, forró, sertanejo,
  pagode, cumbia, tango, vallenato, reggaeton romantico,
  jazz, smooth jazz, blues, swing, bebop, jazz fusion,
  classical, piano, orchestral, soundtrack, new age, chamber music, opera,
  country, folk, bluegrass, americana, singer-songwriter, celtic, indie folk,
  reggae, dancehall, ska, dub,
  afrobeat, afropop, world music, highlife,
  gospel, disco, new wave, post-rock, math rock, noise, experimental, vaporwave, phonk.
  Do NOT invent genres outside this list.

Examples:
- "so so" -> {"energy_band": "low", "valence_band": "neutral", "instrumentalness_band": "vocal", "wants_twist": false, "skip_top_artists": false, "genres": ["indie pop", "soft rock", "chillwave", "dream pop"]}
- "pumped for the gym" -> {"energy_band": "very_high", "valence_band": "positive", "instrumentalness_band": "vocal", "wants_twist": false, "skip_top_artists": false, "genres": ["edm", "trap", "hip-hop", "drill", "phonk"]}
- "I want some samba in my life!" -> {"energy_band": "high", "valence_band": "positive", "instrumentalness_band": "vocal", "wants_twist": false, "skip_top_artists": false, "genres": ["samba", "pagode"]}
- "surprise me!" -> {"energy_band": "high", "valence_band": "positive", "instrumentalness_band": "mixed", "wants_twist": true, "skip_top_artists": false, "genres": ["electronic", "funk", "afrobeat", "disco"]}
- "I want something totally different from what I usually listen to" -> {"energy_band": "medium", "valence_band": "neutral", "instrumentalness_band": "vocal", "wants_twist": false, "skip_top_artists": true, "genres": ["jazz", "bossa nova", "soul", "blues", "swing"]}
- "give me some jazz" -> {"energy_band": "medium", "valence_band": "neutral", "instrumentalness_band": "mixed", "wants_twist": false, "skip_top_artists": false, "genres": ["jazz", "smooth jazz"]}
- "coding late at night" -> {"energy_band": "medium", "valence_band": "neutral", "instrumentalness_band": "instrumental", "wants_twist": false, "skip_top_artists": false, "genres": ["lo-fi hip hop", "chillhop", "ambient", "downtempo"]}
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

    # ── Flags and genres from LLM ──────────────────────────────────────
    wants_twist      = payload.get("wants_twist", False)
    skip_top_artists = payload.get("skip_top_artists", False)
    raw_genres       = payload.get("genres", ["pop"])
    genres           = validate_genres(raw_genres)

    log("FLAGS", f"wants_twist={wants_twist}, skip_top_artists={skip_top_artists}")
    log("GENRES", f"LLM picked: {raw_genres} -> validated: {genres}")

    # Fetch top artists — used to derive a genre hint for the search query
    top_artists = []
    if not skip_top_artists:
        log("SPOTIFY", "Fetching top artists for genre hint...")
        top_artists = get_top_artists(limit=5)
        log("SPOTIFY", f"Found {len(top_artists)} top artists")
    else:
        log("SPOTIFY", "Skipping top artists (user wants something different from usual)")

    # Genre hints personalize the search query without needing the recommendations API
    genre_hints   = get_twist_genre_seeds(top_artists) if top_artists else []
    safe_genre    = genre_hints[0]  if genre_hints else ""
    twist_genre   = genre_hints[-1] if genre_hints else ""

    # Preview the search queries for the summary
    preview_queries = build_search_queries(
        genres,
        payload["valence_band"],
        safe_genre if not skip_top_artists else "",
    )

    # ── Summary before creation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🎧  Mood DJ — Playlist Summary")
    print("=" * 60)
    print(f"  📝 You said:           \"{user_input}\"")
    print(f"  🎵 Playlist name:      {mood_name}")
    print()
    print("  🧠 Detected Mood:")
    print(f"     • Energy:           {payload['energy_band']}")
    print(f"     • Valence:          {payload['valence_band']}")
    print(f"     • Instrumentalness: {payload['instrumentalness_band']}")
    print(f"     • Genres:           {', '.join(genres)}")
    print()
    print(f"  🔍 Search queries ({len(preview_queries)} total):")
    for i, q in enumerate(preview_queries, 1):
        print(f"     {i}. \"{q}\"")
    print()
    if skip_top_artists:
        print("  👤 Top artists:        SKIPPED (discovery mode)")
    else:
        print("  👤 Top artists used for genre hints:")
        for a in top_artists:
            genres_str = ", ".join(a["genres"][:3]) if a["genres"] else "no genres"
            print(f"     • {a['name']}  ({genres_str})")
        print(f"  🏷️  Genre hint:        {safe_genre or '(none)'}")
    print()
    if wants_twist:
        print(f"  🔀 Twist playlist:     YES — you asked for a surprise!")
        print(f"  🏷️  Twist genre hint:  {twist_genre or '(none)'}")
    else:
        print(f"  🔀 Twist playlist:     No")
    print("=" * 60)

    confirm = input("\n>> Ready to create? Type 'yes' to proceed: ").strip().lower()
    if confirm not in ("yes", "y"):
        log("ABORT", "Playlist creation cancelled by user.")
        exit(0)

    print()

    # --- Main playlist ---
    main_url = create_playlist(
        name=f"Mood DJ: {mood_name}",
        genres=genres,
        valence_band=payload["valence_band"],
        genre_hint=safe_genre if not skip_top_artists else "",
    )
    log("DONE", f"Playlist created -> {main_url}")

    # --- Twist playlist (when user asks for a surprise) ---
    if wants_twist:
        log("TWIST", "🔀 Twist playlist triggered by user request")
        log("TWIST", f"Twist genre hint: '{twist_genre}'")

        twist_url = create_playlist(
            name=f"Mood DJ: {mood_name} 🔀",
            genres=genres,
            valence_band=payload["valence_band"],
            genre_hint=twist_genre,
        )
        log("DONE", f"Twist playlist created -> {twist_url}")