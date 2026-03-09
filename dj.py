import os
import json
import random
import datetime
import getpass
import re
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain / LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# =========================================================
# CONFIGURATION
# =========================================================
# This section defines the parameters that control the 
# musical characteristics (energy, valence, instrumentalness) 
# of the generated Spotify playlists.

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
        "vocal":         {"safe": (0.0, 0.2), "spicy": (0.0, 0.4)},
        "mixed":         {"safe": (0.3, 0.6), "spicy": (0.2, 0.7)},
        "instrumental":  {"safe": (0.85, 0.95), "spicy": (0.65, 1.0)},
    }
}

# =========================================================
# UTILITIES
# =========================================================

def log(step, message):
    """Consistent logging format for the console."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{step}] {message}")

def sample_band(band_name, band_value, profile):
    """Samples a random value within the specified band for Spotify API."""
    low, high = BANDS[band_name][band_value][profile]
    return round(random.uniform(low, high), 2)

def should_create_spicy():
    """Returns True if a spicy playlist should be created."""
    return random.random() < SURPRISE_PROBABILITY

def clean_json_output(text: str) -> str:
    """
    Helper function to clean up LLM output that might be wrapped in Markdown.
    LLMs often output JSON as: ```json\n{...}\n```
    This function extracts just the JSON part to prevent json.loads() from failing.
    """
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

# =========================================================
# AUTHENTICATION
# =========================================================

log("INIT", "Initializing environment variables and Spotify Auth")

# Ensure the Google API Key is set in the environment
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google GenAI API Key: ")

# Spotify API Credentials
SPOTIPY_CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID") or input("Spotify Client ID: ")
SPOTIPY_CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET") or input("Spotify Client Secret: ")
SPOTIPY_REDIRECT_URI = "http://127.0.0.1:8889/callback"

# Initialize Spotipy client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-top-read playlist-modify-public playlist-modify-private"
))

# =========================================================
# LANGCHAIN TOOLS
# =========================================================
# LangChain Agents use tools to interact with the outside world.
# The `@tool` decorator registers a function as a LangChain tool.
# IMPORTANT: Tools MUST have docstrings. The LLM reads the docstring
# to understand what the tool does and when it should use it.

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
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
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
        results = sp.search(q=query, limit=5, type="artist,track")
        output = []
        for artist in results.get("artists", {}).get("items", []):
            output.append({
                "type": "artist",
                "name": artist["name"],
                "id": artist["id"],
                "genres": artist.get("genres", [])
            })
        for track in results.get("tracks", {}).get("items", []):
            output.append({
                "type": "track",
                "name": track["name"],
                "id": track["id"],
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
            "energy": f.get("energy"),
            "valence": f.get("valence"),
            "danceability": f.get("danceability"),
            "tempo": f.get("tempo"),
            "acousticness": f.get("acousticness"),
            "instrumentalness": f.get("instrumentalness")
        })
    except Exception as e:
        return f"Audio features lookup failed: {e}"

# This is a regular helper function (NOT a tool) because the LLM does not
# need to decide when to call it. We always call it ourselves in the main
# runtime to get seed artists for the Spotify Recommendations API.
def get_top_artist_ids(limit=5):
    """Fetches the user's top artist IDs from Spotify to use as seeds."""
    results = sp.current_user_top_artists(limit=limit, time_range="medium_term")
    return [a["id"] for a in results.get("items", []) if a.get("id")]

# NOTE: We do NOT use @tool here because this function is called directly
# by our Python code AFTER the agent finishes. The LLM agent only outputs
# a JSON payload with the mood analysis — the actual playlist creation
# is handled entirely by our code below.
def create_playlist(
    name: str,
    seed_artist_ids: list,
    energy: float,
    valence: float,
    instrumentalness: float
) -> str:
    """Creates a Spotify playlist using the Recommendation API based on the parameters."""
    # Ensure the playlist name is never empty (Spotify rejects empty names)
    if not name or not name.strip():
        name = "AI DJ Mix"
    log("SPOTIFY", f"Creating playlist '{name}'")
    
    # Use artist IDs as seeds; fall back to genre if no artists available
    if seed_artist_ids:
        seeds = {"seed_artists": seed_artist_ids[:2]}  # Spotify allows up to 5 seeds total
    else:
        seeds = {"seed_genres": ["pop"]}

    log("SPOTIFY", f"Seeds: {seeds} | Energy: {energy} | Valence: {valence} | Instr: {instrumentalness}")

    try:
        recs = sp.recommendations(
            limit=20,
            target_energy=energy,
            target_valence=valence,
            target_instrumentalness=instrumentalness,
            **seeds
        )
    except Exception as e:
        log("API_FALLBACK", f"Spotify rejected seeds {seeds}. Error: {e}")
        log("API_FALLBACK", "Retrying with generic seed_genres=['pop']...")
        try:
            recs = sp.recommendations(
                limit=20,
                target_energy=energy,
                target_valence=valence,
                target_instrumentalness=instrumentalness,
                seed_genres=["pop"]
            )
        except Exception as e2:
            log("API_ERROR", f"Fallback also failed: {e2}")
            raise

    tracks = [t["uri"] for t in recs["tracks"]]
    user_id = sp.current_user()["id"]

    playlist = sp.user_playlist_create(
        user=user_id,
        name=name,
        public=False,
        description=f"AI DJ | Energy {energy} | Valence {valence}"
    )

    sp.playlist_add_items(playlist["id"], tracks)
    return playlist["external_urls"]["spotify"]

# =========================================================
# AGENT SETUP
# =========================================================

# 1. Initialize the LLM (Large Language Model)
# We are using Google's Gemini 2.0 Flash Lite via LangChain's integration.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.0 # Zero temperature for fully deterministic output
)

# 2. Define the System Prompt
# The prompt tells the LLM its persona and the rules it must follow.
# In LangGraph, we pass the system prompt directly as a string (no template escaping needed).

SYSTEM_PROMPT = """
You are an Elite AI DJ.

Your task is to map human mood into structured musical intent.

You have access to tools to enrich your analysis:
- get_current_context: returns the current day and time
- get_weather: returns the weather for a city (use it if the user mentions a location)
- search_spotify: searches for artists/tracks by name (use it if the user mentions a specific artist or song)
- get_audio_features: returns audio features of a Spotify track (use it to analyze a track the user references)

After gathering context, you MUST output ONLY a JSON object with these fields:
- mood_name: a short, creative name for the mood
- energy_band: one of [very_low, low, medium, high, very_high]
- valence_band: one of [negative, neutral, positive]
- instrumentalness_band: one of [vocal, mixed, instrumental]

Rules:
- If user mentions focus/study/code/read -> instrumentalness_band MUST be "instrumental"
- Do NOT output numeric values, only the band names above
- Do NOT explain reasoning, output ONLY the JSON object
"""

# 3. Bind Tools and Create the Agent
# In LangGraph, `create_react_agent` replaces the old AgentExecutor + create_tool_calling_agent pattern.
# It creates a ReAct-style agent that can decide which tools to call, observe results, and respond.
# We give the agent all available tools so it can enrich its analysis.
tools = [get_current_context, get_weather, search_spotify, get_audio_features]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT
)

# =========================================================
# MAIN RUNTIME
# =========================================================

if __name__ == "__main__":
    log("READY", "🎧 AI DJ is live")
    user_input = input(">> How are you feeling or what do you need? ")

    log("LLM", "Agent is analyzing intent and invoking tools...")
    
    # Run the LangGraph agent with the user's input
    # LangGraph agents accept a dict with "messages" key containing message objects.
    result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    
    # Debug: print all messages returned by the agent to understand the flow
    print("\n--- AGENT MESSAGE CHAIN ---")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        # msg.content can be a string OR a list of dicts (for tool calls)
        if isinstance(msg.content, str):
            content_preview = msg.content[:100] if msg.content else "(empty)"
        elif isinstance(msg.content, list):
            content_preview = str([x.get("text", "") for x in msg.content if isinstance(x, dict) and "text" in x])[:100] or "(tool calls only)"
        else:
            content_preview = "(unknown format)"
        print(f"  [{i}] {msg_type}: {content_preview}")
    print("---------------------------\n")

    # Extract the final AI message content from the LangGraph response.
    # LangGraph returns a list of messages. We need to find the last AIMessage 
    # that has actual text content (not empty, not just tool calls).
    raw_output = ""
    for msg in reversed(result["messages"]):
        content = msg.content
        # Handle content as a list of dicts (e.g., [{"type": "text", "text": "..."}, ...])
        if isinstance(content, list):
            text_parts = [x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text"]
            content = "\n".join(text_parts)
        if isinstance(content, str) and content.strip():
            raw_output = content
            break

    print("\n--- LLM RAW OUTPUT ---")
    print(raw_output)
    print("----------------------\n")
    clean_json_str = clean_json_output(raw_output)
    
    if not clean_json_str:
        log("ERROR", "Agent did not return any text content. Cannot proceed.")
        exit(1)

    try:
        payload = json.loads(clean_json_str)
    except json.JSONDecodeError as e:
        log("ERROR", f"Failed to parse LLM output as JSON: {e}")
        log("RAW OUTPUT", raw_output)
        exit(1)

    log("INTENT", json.dumps(payload, indent=2))

    # Fetch the user's top artists from Spotify to use as seeds.
    # We do this ourselves instead of relying on the LLM, which avoids
    # hallucinated artist IDs and keeps the LLM's job simple.
    log("SPOTIFY", "Fetching top artists for seeding...")
    top_artist_ids = get_top_artist_ids(limit=5)
    log("SPOTIFY", f"Found {len(top_artist_ids)} top artists")

    # Calculate actual Spotify API parameters using our bands
    energy_safe = sample_band("energy", payload["energy_band"], "safe")
    valence_safe = sample_band("valence", payload["valence_band"], "safe")
    instr_safe = sample_band("instrumentalness", payload["instrumentalness_band"], "safe")

    print(f"\n[SAFE PLAYLIST VALUES] Energy: {energy_safe} | Valence: {valence_safe} | Instrumentalness: {instr_safe}\n")

    # Create the primary 'safe' playlist using the user's top artists as seeds
    mood_name = payload.get("mood_name") or "Mood Mix"
    safe_url = create_playlist(
        name=f"AI DJ: {mood_name}",
        seed_artist_ids=top_artist_ids[:2],
        energy=energy_safe,
        valence=valence_safe,
        instrumentalness=instr_safe
    )

    log("DONE", f"Safe playlist created -> {safe_url}")

    # Occasionally create a 'spicy' alternate playlist
    # The spicy playlist uses different (less-listened) artists as seeds
    if should_create_spicy():
        log("SURPRISE", "🌶️ Spicy playlist triggered")

        energy_spicy = sample_band("energy", payload["energy_band"], "spicy")
        valence_spicy = sample_band("valence", payload["valence_band"], "spicy")
        instr_spicy = sample_band("instrumentalness", payload["instrumentalness_band"], "spicy")

        print(f"\n[SPICY PLAYLIST VALUES] Energy: {energy_spicy} | Valence: {valence_spicy} | Instrumentalness: {instr_spicy}\n")

        # Use the tail end of top artists for variety
        spicy_seeds = top_artist_ids[2:4] if len(top_artist_ids) > 2 else top_artist_ids
        spicy_url = create_playlist(
            name=f"AI DJ: {mood_name} 🌶️",
            seed_artist_ids=spicy_seeds,
            energy=energy_spicy,
            valence=valence_spicy,
            instrumentalness=instr_spicy
        )

        log("DONE", f"Spicy playlist created -> {spicy_url}")
    else:
        log("SURPRISE", "No spicy playlist this time")
