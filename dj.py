import os
import json
import random
import datetime
import getpass
import re
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
def get_my_top_artists() -> str:
    """
    Retrieves the current user's top artists from Spotify.
    Useful for selecting artists that the user already likes.
    Returns a JSON string containing artist names, IDs, and genres.
    """
    results = sp.current_user_top_artists(limit=10, time_range="medium_term")
    artists = [{
        "name": a.get("name"),
        "id": a.get("id"),
        "genres": a.get("genres", [])
    } for a in results.get("items", []) if a.get("genres")]
    return json.dumps(artists)

# NOTE: We do NOT use @tool here because this function is called directly
# by our Python code AFTER the agent finishes. The LLM agent only outputs
# a JSON payload with the mood analysis — the actual playlist creation
# is handled entirely by our code below.
def create_playlist(
    name: str,
    seed_type: str,
    seed_value: str,
    energy: float,
    valence: float,
    instrumentalness: float
) -> str:
    """Creates a Spotify playlist using the Recommendation API based on the parameters."""
    # Ensure the playlist name is never empty (Spotify rejects empty names)
    if not name or not name.strip():
        name = "AI DJ Mix"
    log("SPOTIFY", f"Creating playlist '{name}'")
    
    seeds = {}
    if seed_type == "artist" and seed_value:
        seeds["seed_artists"] = [seed_value]
    elif seed_type == "genre" and seed_value:
        seeds["seed_genres"] = [seed_value]
    else:
        # Fallback to prevent an empty seeds dictionary which crashes the Spotify API
        seeds["seed_genres"] = ["pop"]

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

You MUST output JSON with:
- mood_name (short, creative)
- energy_band: one of [very_low, low, medium, high, very_high]
- valence_band: one of [negative, neutral, positive]
- instrumentalness_band: one of [vocal, mixed, instrumental]

You MUST also choose TWO seeds:
- safe_seed: aligned with user's listening history
- spicy_seed: exploratory but mood-compatible

Each seed must be formatted as:
{
  "type": "artist" | "genre",
  "value": "<MUST_BE_EXACT_SPOTIFY_ID_OR_ALLOWED_GENRE>"
}

Rules:
- If type is "artist", you MUST provide the EXACT Spotify ID (e.g., "4q3ewBCX7sLwd24euuV69X"), NOT the artist's name!
- If type is "genre", use a simple, broad genre (e.g., "pop", "rock", "hip-hop", "classical", "jazz").
- If user mentions focus/study/code/read -> instrumentalness_band MUST be "instrumental"
- Safe seed prioritizes taste match
- Spicy seed prioritizes novelty without breaking mood
- Do NOT output numeric values
- Do NOT explain reasoning
"""

# 3. Bind Tools and Create the Agent
# In LangGraph, `create_react_agent` replaces the old AgentExecutor + create_tool_calling_agent pattern.
# It creates a ReAct-style agent that can decide which tools to call, observe results, and respond.
tools = [get_current_context, get_my_top_artists]

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

    # Calculate actual Spotify API parameters using our bands
    energy_safe = sample_band("energy", payload["energy_band"], "safe")
    valence_safe = sample_band("valence", payload["valence_band"], "safe")
    instr_safe = sample_band("instrumentalness", payload["instrumentalness_band"], "safe")

    print(f"\n[SAFE PLAYLIST VALUES] Energy: {energy_safe} | Valence: {valence_safe} | Instrumentalness: {instr_safe}\n")

    # Create the primary 'safe' playlist
    mood_name = payload.get("mood_name") or "Mood Mix"
    safe_seed = payload.get("safe_seed", {})
    safe_url = create_playlist(
        name=f"AI DJ: {mood_name}",
        seed_type=safe_seed.get("type", "genre"),
        seed_value=safe_seed.get("value", "pop"),
        energy=energy_safe,
        valence=valence_safe,
        instrumentalness=instr_safe
    )

    log("DONE", f"Safe playlist created → {safe_url}")

    # Occasionally create a 'spicy' alternate playlist
    if should_create_spicy():
        log("SURPRISE", "🌶️ Spicy playlist triggered")

        energy_spicy = sample_band("energy", payload["energy_band"], "spicy")
        valence_spicy = sample_band("valence", payload["valence_band"], "spicy")
        instr_spicy = sample_band("instrumentalness", payload["instrumentalness_band"], "spicy")

        print(f"\n[SPICY PLAYLIST VALUES] Energy: {energy_spicy} | Valence: {valence_spicy} | Instrumentalness: {instr_spicy}\n")

        spicy_seed = payload.get("spicy_seed", {})
        spicy_url = create_playlist(
            name=f"AI DJ: {mood_name} 🌶️",
            seed_type=spicy_seed.get("type", "genre"),
            seed_value=spicy_seed.get("value", "pop"),
            energy=energy_spicy,
            valence=valence_spicy,
            instrumentalness=instr_spicy
        )

        log("DONE", f"Spicy playlist created → {spicy_url}")
    else:
        log("SURPRISE", "No spicy playlist this time")
