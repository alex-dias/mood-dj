import os
import json
import random
import datetime
import getpass
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

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
SPOTIPY_REDIRECT_URI = "http://localhost:8888/callback"

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
    now = datetime.datetime.now()
    return now.strftime("%A, %H:%M")

@tool
def get_my_top_artists() -> str:
    results = sp.current_user_top_artists(limit=10, time_range="medium_term")
    artists = [{
        "name": a["name"],
        "id": a["id"],
        "genres": a["genres"]
    } for a in results["items"]]
    return json.dumps(artists)

@tool
def create_playlist(
    name: str,
    seed_type: str,
    seed_value: str,
    energy: float,
    valence: float,
    instrumentalness: float
) -> str:
    log("SPOTIFY", f"Creating playlist '{name}'")
    
    seeds = {}
    if seed_type == "artist":
        seeds["seed_artists"] = [seed_value]
    else:
        seeds["seed_genres"] = [seed_value]

    recs = sp.recommendations(
        limit=20,
        target_energy=energy,
        target_valence=valence,
        target_instrumentalness=instrumentalness,
        **seeds
    )

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
# LLM + PROMPT
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
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

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(
    llm,
    [get_current_context, get_my_top_artists],
    prompt
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=False)

# =========================================================
# RUNTIME
# =========================================================

if __name__ == "__main__":
    log("READY", "🎧 AI DJ is live")
    user_input = input(">> How are you feeling or what do you need? ")

    log("LLM", "Analyzing intent")
    result = agent_executor.invoke({"input": user_input})
    payload = json.loads(result["output"])

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
