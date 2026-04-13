import os
import json
import re
import datetime
import getpass

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# =========================================================
# CONFIG
# =========================================================

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube"]
CLIENT_SECRET_FILE = os.path.join(os.path.dirname(__file__), "client_secret.json")
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "yt_token.pickle")
MAX_SONGS = 15

# =========================================================
# LOGGING
# =========================================================

def log(step, message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{step}] {message}")

# =========================================================
# YOUTUBE AUTH
# =========================================================

def get_youtube_client():
    """Authenticate with YouTube via OAuth 2.0 and return a service client."""
    credentials = None

    # Load cached token
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            credentials = pickle.load(token)

    # Refresh or create credentials
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            log("AUTH", "Refreshing YouTube token")
            credentials.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRET_FILE):
                log("ERROR", f"Missing {CLIENT_SECRET_FILE}. Download it from Google Cloud Console.")
                log("ERROR", "Go to: https://console.cloud.google.com/apis/credentials")
                log("ERROR", "Create an OAuth 2.0 Client ID (Desktop app), download the JSON,")
                log("ERROR", f"and save it as '{CLIENT_SECRET_FILE}'")
                exit(1)

            log("AUTH", "Opening browser for YouTube authentication...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, YOUTUBE_SCOPES)
            credentials = flow.run_local_server(port=8890)

        # Cache the token
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(credentials, token)

    log("AUTH", "YouTube authenticated ✓")
    return build("youtube", "v3", credentials=credentials)

# =========================================================
# INIT
# =========================================================

log("INIT", "Initializing YouTube DJ agent")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google API Key: ")

youtube = get_youtube_client()

# =========================================================
# YOUTUBE HELPERS
# =========================================================

def search_youtube_video(query: str) -> dict | None:
    """Search YouTube for a single video matching the query. Returns {id, title, url} or None."""
    try:
        response = youtube.search().list(
            q=query,
            part="snippet",
            maxResults=1,
            type="video",
            videoCategoryId="10"  # Music category
        ).execute()

        items = response.get("items", [])
        if not items:
            return None

        video = items[0]
        video_id = video["id"]["videoId"]
        return {
            "id": video_id,
            "title": video["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }
    except Exception as e:
        log("WARNING", f"YouTube search failed for '{query}': {e}")
        return None


def create_yt_playlist(name: str, description: str, video_ids: list[str]) -> str:
    """Create a YouTube playlist and add videos to it. Returns the playlist URL."""
    log("YOUTUBE", f"Creating playlist '{name}' with {len(video_ids)} videos")

    # Create the playlist
    playlist_response = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": name,
                "description": description,
            },
            "status": {
                "privacyStatus": "private"
            }
        }
    ).execute()

    playlist_id = playlist_response["id"]
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"

    # Add each video to the playlist
    for i, video_id in enumerate(video_ids):
        try:
            youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id
                        }
                    }
                }
            ).execute()
            log("YOUTUBE", f"  Added video {i+1}/{len(video_ids)}: {video_id}")
        except Exception as e:
            log("WARNING", f"  Failed to add video {video_id}: {e}")

    return playlist_url

# =========================================================
# TOOLS (for LangChain agent)
# =========================================================

@tool
def get_current_context() -> str:
    """Get the current day of the week and time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %H:%M")

# =========================================================
# LLM + PROMPT
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.7,
    max_output_tokens=2048,
    timeout=30,
)

SYSTEM_PROMPT = """
You are an Elite AI DJ.

Your task is to take a user's mood description and suggest songs that perfectly match it.

You MUST output ONLY a JSON object (no markdown fences, no explanation) with:
- "mood_name": a short creative name for the mood (e.g. "Midnight Drive", "Sunday Sunrise")
- "songs": a list of 10 to 15 songs, each as {"artist": "<artist>", "title": "<song title>"}

Rules:
- Pick songs from diverse artists (avoid repeating the same artist)
- Include a mix of well-known and slightly less mainstream tracks
- All songs must match the mood described by the user
- Consider the time of day and day of the week for context
- Focus on REAL songs that actually exist
- If user mentions focus/study/code/read → prioritize instrumental or ambient tracks
- Do NOT output anything other than the JSON object
- Do NOT explain your reasoning
"""

agent_executor = create_react_agent(
    llm,
    tools=[get_current_context],
    prompt=SYSTEM_PROMPT,
)

# =========================================================
# RUNTIME
# =========================================================

if __name__ == "__main__":
    log("READY", "🎧 YouTube AI DJ is live")
    user_input = input(">> How are you feeling or what do you need? ")

    log("LLM", "Analyzing mood and selecting songs...")
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

    mood_name = payload.get("mood_name", "AI DJ Mix")
    songs = payload.get("songs", [])

    if not songs:
        log("ERROR", "No songs returned by the LLM")
        exit(1)

    log("SEARCH", f"Searching YouTube for {len(songs)} songs...")

    # Search YouTube for each song
    found_videos = []
    for song in songs:
        query = f"{song['artist']} - {song['title']} official audio"
        log("SEARCH", f"  Searching: {song['artist']} - {song['title']}")
        video = search_youtube_video(query)
        if video:
            found_videos.append(video)
            log("SEARCH", f"    ✓ Found: {video['title']}")
        else:
            log("SEARCH", f"    ✗ Not found")

    if not found_videos:
        log("ERROR", "Could not find any songs on YouTube")
        exit(1)

    log("SEARCH", f"Found {len(found_videos)}/{len(songs)} songs on YouTube")

    # Create YouTube playlist
    video_ids = [v["id"] for v in found_videos]
    playlist_url = create_yt_playlist(
        name=f"AI DJ: {mood_name}",
        description=f"AI DJ playlist for mood: {mood_name} | Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        video_ids=video_ids
    )

    log("DONE", f"🎵 Playlist created → {playlist_url}")
    print(f"\n{'='*60}")
    print(f"  🎧  Your YouTube playlist is ready!")
    print(f"  🎵  Mood: {mood_name}")
    print(f"  📀  {len(found_videos)} tracks")
    print(f"  🔗  {playlist_url}")
    print(f"{'='*60}\n")
