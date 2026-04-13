# Mood DJ 🎧

**Mood DJ** is a personal, AI-powered music curation project that completely rethinks how we generate Spotify playlists. Instead of searching by genre or artist, you simply tell the script how you feel, what you're doing, or the exact vibe you want in natural language. The AI acts as your personal DJ, translating your intent into tailored Spotify playlists.

> **Note:** This is an open-source personal project built for fun and musical exploration. It is **not** a commercial product and does not aim to replace built-in Spotify functionality. Development happens in my free time, so updates will be released as they are done!

---

## 🌟 Capabilities

1. **Natural Language Mood Analysis** 🧠
The system translates casual text input into highly precise Spotify audio metrics, automatically mapping intent to specific energy, valence, and instrumentalness targets. 
2. **Intelligent Genre Matching** 🎸
Rather than relying on static keywords, the AI dynamically selects from a curated master list of over 80+ Spotify-optimized genres to ensure the music accurately matches the desired style.
3. **Multi-Query Synthesis** 🔍
To guarantee playlist variety, the engine executes a barrage of specialized searches behind the scenes, combining pure genres with mood adjectives out of the user's view.
4. **Personalized Taste Injection** 🎧
The system scans the user's Spotify history to identify nuanced sub-genres of their top artists, using those overlaps to subtly flavor the search results.
5. **Discovery Mode** 🚀
When a user explicitly wants to break out of their bubble (e.g., "play something different"), the AI silences top artist data, preventing historical listening habits from polluting a fresh playlist.
6. **The "Twist" Wildcard** 🔀
When users ask for a surprise, the engine spins up an alternate, parallel playlist. It selects a secondary, less-common genre from their listening profile to generate a fun sonic curveball.
7. **Adaptive Focus Sizing** 🎯
The AI scales its search breadth based on user intent. Vague moods trigger a wide net using up to 5 diverse genres, while specific requests remain aggressively focused on just 1 or 2 core genres.
8. **Creative Playlist Naming** ✍️
A secondary AI runs in tandem to act as a copywriter, generating evocative, poetic playlist titles (like "Heartbeat of Joy" or "Electric Chaos") based on mood metrics.
9. **Holistic Pre-Flight Summary** 📊
Before permanently touching your Spotify account, the terminal outputs a beautifully structured summary of the entire AI decision process—displaying target metrics and previewing search queries.
10. **End-to-End Playlist Automation** ⚙️
From input to output, the script securely manages OAuth flows, builds the private playlist entity, dedupes tracks, and auto-populates the final result seamlessly.

---

## 💬 Use Cases & Examples

You interact with Mood DJ just like you are talking to a human DJ:

*   **The Focused Specific:** 
    *   *You say:* `"I want some samba in my life!"`
    *   *AI Action:* Locks onto `["samba", "pagode"]` and creates a high-energy, positive valence playlist titled *Heartbeat of Joy*. 
*   **The Vague Vibe:** 
    *   *You say:* `"Coding late at night"`
    *   *AI Action:* Identifies low energy, neutral valence, and high instrumentalness. Selects `["lo-fi hip hop", "chillhop", "ambient"]`, generating carefully paced, wordless focus tracks.
*   **Discovery Mode:**
    *   *You say:* `"I want something totally different from what I usually listen to"`
    *   *AI Action:* Activates the `skip_top_artists` flag. Pulls completely unrelated genres based on your mood, ignoring the safety net of your listening history.
*   **The Request for Chaos (Twist):** 
    *   *You say:* `"Surprise me with something wild!"`
    *   *AI Action:* Creates an upbeat, multi-genre playlist, plus automatically triggers a **Twist Playlist**. It pulls a bizarre, rarely-listened-to genre from your profile, creating an unpredictable alternative mix.

---

## 🛠️ Setup & Requirements

### 1. Python Environment
This script requires Python 3.10+ and standard dependency management.
```bash
pip install spotipy langchain-anthropic langgraph python-dotenv requests
```

### 2. Spotify Developer Setup (Critical)
To let Mood DJ create playlists on your account, you need a Spotify Developer App.

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and log in.
2. Click **Create App**.
3. **App Name/Description:** (e.g., Mood DJ)
4. **Redirect URI:** Set this exactly to: `http://127.0.0.1:8889/callback`
5. Click **Save**.
6. **IMPORTANT:** Under **User Management / Users and Access**, you MUST add the Spotify email address of the account you plan to use. As of recent Spotify API changes, apps in Development mode can only be used by explicitly whitelisted users.
7. Go to **Settings** in your app to retrieve your `Client ID` and `Client Secret`.

### 3. API Keys (.env)
Create a `.env` file in the root of the project folder:
```env
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret

ANTHROPIC_API_KEY=your_anthropic_claude_api_key

# Optional (for weather-based context injection)
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
```

---

## 🚀 Usage

Run the program from the terminal:

```bash
python dj.py
```

*   On your absolute first run, a browser window will pop up asking you to log into Spotify and authorize the app. This allows the script to create private playlists for you. Standard Spotipy caching is used (`.cache`) so you only have to do this once.
*   Type your prompt, hit Enter, review the plan generated by the AI, and type `yes` to finalize the playlist!

**Optional Flags:**
*   `--debug`: Run `python dj.py --debug` to view the raw LLM outputs and LangGraph agent tool calls in the terminal.