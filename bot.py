import os
import re
import sqlite3
import logging
from datetime import datetime, timedelta, timezone

import discord
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

# --- Setup Logging ---
# A simple logger for better tracking of events and errors.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Load Configuration ---
# Load environment variables from the .env file.
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", 0))
BOT_NAME = os.getenv("BOT_NAME", "AI Bot")
PROMPT_FILENAME = os.getenv("SYSTEM_PROMPT_FILE", "prompt.md")

# Validate that essential configurations are loaded.
if not all([TOKEN, GEMINI_API_KEY, CHANNEL_ID]):
    logging.critical("FATAL ERROR: Essential environment variables (DISCORD_TOKEN, GEMINI_API_KEY, CHANNEL_ID) are missing.")
    exit()

# Try to load the System Prompt from the specified file.
try:
    with open(PROMPT_FILENAME, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
    logging.info(f"Successfully loaded personality from '{PROMPT_FILENAME}'.")
except FileNotFoundError:
    logging.error(f"Personality file '{PROMPT_FILENAME}' not found. Using a default prompt.")
    SYSTEM_PROMPT = "You are a helpful assistant."

# --- Global Constants ---
DB_FILE = "bot_memory.db"
SIMILARITY_THRESHOLD = 0.75
TOPIC_EXPIRATION_MINUTES = 30


class MemoryManager:
    """Handles all interactions with the SQLite database for conversation memory."""

    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file)
            self._create_tables()
        except sqlite3.Error as e:
            logging.critical(f"Database connection error: {e}")
            exit()

    def _create_tables(self):
        """Creates the necessary database tables if they don't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    topic_id INTEGER PRIMARY KEY,
                    channel_id INTEGER NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id INTEGER NOT NULL,
                    author TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    embedding BLOB,
                    FOREIGN KEY(topic_id) REFERENCES topics(topic_id) ON DELETE CASCADE
                )
            """)

    def get_active_topics(self, channel_id: int):
        """Retrieves all active (non-expired) topics for a given channel."""
        self._archive_old_topics(channel_id)
        with self.conn:
            cursor = self.conn.execute("SELECT topic_id FROM topics WHERE channel_id = ?", (channel_id,))
            return [self.get_topic_by_id(row[0]) for row in cursor.fetchall()]

    def get_topic_by_id(self, topic_id: int):
        """Retrieves all messages for a specific topic."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT author, text, timestamp, embedding FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (topic_id,)
            )
            messages = []
            for row in cursor.fetchall():
                embedding = np.frombuffer(row[3], dtype=np.float32) if row[3] else None
                messages.append({"author": row[0], "text": row[1], "timestamp": row[2], "embedding": embedding})
            return {"id": topic_id, "messages": messages}

    def add_message(self, topic_id: int, author: str, text: str, timestamp: datetime, embedding: list = None):
        """Adds a message to a topic and updates the topic's last_updated timestamp."""
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes() if embedding is not None else None
        timestamp_iso = timestamp.isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (topic_id, author, text, timestamp, embedding) VALUES (?, ?, ?, ?, ?)",
                (topic_id, author, text, timestamp_iso, embedding_blob)
            )
            self.conn.execute("UPDATE topics SET last_updated = ? WHERE topic_id = ?", (timestamp_iso, topic_id))

    def create_new_topic(self, channel_id: int, first_message: dict):
        """Creates a new topic with its first message."""
        timestamp = first_message["timestamp"]
        topic_id = int(timestamp.timestamp())
        with self.conn:
            self.conn.execute("INSERT INTO topics (topic_id, channel_id, last_updated) VALUES (?, ?, ?)",
                              (topic_id, channel_id, timestamp.isoformat()))
        self.add_message(topic_id, **first_message)
        logging.info(f"Created new topic with ID: {topic_id}")
        return self.get_topic_by_id(topic_id)

    def _archive_old_topics(self, channel_id: int):
        """Deletes topics that have been inactive for too long."""
        expiration_time = datetime.now(timezone.utc) - timedelta(minutes=TOPIC_EXPIRATION_MINUTES)
        with self.conn:
            cursor = self.conn.execute("DELETE FROM topics WHERE channel_id = ? AND last_updated < ?",
                                       (channel_id, expiration_time.isoformat()))
            if cursor.rowcount > 0:
                logging.info(f"Archived {cursor.rowcount} old topic(s).")

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")


class ChatBot:
    """The main ChatBot class, handling Discord events and AI interactions."""

    def __init__(self, name, token, gemini_key):
        self.client = discord.Client()
        self.name = name
        self.token = token
        self.memory = MemoryManager(DB_FILE)

        genai.configure(api_key=gemini_key)
        self.gen_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT.format(bot_name=self.name)
        )
        self.embedding_model = "models/embedding-001"

        # Register events
        self.client.event(self.on_ready)
        self.client.event(self.on_message)

    async def _get_embedding(self, text: str):
        """Generates embedding for a given text."""
        try:
            return genai.embed_content(model=self.embedding_model, content=text)["embedding"]
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return None

    def _calculate_similarity(self, emb1, emb2):
        """Calculates cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None: return 0.0
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def _get_relevant_topic(self, channel_id: int, new_message: dict):
        """Finds the most relevant existing topic or creates a new one."""
        active_topics = self.memory.get_active_topics(channel_id)
        if not active_topics:
            return self.memory.create_new_topic(channel_id, new_message)

        new_msg_emb = new_message["embedding"]
        best_match = None
        highest_similarity = -1

        for topic in active_topics:
            if topic["messages"]:
                # Compare with the last message's embedding in the topic
                last_msg_emb = topic["messages"][-1].get("embedding")
                similarity = self._calculate_similarity(new_msg_emb, last_msg_emb)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = topic
        
        if highest_similarity >= SIMILARITY_THRESHOLD and best_match:
            logging.info(f"Found matching topic {best_match['id']} with similarity {highest_similarity:.2f}")
            return best_match
        else:
            logging.info(f"No suitable topic found (max similarity: {highest_similarity:.2f}). Creating new topic.")
            return self.memory.create_new_topic(channel_id, new_message)

    async def _generate_response(self, topic: dict):
        """Generates a response from the AI based on the conversation history."""
        history = []
        for msg in topic["messages"]:
            role = "user" if msg["author"] != self.name else "model"
            history.append({'role': role, 'parts': [msg['text']]})

        try:
            response = await self.gen_model.generate_content_async(history)
            return response.text.strip()[:2000] # Truncate to Discord's character limit
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            return "Sorry, I encountered an issue while thinking. Could you try again?"

    async def on_ready(self):
        """Called when the bot is connected and ready."""
        logging.info(f"Logged in as {self.client.user}. Bot Name: {self.name}")
        logging.info(f"Operating in Channel ID: {CHANNEL_ID}")

    async def on_message(self, message: discord.Message):
        """Called on every new message in a channel the bot can see."""
        # --- Basic message validation ---
        if message.author.bot or message.channel.id != CHANNEL_ID:
            return

        is_reply_to_me = message.reference and message.reference.resolved and message.reference.resolved.author == self.client.user
        if not (self.client.user.mentioned_in(message) or is_reply_to_me):
            return

        cleaned_text = re.sub(r'<@!?(\d+)>', '', message.content).strip()
        if not cleaned_text:
            return

        # --- Process message and generate response ---
        async with message.channel.typing():
            now = datetime.now(timezone.utc)
            embedding = await self._get_embedding(cleaned_text)
            
            new_message_entry = {
                "author": message.author.display_name,
                "text": cleaned_text,
                "timestamp": now,
                "embedding": embedding
            }

            # Find or create a conversation topic
            topic = self._get_relevant_topic(message.channel.id, new_message_entry)
            
            # If the topic already existed, add the new message to it
            # (If it's new, the first message is already added by `create_new_topic`)
            if len(topic["messages"]) > 1:
                self.memory.add_message(topic['id'], **new_message_entry)
                topic["messages"].append(new_message_entry)

            # Generate the AI's response
            response_text = await self._generate_response(topic)

        # --- Send response and save it to memory ---
        if response_text:
            response_message = await message.reply(response_text)
            
            # Save the bot's own response to memory to maintain context
            response_embedding = await self._get_embedding(response_text)
            self.memory.add_message(
                topic_id=topic['id'],
                author=self.name,
                text=response_text,
                timestamp=datetime.now(timezone.utc),
                embedding=response_embedding
            )

    def run(self):
        """Starts the bot."""
        try:
            self.client.run(self.token)
        except discord.errors.LoginFailure:
            logging.critical("Login failed: Invalid Discord Token.")
        except Exception as e:
            logging.critical(f"An error occurred while running the bot: {e}")
        finally:
            self.memory.close()


if __name__ == "__main__":
    bot = ChatBot(name=BOT_NAME, token=TOKEN, gemini_key=GEMINI_API_KEY)
    bot.run()