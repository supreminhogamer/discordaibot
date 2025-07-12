# Discord AI Chatbot

A friendly and modular Discord chatbot powered by Google Gemini. It features a persistent memory system using SQLite and a fully customizable personality.

## Features

-   Conversational AI using Google Gemini.
-   Context-aware conversations grouped into topics.
-   Persistent memory stored in a local SQLite database.
-   Fully customizable personality via an external `prompt.md` file.
-   Secure configuration using environment variables.
-   Self-Bot configuration, uses a user account instead of a bot one.

## Setup Instructions

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/supreminhogamer/discordaibot.git
    cd discordaibot
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    -   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    -   Open the `.env` file and fill in your actual credentials (Discord Token, Gemini API Key, etc.).

5.  **Customize the Personality (Optional)**
    -   Edit the `prompt.md` file to change the bot's behavior, tone, and style.

6.  **Run the Bot**
    ```bash
    python bot.py
    ```

## To-Dos
- Multi-Channel Support
- Google Search Support
- Graphical User Interface 
