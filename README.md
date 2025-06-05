# Discord Voice Recognition Bot

This repository contains an example Discord bot that records users in a voice channel and transcribes their speech using [Py-cord](https://docs.pycord.dev) and [OpenAI Whisper](https://github.com/openai/whisper).

## Features

- Joins a voice channel and records each speaker individually.
- Detects when a user stops talking using voice activity detection (WebRTC VAD).
- Transcribes the recorded audio with the Whisper model asynchronously.
- Sends the transcribed text back to the text channel.

## Setup

Install dependencies with `pip` (Python 3.11 recommended):

```bash
pip install py-cord openai-whisper torch webrtcvad numpy tqdm tiktoken more-itertools numba
```

FFmpeg must also be installed and accessible in `PATH`.

Create a Discord application and bot token, then set the environment variable:

```bash
export DISCORD_TOKEN=YOUR_TOKEN_HERE
```

Run the bot:

```bash
python bot.py
```

Use `!join` in a text channel while connected to a voice channel to start recording, and `!leave` to stop.
