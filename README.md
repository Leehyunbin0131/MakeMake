# Discord Voice Recognition Bot

This repository contains an example Discord bot that records users in a voice channel and transcribes their speech using [Py-cord](https://docs.pycord.dev) and [OpenAI Whisper](https://github.com/openai/whisper).

## Features

- Joins a voice channel and records each speaker individually.
- Detects when a user stops talking using voice activity detection (WebRTC VAD).
- Transcribes the recorded audio with the Whisper model asynchronously.
- Sends the transcribed text back to the text channel.

## Setup

Install dependencies with `pip` (Python 3.10 or higher recommended):

```bash
pip install py-cord openai-whisper torch webrtcvad numpy tqdm tiktoken more-itertools numba PyNaCl RealtimeSTT
```

FFmpeg must also be installed and accessible in `PATH`.
The bot relies on the [RealtimeSTT](https://pypi.org/project/RealtimeSTT/) package
for on-the-fly transcription, so ensure it is installed as shown above.

Create a Discord application and bot token, then set the environment variable:

```bash
export DISCORD_TOKEN=YOUR_TOKEN_HERE
```

Run the bot:

```bash
python bot.py
```

On Windows, the script must be executed directly (not imported) to avoid
multiprocessing startup errors.

The recorder is initialized to use the CPU with float32 precision to avoid
`ctranslate2` compute type warnings.

Use `!join` in a text channel while connected to a voice channel to start recording, and `!leave` to stop.
