import os
import asyncio
import wave
import io
import time

from speech_segmenter import SentenceSegmenter
import discord
from discord.ext import commands
import torch
from faster_whisper import WhisperModel
try:
    import nacl
except ImportError as e:
    raise RuntimeError(
        "PyNaCl library is required for voice support. Install with 'pip install PyNaCl'"
    ) from e

TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN environment variable is not set")

# load faster-whisper model once with Korean support
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel("base", device=device,
                     compute_type="float16" if device == "cuda" else "int8")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)

record_queue = asyncio.Queue()

class VADSink(discord.sinks.Sink):
    def __init__(self, *, loop=None, queue=None):
        super().__init__()
        self.segmenter = SentenceSegmenter()
        self.loop = loop
        self.queue = queue

    def write(self, data, user):
        for segment in self.segmenter.process(data, user):
            self._finish(user, segment)

    def _finish(self, user, audio):
        if not audio:
            return
        # handle both discord.Member objects and plain user IDs
        user_id = getattr(user, "id", user)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(audio)
        # queue the user id instead of the user object for later lookup
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait, (user_id, buffer.getvalue())
        )

def transcribe_bytes(audio_bytes: bytes) -> str:
    """Transcribe audio bytes to Korean text using faster-whisper."""
    segments, _ = model.transcribe(io.BytesIO(audio_bytes), language="ko")
    return "".join(seg.text for seg in segments).strip()

async def transcribe_worker(text_channel):
    while True:
        user_id, audio_bytes = await record_queue.get()
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, transcribe_bytes, audio_bytes)
            user = bot.get_user(user_id)
            name = user.display_name if user else f"User {user_id}"
            await text_channel.send(f"{name}: {text}")
        except Exception as e:
            await text_channel.send(f"Error transcribing for <@{user_id}>: {e}")
        finally:
            record_queue.task_done()

@bot.command(name="join")
async def join(ctx: commands.Context):
    if not ctx.author.voice:
        await ctx.send("You must be in a voice channel")
        return
    channel = ctx.author.voice.channel
    vc = await channel.connect()
    sink = VADSink(loop=bot.loop, queue=record_queue)
    vc.start_recording(sink, finished_callback, ctx.channel)
    bot.loop.create_task(transcribe_worker(ctx.channel))
    await ctx.send("Recording started")

async def finished_callback(sink, channel):
    await record_queue.join()
    await channel.send("Recording finished")

@bot.command(name="leave")
async def leave(ctx: commands.Context):
    if ctx.voice_client:
        ctx.voice_client.stop_recording()
        await ctx.voice_client.disconnect()
        await ctx.send("Left voice channel")
    else:
        await ctx.send("Not in a voice channel")

bot.run(TOKEN)
