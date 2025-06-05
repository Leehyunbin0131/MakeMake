import os
import asyncio
import wave
import time
from collections import defaultdict

import numpy as np
import webrtcvad
import discord
from discord.ext import commands
import whisper
try:
    import nacl
except ImportError as e:
    raise RuntimeError(
        "PyNaCl library is required for voice support. Install with 'pip install PyNaCl'"
    ) from e

TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN environment variable is not set")

# load whisper model once
model = whisper.load_model("base")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)

record_queue = asyncio.Queue()

class VADSink(discord.sinks.Sink):
    def __init__(self, *, silence_timeout=2.0, loop=None, queue=None):
        super().__init__()
        self.vad = webrtcvad.Vad(2)
        self.silence_timeout = silence_timeout
        self.buffers = defaultdict(bytearray)
        self.last_voice = defaultdict(lambda: time.monotonic())
        self.loop = loop
        self.queue = queue

    def write(self, data, user):
        buf = self.buffers[user]
        buf.extend(data)

        mono = np.frombuffer(data, dtype=np.int16)[::2].tobytes()
        speech = self.vad.is_speech(mono, sample_rate=48000)
        now = time.monotonic()
        if speech:
            self.last_voice[user] = now
        elif now - self.last_voice[user] >= self.silence_timeout and len(buf) > 0:
            self._finish(user)

    def _finish(self, user):
        buf = self.buffers.pop(user, None)
        if not buf:
            return
        # use the user's id in the filename to avoid illegal characters
        path = f"record_{user.id}_{int(time.time()*1000)}.wav"
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(bytes(buf))
        # queue the user id instead of the user object for later lookup
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (user.id, path))

async def transcribe_worker(text_channel):
    while True:
        user_id, path = await record_queue.get()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, model.transcribe, path)
            user = bot.get_user(user_id)
            name = user.display_name if user else f"User {user_id}"
            await text_channel.send(f"{name}: {result['text']}")
        except Exception as e:
            await text_channel.send(f"Error transcribing for <@{user_id}>: {e}")
        finally:
            if os.path.exists(path):
                os.remove(path)
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
