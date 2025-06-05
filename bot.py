import os
import asyncio
import io
import wave

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

class StreamingWhisperSink(discord.sinks.Sink):
    """Stream audio to Whisper in ~500ms chunks while also using VAD."""

    def __init__(self, *, loop=None, queue=None):
        super().__init__()
        self.segmenter = SentenceSegmenter()
        self.loop = loop
        self.queue = queue
        self.buffers = {}
        self.chunk_bytes = int(48000 * 0.5 * 2 * 2)  # 500ms of stereo 16bit audio

    def write(self, data, user):
        user_id = getattr(user, "id", user)
        buf = self.buffers.setdefault(user_id, bytearray())
        buf.extend(data)

        while len(buf) >= self.chunk_bytes:
            chunk = bytes(buf[: self.chunk_bytes])
            del buf[: self.chunk_bytes]
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait, (user_id, chunk, False)
            )

        for segment in self.segmenter.process(data, user_id):
            self._finish(user_id, segment)

    def _finish(self, user_id, segment):
        buf = self.buffers.get(user_id, bytearray())
        if buf:
            self.loop.call_soon_threadsafe(
                self.queue.put_nowait, (user_id, bytes(buf), False)
            )
            buf.clear()
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait, (user_id, None, True)
        )

    def cleanup(self):
        for user_id, buf in self.buffers.items():
            if buf:
                self.loop.call_soon_threadsafe(
                    self.queue.put_nowait, (user_id, bytes(buf), True)
                )
        super().cleanup()

def transcribe_stream(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using faster-whisper with low latency."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(48000)
        wf.writeframes(audio_bytes)
    wav_buffer.seek(0)

    segments, _ = model.transcribe(
        wav_buffer, language="ko", beam_size=1, best_of=1
    )
    return "".join(seg.text for seg in segments).strip()

async def transcribe_worker(text_channel):
    messages = {}
    partials = {}
    while True:
        user_id, audio_bytes, finished = await record_queue.get()
        try:
            if audio_bytes:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    None, transcribe_stream, audio_bytes
                )
                if text:
                    partials[user_id] = partials.get(user_id, "") + text

            display = partials.get(user_id, "")
            if finished and display:
                user = bot.get_user(user_id)
                name = user.display_name if user else f"User {user_id}"
                if user_id in messages:
                    await messages[user_id].edit(content=f"{name}: {display}")
                else:
                    await text_channel.send(f"{name}: {display}")
                messages.pop(user_id, None)
                partials[user_id] = ""
            elif not finished and display:
                user = bot.get_user(user_id)
                name = user.display_name if user else f"User {user_id}"
                content = f"{name}: {display}..."
                if user_id in messages:
                    await messages[user_id].edit(content=content)
                else:
                    messages[user_id] = await text_channel.send(content)
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
    sink = StreamingWhisperSink(loop=bot.loop, queue=record_queue)
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
