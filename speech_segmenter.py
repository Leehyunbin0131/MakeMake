import time
import numpy as np
import webrtcvad
import _webrtcvad

class SentenceSegmenter:
    """Accumulate audio and yield segments when silence indicates a sentence end."""
    def __init__(self, *, vad_level=2, sample_rate=48000, frame_ms=30, silence_ms=1000):
        self.vad = webrtcvad.Vad(vad_level)
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.silence_frames = silence_ms // frame_ms
        self.buffers = {}
        self.last_voice = {}

    def add_user(self, user):
        self.buffers[user] = bytearray()
        self.last_voice[user] = time.monotonic()

    def remove_user(self, user):
        self.buffers.pop(user, None)
        self.last_voice.pop(user, None)

    def process(self, data, user):
        if user not in self.buffers:
            self.add_user(user)
        buf = self.buffers[user]
        buf.extend(data)
        output = []
        while len(buf) >= self.frame_bytes:
            frame = bytes(buf[:self.frame_bytes])
            del buf[:self.frame_bytes]
            mono = np.frombuffer(frame, dtype=np.int16)[::2].tobytes()
            try:
                speech = self.vad.is_speech(mono, sample_rate=self.sample_rate)
            except _webrtcvad.Error:
                continue
            now = time.monotonic()
            if speech:
                self.last_voice[user] = now
                output.append(frame)
            elif now - self.last_voice[user] >= (self.silence_frames * self.frame_bytes / (self.sample_rate * 2)) and output:
                # enough silence, sentence finished
                sentence = b''.join(output)
                output.clear()
                yield sentence
        if output:
            # put back partial frames if not finished
            self.buffers[user] = bytearray(output) + buf
            for _ in range(len(output)):
                output.pop()
        else:
            self.buffers[user] = buf
