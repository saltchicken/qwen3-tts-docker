import requests
import sounddevice as sd
import struct
import argparse
import queue
import threading
import re


class AudioPipeline:
    def __init__(self, server_url, voice=None, temp=0.9):
        self.server_url = server_url
        self.voice = voice
        self.temp = temp
        
        # Queues for the pipeline
        self.sentence_queue = queue.Queue() # Text sentences waiting for TTS
        self.audio_chunk_queue = queue.Queue() # Audio bytes waiting for playback
        
        # Control flags
        self.playback_finished = threading.Event()
        self.tts_processing_finished = threading.Event()
        self.stop_signal = False

        # Audio Config
        self.sample_rate = None
        self.channels = 1
        self.sd_stream = None

    def _parse_wav_header(self, header_bytes):
        """Parses WAV header to get sample rate and data offset."""
        try:
            if len(header_bytes) < 44 or header_bytes[0:4] != b'RIFF':
                return None, 0
            
            fmt_loc = header_bytes.find(b'fmt ')
            if fmt_loc == -1: return None, 0
            
            sr_offset = fmt_loc + 12
            sample_rate = struct.unpack('<I', header_bytes[sr_offset:sr_offset+4])[0]
            
            data_loc = header_bytes.find(b'data')
            if data_loc == -1: return sample_rate, 44
            
            header_size = data_loc + 8 
            return sample_rate, header_size
        except:
            return None, 0

    def tts_worker(self):
        """Thread: Pops sentences, requests TTS, pushes audio chunks."""
        while not self.stop_signal:
            try:
                # Wait for a sentence (timeout allows checking stop_signal)
                text = self.sentence_queue.get(timeout=0.5)
                if text is None: # Sentinel value
                    break
            except queue.Empty:
                continue

            print(f"   [TTS Worker] Processing: '{text[:30]}...'", flush=True)
            
            payload = {
                "text": text,
                "temperature": self.temp,
                "voice": self.voice
            }

            try:

                with requests.post(self.server_url, json=payload, stream=True, timeout=30) as response:
                    if response.status_code != 200:
                        print(f"‼️ Server Error {response.status_code}")
                        continue
                    
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            self.audio_chunk_queue.put(chunk)
            except Exception as e:
                print(f"‼️ TTS Network Error: {e}")

            self.sentence_queue.task_done()
        
        # Signal that no more audio will be produced
        self.tts_processing_finished.set()
        self.audio_chunk_queue.put(None) # Sentinel for player

    def player_worker(self):
        """Thread: Pops audio chunks, plays continuous stream."""
        buffer = b""
        stream_open = False
        
        while not self.stop_signal:
            try:
                chunk = self.audio_chunk_queue.get(timeout=0.5)
                if chunk is None: # Sentinel
                    break
            except queue.Empty:
                continue

            if not stream_open:
                buffer += chunk
                # Need enough bytes for header
                if len(buffer) < 44:
                    continue
                
                # Parse header
                sr, header_len = self._parse_wav_header(buffer)
                if sr:
                    self.sample_rate = sr
                    print(f"   [Player] Stream started at {sr}Hz", flush=True)
                    
                    self.sd_stream = sd.RawOutputStream(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype='int16', 
                        blocksize=2048
                    )
                    self.sd_stream.start()
                    
                    # Write initial data (skipping header)
                    self.sd_stream.write(buffer[header_len:])
                    buffer = b"" # Clear buffer
                    stream_open = True
                else:
                    # If we can't find header yet, keep buffering
                    continue
            else:

                # (because server treats each request as new). We must detect and strip it
                # to avoid loud "pops" or static.
                
                # Simple heuristic: Check if chunk starts with RIFF
                if chunk.startswith(b'RIFF'):
                    _, h_len = self._parse_wav_header(chunk)
                    if h_len > 0:
                        # Strip the header, play the rest
                        self.sd_stream.write(chunk[h_len:])
                    else:
                        self.sd_stream.write(chunk)
                else:
                    self.sd_stream.write(chunk)

            self.audio_chunk_queue.task_done()

        if self.sd_stream:
            self.sd_stream.stop()
            self.sd_stream.close()
        self.playback_finished.set()

    def start(self):
        self.t_tts = threading.Thread(target=self.tts_worker, daemon=True)
        self.t_player = threading.Thread(target=self.player_worker, daemon=True)
        self.t_tts.start()
        self.t_player.start()

    def add_text(self, text):
        self.sentence_queue.put(text)

    def close(self):
        self.sentence_queue.put(None) # Stop TTS
        self.t_tts.join() # Wait for TTS to finish pending
        self.t_player.join() # Wait for player to finish pending

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Streaming Client (Threaded)")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--url", default="http://localhost:8123/tts", help="Server URL")
    parser.add_argument("--temp", type=float, default=0.9, help="Temperature (creativity)")
    parser.add_argument("--voice", type=str, default=None, help="Voice ID (filename in refs/ without extension)")
    
    args = parser.parse_args()


    client = AudioPipeline(server_url=args.url, voice=args.voice, temp=args.temp)
    client.start()

    full_text = ""
    if args.text:
        full_text = args.text
    else:
        # Default test text
        full_text = (
            "Here is a more comprehensive test to verify the streaming capabilities of your server. "
            "We are sending a significantly larger block of text to ensure that the sentence splitting logic works seamlessly. "
            "By the time you hear this sentence, the GPU should have already finished processing the beginning."
        )


    # This mimics how gemini-client.py feeds the pipeline
    print(f" >> Sending text to pipeline...", flush=True)
    
    # Regex to split on punctuation (.!?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    for s in sentences:
        if s.strip():
            client.add_text(s.strip())

    # Wait for completion
    try:
        client.close()
        print(" >> Done.")
    except KeyboardInterrupt:
        print("\nStopped.")
        client.stop_signal = True
        client.close()