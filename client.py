import requests
import sounddevice as sd
import struct
import argparse
import queue
import threading
import re
import time

class Qwen3TTSClient:
    def __init__(self, server_url="http://localhost:8123/tts", voice=None, temp=0.9):
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
        self.lock = threading.Lock()

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
        while True:
            if self.stop_signal:
                time.sleep(0.1)
                continue
                
            try:
                # Wait for a sentence (timeout allows checking stop_signal regularly)
                text = self.sentence_queue.get(timeout=0.2)
                if text is None: # Sentinel value for complete shutdown
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
                # Use a shorter stream session and check stop_signal per chunk
                with requests.post(self.server_url, json=payload, stream=True, timeout=10) as response:
                    if response.status_code != 200:
                        print(f"‼️ Server Error {response.status_code}")
                        self.sentence_queue.task_done()
                        continue
                    
                    for chunk in response.iter_content(chunk_size=4096):
                        if self.stop_signal:
                            break
                        if chunk:
                            self.audio_chunk_queue.put(chunk)
            except Exception as e:
                if not self.stop_signal:
                    print(f"‼️ TTS Network Error: {e}")

            self.sentence_queue.task_done()

    def player_worker(self):
        """Thread: Pops audio chunks, plays continuous stream."""
        buffer = b""
        stream_open = False
        
        while True:
            if self.stop_signal:
                # Clean up stream if it was open when interrupted
                with self.lock:
                    if self.sd_stream:
                        try:
                            self.sd_stream.abort()
                            self.sd_stream.close()
                        except:
                            pass
                        self.sd_stream = None
                    stream_open = False
                    buffer = b""
                time.sleep(0.1)
                continue

            try:
                chunk = self.audio_chunk_queue.get(timeout=0.2)
                if chunk is None: # Sentinel value for complete shutdown
                    break
            except queue.Empty:
                continue

            with self.lock:
                if self.stop_signal:
                    self.audio_chunk_queue.task_done()
                    continue

                if not stream_open:
                    buffer += chunk
                    if len(buffer) < 44:
                        self.audio_chunk_queue.task_done()
                        continue
                    
                    sr, header_len = self._parse_wav_header(buffer)
                    if sr:
                        self.sample_rate = sr
                        print(f"   [Player] Stream started at {sr}Hz", flush=True)
                        
                        try:
                            self.sd_stream = sd.RawOutputStream(
                                samplerate=self.sample_rate,
                                channels=self.channels,
                                dtype='int16', 
                                blocksize=1024
                            )
                            self.sd_stream.start()
                            self.sd_stream.write(buffer[header_len:])
                            buffer = b""
                            stream_open = True
                        except Exception as e:
                            print(f"‼️ Audio Output Error: {e}")
                    else:
                        self.audio_chunk_queue.task_done()
                        continue
                else:
                    try:
                        if chunk.startswith(b'RIFF'):
                            _, h_len = self._parse_wav_header(chunk)
                            if h_len > 0:
                                self.sd_stream.write(chunk[h_len:])
                            else:
                                self.sd_stream.write(chunk)
                        else:
                            self.sd_stream.write(chunk)
                    except Exception as e:
                        if not self.stop_signal:
                            print(f"‼️ Playback stream write error: {e}")

            self.audio_chunk_queue.task_done()

    def interrupt(self):
        """Instantly terminates current playback and clears all queues."""
        print("\n⚡ Interruption triggered! Cutting audio pipeline immediately...", flush=True)
        
        # 1. Activate stop signal to hold worker actions
        self.stop_signal = True
        
        # 2. Force abort the audio hardware output instantly (clears device buffer)
        with self.lock:
            if self.sd_stream:
                try:
                    self.sd_stream.abort() # Immediate stop, unlike stream.stop()
                    self.sd_stream.close()
                except Exception:
                    pass
                self.sd_stream = None

        # 3. Drain text and audio queues completely
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
            except queue.Empty:
                break
                
        while not self.audio_chunk_queue.empty():
            try:
                self.audio_chunk_queue.get_nowait()
                self.audio_chunk_queue.task_done()
            except queue.Empty:
                break

        # Short cool down to let loops drop out of active HTTP requests/writes
        time.sleep(0.2)
        
        # 4. Clear signal so the engine is ready for new speech input
        self.stop_signal = False
        print(" >> Pipeline reset and ready.", flush=True)

    def start(self):
        """Starts the background worker threads."""
        self.t_tts = threading.Thread(target=self.tts_worker, daemon=True)
        self.t_player = threading.Thread(target=self.player_worker, daemon=True)
        self.t_tts.start()
        self.t_player.start()

    def add_text(self, text):
        """Adds a single raw string to the pipeline."""
        self.sentence_queue.put(text)

    def speak(self, text):
        """Splits block text into sentences and queues them."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            if s.strip():
                self.add_text(s.strip())

    def close(self):
        """Blocks until the pipeline finishes processing all queued text."""
        self.sentence_queue.put(None) 
        self.audio_chunk_queue.put(None)
        if hasattr(self, 't_tts') and self.t_tts.is_alive():
            self.t_tts.join() 
        if hasattr(self, 't_player') and self.t_player.is_alive():
            self.t_player.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Streaming Client (Threaded)")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--url", default="http://localhost:8123/tts", help="Server URL")
    parser.add_argument("--temp", type=float, default=0.9, help="Temperature")
    parser.add_argument("--voice", type=str, default=None, help="Voice ID")
    
    args = parser.parse_args()

    client = Qwen3TTSClient(server_url=args.url, voice=args.voice, temp=args.temp)
    client.start()

    full_text = (
        "Here is a more comprehensive test to verify the streaming capabilities of your server. "
        "We are sending a significantly larger block of text to ensure that the sentence splitting logic works seamlessly. "
        "By the time you hear this sentence, the GPU should have already finished processing the beginning."
    )
    if args.text:
        full_text = args.text

    print(" >> Sending text to pipeline (Press Ctrl+C to test immediate interrupt)...", flush=True)
    client.speak(full_text)

    # Monitor loop allowing user interaction/KeyboardInterrupt
    try:
        while not client.sentence_queue.empty() or not client.audio_chunk_queue.empty():
            time.sleep(0.1)
        
        # Give a moment for final audio block to play through hardware
        time.sleep(1.0)
        client.close()
        print(" >> Done.")
        
    except KeyboardInterrupt:
        # Trigger immediate hardware/queue purge
        client.interrupt()
        
        # Optional: Test sending a brand new phrase right after to verify reusability!
        print("\n >> Verification: Sending a new short sentence to ensure pipeline recovered...")
        client.speak("Pipeline successfully recovered from interrupt.")
        
        # Let it finish playing the recovery message before shutting down the script
        while not client.sentence_queue.empty() or not client.audio_chunk_queue.empty():
            time.sleep(0.1)
        time.sleep(1.0)
        client.close()
