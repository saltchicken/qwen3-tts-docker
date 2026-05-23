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
        
        # 🔥 NEW: Session tracking to invalidate delayed network responses
        self.session_id = 0 

        # Audio Config
        self.sample_rate = None
        self.channels = 1
        self.sd_stream = None

    def _parse_wav_header(self, header_bytes):
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
        while True:
            if self.stop_signal:
                time.sleep(0.1)
                continue
                
            # 🔥 Capture the session ID right before we start processing
            current_session = self.session_id

            try:
                text = self.sentence_queue.get(timeout=0.2)
                if text is None: break
            except queue.Empty:
                continue

            if current_session != self.session_id:
                self.sentence_queue.task_done()
                continue


            # print(f"   [TTS Worker] Processing: '{text[:30]}...'", flush=True)
            
            payload = {
                "text": text,
                "temperature": self.temp,
                "voice": self.voice
            }

            try:
                with requests.post(self.server_url, json=payload, stream=True, timeout=60) as response:
                    # 🔥 If an interrupt happened while waiting for the server to reply, drop it instantly!
                    if self.session_id != current_session:
                        self.sentence_queue.task_done()
                        continue

                    if response.status_code != 200:
                        print(f"‼️ Server Error {response.status_code}")
                        self.sentence_queue.task_done()
                        continue
                    
                    for chunk in response.iter_content(chunk_size=4096):
                        # 🔥 Also break mid-stream if the session ID changes
                        if self.session_id != current_session or self.stop_signal:
                            break
                        if chunk:
                            # Pass the session ID along with the chunk
                            self.audio_chunk_queue.put((chunk, current_session))
            except Exception as e:
                if not self.stop_signal and self.session_id == current_session:
                    print(f"‼️ TTS Network Error: {e}")

            self.sentence_queue.task_done()

    def player_worker(self):
        buffer = b""
        stream_open = False
        
        while True:
            if self.stop_signal:
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
                # 🔥 Unpack the chunk AND the session ID it belongs to
                item = self.audio_chunk_queue.get(timeout=0.2)
                if item is None: break
                chunk, chunk_session = item
            except queue.Empty:
                continue

            with self.lock:
                # 🔥 If this chunk belongs to an old session, throw it out immediately!
                if self.stop_signal or chunk_session != self.session_id:
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
                        # print(f"   [Player] Stream started at {sr}Hz", flush=True)
                        
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
        # print("\n⚡ Interruption triggered! Cutting audio pipeline immediately...", flush=True)
        
        self.stop_signal = True
        
        # 🔥 NEW: Increment session ID to permanently invalidate all delayed network responses
        self.session_id += 1 
        
        with self.lock:
            if self.sd_stream:
                try:
                    self.sd_stream.abort()
                    self.sd_stream.close()
                except Exception:
                    pass
                self.sd_stream = None

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

        time.sleep(0.2)
        self.stop_signal = False
        print(" >> Pipeline reset and ready.", flush=True)

    def start(self):
        self.t_tts = threading.Thread(target=self.tts_worker, daemon=True)
        self.t_player = threading.Thread(target=self.player_worker, daemon=True)
        self.t_tts.start()
        self.t_player.start()

    def add_text(self, text):
        self.sentence_queue.put(text)

    def speak(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for s in sentences:
            if s.strip():
                self.add_text(s.strip())

    def close(self):
        self.sentence_queue.put(None) 
        self.audio_chunk_queue.put(None)
        if hasattr(self, 't_tts') and self.t_tts.is_alive():
            self.t_tts.join() 
        if hasattr(self, 't_player') and self.t_player.is_alive():
            self.t_player.join()
