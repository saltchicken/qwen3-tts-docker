import requests
import sounddevice as sd
import struct
import argparse

class TTSClient:
    def __init__(self, server_url="http://localhost:8123/tts"):
        self.server_url = server_url
        self.sample_rate = None
        self.header_size = 0

    def _parse_wav_header(self, header_bytes):
        """
        Parses the WAV header to extract sample rate and finding the start of data.
        Returns (sample_rate, data_start_offset)
        """
        try:
            if header_bytes[0:4] != b'RIFF':
                return None, 0
            
            fmt_loc = header_bytes.find(b'fmt ')
            if fmt_loc == -1: 
                return None, 0
                
            sr_offset = fmt_loc + 12
            sample_rate = struct.unpack('<I', header_bytes[sr_offset:sr_offset+4])[0]
            
            data_loc = header_bytes.find(b'data')
            if data_loc == -1:
                return sample_rate, 44
                
            header_size = data_loc + 8 
            
            return sample_rate, header_size
        except Exception as e:
            print(f"Header parse warning: {e}")
            return None, 0

    def stream_audio(self, text, temperature=0.9):
        payload = {
            "text": text,
            "temperature": temperature
        }


        print(f" >> Sending (Temp: {temperature}): {text[:50]}...", flush=True)

        try:

            with requests.post(self.server_url, json=payload, stream=True, timeout=10) as response:
                if response.status_code != 200:
                    print(f"Server Error: {response.status_code}")
                    return

                output_stream = None
                first_chunk_buffer = b""
                is_header_processed = False

                print(" >> Connected. Waiting for audio...", flush=True)

                for chunk in response.iter_content(chunk_size=1024):
                    if not chunk:
                        continue

                    if not is_header_processed:
                        first_chunk_buffer += chunk
                        # Wait for minimal header size
                        if len(first_chunk_buffer) < 44:
                            continue
                        
                        sr, header_len = self._parse_wav_header(first_chunk_buffer)
                        
                        if not sr:
                            print("Could not detect WAV header. Aborting.")
                            return

                        print(f" >> Stream started. Rate: {sr}Hz", flush=True)
                        
                        output_stream = sd.RawOutputStream(
                            samplerate=sr,
                            channels=1,
                            dtype='int16', 
                            blocksize=1024
                        )
                        output_stream.start()

                        # Write the data part of the buffer (skipping header)
                        output_stream.write(first_chunk_buffer[header_len:])
                        
                        is_header_processed = True
                    else:
                        output_stream.write(chunk)

                if output_stream:
                    print(" >> Playback finished.", flush=True)
                    output_stream.stop()
                    output_stream.close()

        except requests.exceptions.Timeout:
            print("Error: Server connection timed out. The model might be stuck generating.")
        except requests.exceptions.ConnectionError:
            print("Could not connect to server. Is it running?")
        except KeyboardInterrupt:
            print("\nStopped.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Streaming Client")
    parser.add_argument("text", nargs="?", help="Text to speak")
    parser.add_argument("--url", default="http://localhost:8123/tts", help="Server URL")
    parser.add_argument("--temp", type=float, default=0.9, help="Temperature (creativity)")
    
    args = parser.parse_args()

    client = TTSClient(server_url=args.url)

    if args.text:
        client.stream_audio(args.text, args.temp)
    else:
        # Default test text
        long_text = (
            "Here is a more comprehensive test to verify the streaming capabilities of your server. "
            "We are sending a significantly larger block of text to ensure that the sentence splitting logic works seamlessly. "
            "By the time you hear this sentence, the GPU should have already finished processing the beginning."
        )
        client.stream_audio(long_text, args.temp)