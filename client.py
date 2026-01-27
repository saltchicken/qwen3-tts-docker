import requests
import sounddevice as sd
import struct
import sys

# ‼️ Configuration
SERVER_URL = "http://localhost:8123/tts"

def parse_wav_header(header_bytes):
    """
    Parses the WAV header to extract sample rate and finding the start of data.
    Returns (sample_rate, data_start_offset)
    """
    try:
        # Check for 'RIFF'
        if header_bytes[0:4] != b'RIFF':
            return None, 0
        
        # Find 'fmt ' chunk to get sample rate
        fmt_loc = header_bytes.find(b'fmt ')
        if fmt_loc == -1: 
            return None, 0
            
        # Sample rate is at offset 12 from 'fmt ' (fmt_id + size + audio_format + num_channels)
        sr_offset = fmt_loc + 12
        # Unpack Little Endian unsigned int
        sample_rate = struct.unpack('<I', header_bytes[sr_offset:sr_offset+4])[0]
        
        # Find 'data' chunk to know where actual audio begins
        data_loc = header_bytes.find(b'data')
        if data_loc == -1:
            # Fallback: standard 44 byte header if 'data' tag missing (unlikely in valid wav)
            return sample_rate, 44
            
        # 'data' tag is 4 bytes, followed by 4 bytes size, then data
        header_size = data_loc + 8 
        
        return sample_rate, header_size
    except Exception as e:
        print(f"Header parse warning: {e}")
        return None, 0

def stream_audio(text, speaker="john_e"):
    payload = {
        "text": text,
        "language": "English",
        "speaker": speaker,
    }

    print(f" >> Sending: {text[:50]}...")

    try:
        # ‼️ stream=True is critical here
        with requests.post(SERVER_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Server Error: {response.status_code}")
                return

            output_stream = None
            first_chunk_buffer = b""
            is_header_processed = False

            # Read from network in small chunks
            for chunk in response.iter_content(chunk_size=1024):
                if not chunk:
                    continue

                if not is_header_processed:
                    # Accumulate bytes until we have enough to read the header (44 is standard)
                    first_chunk_buffer += chunk
                    if len(first_chunk_buffer) < 44:
                        continue
                    
                    # Parse header
                    sr, header_len = parse_wav_header(first_chunk_buffer)
                    
                    if not sr:
                        print("Could not detect WAV header. Aborting.")
                        return

                    print(f" >> Stream started. Rate: {sr}Hz")
                    
                    # Initialize the audio device (Raw output avoids overhead)
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
                    # Write subsequent chunks directly to audio device
                    output_stream.write(chunk)

            if output_stream:
                print(" >> Playback finished.")
                output_stream.stop()
                output_stream.close()

    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Is it running?")
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test text
    long_text = (
        "Here is a more comprehensive test to verify the streaming capabilities of your server. "
        "We are sending a significantly larger block of text to ensure that the sentence splitting logic works seamlessly across multiple boundaries. "
        "Does it handle questions correctly? It certainly should! "
        "Furthermore, this paragraph includes pauses, longer complex sentences, and standard declarative statements to fully exercise the model's intonation. "
        "By the time you hear this sentence, the GPU should have already finished processing the beginning and moved on to the end, proving that the latency is minimal. "
        "Enjoy the continuous stream of generated speech."
    )
    
    stream_audio(long_text)
