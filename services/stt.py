import os
import queue
import threading
import time
from google.cloud import texttospeech  # ensure package presence
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core import exceptions as gcp_exceptions
from ..sockets.socketio import socketio


stt_client = SpeechClient()


class StreamingAudioProcessor:
    def __init__(self, session_id, on_transcript_callback):
        self.session_id = session_id
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.is_listening = False
        self.stream_thread = None
        self.on_transcript_callback = on_transcript_callback
        self.current_transcript = ""
        self.last_final_time = None
        self.SILENCE_THRESHOLD = 3
        self.restart_lock = threading.Lock()
        self.restart_count = 0
        self.max_restarts = 10  # Reduced from 300 to prevent infinite loops
        self.last_restart_time = 0
        self.restart_cooldown = 2.0  # Minimum seconds between restarts
        self.active_stream = None  # Track active stream for cleanup

        self.rate = 16000
        self.chunk_size = int(self.rate / 10)
        self.language_code = "en-US"
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.last_audio_time = time.time()
        self.recognizer = f"projects/{self.project_id}/locations/global/recognizers/_"
        
        # Configure recognition with optimized settings for accuracy
        self.config = cloud_speech.RecognitionConfig(
            language_codes=[self.language_code],
            model="latest_long",  # Best model for long-form speech
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,
                enable_spoken_punctuation=False,
                enable_spoken_emojis=False,
            ),
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.rate,
                audio_channel_count=1,
            ),
        )
        self.streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=self.config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True
            ),
        )

    def start(self):
        with self.restart_lock:
            if self.is_running:
                return
            self.is_running = True
            self.is_listening = True
            self.restart_count = 0
            self.last_restart_time = 0
            self.stream_thread = threading.Thread(target=self._stream_audio, daemon=True)
            self.stream_thread.start()
            print(f"[STT_V2] Started for {self.session_id}")

    def stop(self):
        with self.restart_lock:
            if not self.is_running:
                return
            self.is_running = False
            self.is_listening = False
            # Clear queue and signal shutdown
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.audio_queue.put(None)
            # Close active stream if exists
            self.active_stream = None
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=3)
            print(f"[STT_V2] Stopped for {self.session_id}")

    def mute(self):
        """Mute listening - stop processing transcripts but keep stream alive"""
        self.is_listening = False
        self.current_transcript = ""
        self.last_final_time = None
        print(f"[STT_V2] Muted for {self.session_id}")

    def unmute(self):
        """Unmute listening - resume processing transcripts"""
        self.is_listening = True
        self.current_transcript = ""
        self.last_final_time = None
        print(f"[STT_V2] Unmuted for {self.session_id}")

    def add_audio(self, audio_bytes):
        """Add audio chunk to queue - accepts audio even when muted to keep stream alive"""
        if self.is_running:
            try:
                # Update last audio time when we receive audio (even if muted)
                self.last_audio_time = time.time()
                self.audio_queue.put(audio_bytes, block=False)
            except queue.Full:
                # If queue is full, drop oldest chunk
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put(audio_bytes, block=False)
                except queue.Empty:
                    pass

    def _audio_generator(self):
        """Generate audio chunks from queue, with proper shutdown handling"""
        silent_chunk = b'\x00' * int(self.rate / 10)
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=1)
                if chunk is None:  # Shutdown signal
                    break
                yield chunk
            except queue.Empty:
                # Send silence periodically to keep connection alive
                if time.time() - self.last_audio_time > 8:
                    yield silent_chunk
                    self.last_audio_time = time.time()
                else:
                    continue

    def _create_streaming_requests(self, audio_generator):
        """Create request iterator for streaming recognition"""
        first_request_sent = False

        def request_iterator():
            nonlocal first_request_sent
            try:
                # Send config first
                if not first_request_sent:
                    first_request_sent = True
                    yield cloud_speech.StreamingRecognizeRequest(
                        recognizer=self.recognizer,
                        streaming_config=self.streaming_config
                    )
                
                # Stream audio chunks
                for content in audio_generator:
                    if not self.is_running:
                        break
                    try:
                        yield cloud_speech.StreamingRecognizeRequest(audio=content)
                    except Exception as e:
                        print(f"[STT_V2] Error in request iterator: {str(e)}")
                        break
            except Exception as e:
                print(f"[STT_V2] Request iterator error: {str(e)}")
                raise
        
        return request_iterator()

    def _stream_audio(self):
        """Main streaming loop with improved error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while self.is_running and self.restart_count < self.max_restarts:
            try:
                # Create fresh generators for each attempt
                audio_generator = self._audio_generator()
                request_iterator = self._create_streaming_requests(audio_generator)
                
                # Create streaming recognition client
                responses = stt_client.streaming_recognize(requests=request_iterator)
                self.active_stream = responses
                
                # Process responses - don't break on mute, just skip processing
                self._process_responses(responses)
                
                # If we exit normally, reset error count
                consecutive_errors = 0
                
            except gcp_exceptions.Cancelled:
                # Stream was cancelled - this is normal when stopping
                if not self.is_running:
                    break
                print(f"[STT_V2] Stream cancelled for {self.session_id}")
                
            except gcp_exceptions.Unknown as e:
                # Handle "Exception iterating requests" error
                error_str = str(e)
                if "iterating requests" in error_str:
                    consecutive_errors += 1
                    print(f"[STT_V2 ERROR] Iterator error for {self.session_id} (consecutive: {consecutive_errors})")
                    
                    # If too many consecutive errors, stop restarting
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[STT_V2] Too many consecutive errors, stopping for {self.session_id}")
                        break
                    
                    # Clean up and wait before restart
                    self.active_stream = None
                    if self._should_restart():
                        time.sleep(min(1.0 * consecutive_errors, 5.0))  # Exponential backoff
                        continue
                    break
                else:
                    raise
                    
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                print(f"[STT_V2 ERROR] {self.session_id}: {error_type}: {error_str}")
                
                # Check if it's a recoverable error
                if isinstance(e, (gcp_exceptions.ServiceUnavailable, 
                                gcp_exceptions.DeadlineExceeded,
                                ConnectionError)):
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[STT_V2] Too many consecutive errors, stopping")
                        break
                    self.active_stream = None
                    if self._should_restart():
                        time.sleep(min(1.0 * consecutive_errors, 5.0))
                        continue
                else:
                    # Non-recoverable error, stop
                    import traceback
                    print(traceback.format_exc())
                    break
                    
            finally:
                self.active_stream = None
        
        if self.is_running:
            print(f"[STT_V2] Stream ended for {self.session_id}")

    def _should_restart(self):
        """Check if we should attempt a restart"""
        with self.restart_lock:
            if not self.is_running:
                return False
            if self.restart_count >= self.max_restarts:
                return False
            
            # Cooldown check
            current_time = time.time()
            if current_time - self.last_restart_time < self.restart_cooldown:
                return False
            
            self.restart_count += 1
            self.last_restart_time = current_time
            print(f"[STT_V2] Auto-restarting stream for {self.session_id} (restart #{self.restart_count})")
            
            # Clear queue
            cleared = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            if cleared > 0:
                print(f"[STT_V2] Cleared {cleared} pending audio chunks")
            
            return True

    def _process_responses(self, responses):
        """Process STT responses - continue even when muted to keep stream alive"""
        try:
            for response in responses:
                # Always check if stream should continue
                if not self.is_running:
                    break
                
                # Skip empty responses
                if not response.results:
                    continue
                    
                result = response.results[0]
                if not result.alternatives:
                    continue
                    
                transcript = result.alternatives[0].transcript
                
                # Only process and emit if we're listening
                if self.is_listening:
                    if not result.is_final:
                        # Interim results
                        socketio.emit('interim_transcript', {'text': transcript}, room=self.session_id)
                    else:
                        # Final results
                        if transcript.strip() and len(transcript.strip()) > 1:
                            print(f"[STT_V2 FINAL PART] {self.session_id}: {transcript.strip()}")
                            self.current_transcript += transcript.strip() + " "
                            self.last_final_time = time.time()
                            socketio.emit('final_transcript_part', {'text': transcript.strip()}, room=self.session_id)
                        # Start silence check thread
                        threading.Thread(target=self._check_silence, daemon=True).start()
                # If muted, we still iterate but don't process - this keeps the stream alive
                
        except gcp_exceptions.Cancelled:
            # Stream cancelled - normal when stopping
            if self.is_running:
                print(f"[STT_V2] Response processing cancelled for {self.session_id}")
            raise
        except Exception as e:
            print(f"[STT_V2 PROCESS ERROR] {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    def _check_silence(self):
        """Check for silence period and commit transcript"""
        time.sleep(self.SILENCE_THRESHOLD)
        if self.last_final_time and (time.time() - self.last_final_time) >= self.SILENCE_THRESHOLD:
            if self.current_transcript.strip() and self.is_listening:
                complete_text = self.current_transcript.strip()
                self.current_transcript = ""
                self.last_final_time = None
                if len(complete_text) > 2 and not complete_text.isspace():
                    print(f"[STT_V2 COMPLETE] {self.session_id}: {complete_text}")
                    if self.on_transcript_callback:
                        self.on_transcript_callback(complete_text)
            else:
                self.current_transcript = ""
                self.last_final_time = None


