import base64
from google.cloud import texttospeech


tts_client = texttospeech.TextToSpeechClient()


def synthesize_speech(text):
    try:
        if not text or len(text.strip()) == 0:
            return ""

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-IN",
            name="en-IN-Chirp3-HD-Erinome"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=1.0,
            pitch=0.0
        )
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        return base64.b64encode(response.audio_content).decode("utf-8")
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return ""


