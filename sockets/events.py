import base64
from flask import request
from .socketio import socketio
from ..state import sessions
from ..services.stt import StreamingAudioProcessor
from ..services.llm import get_llm_response
from ..services.tts import synthesize_speech
from ..services.flow import process_user_transcript


@socketio.on("connect")
def handle_connect():
    session_id = request.sid
    sessions[session_id] = {'is_running': False, 'chat_history': [], 'audio_processor': None}
    socketio.emit("info", {"message": "Connected to server"}, room=session_id)


@socketio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    if session_id in sessions:
        if sessions[session_id].get('audio_processor'):
            sessions[session_id]['audio_processor'].stop()
        del sessions[session_id]


@socketio.on("start_interview")
def start_interview(data):
    session_id = request.sid

    if sessions[session_id]['is_running']:
        socketio.emit("error", {"message": "Interview already running"}, room=session_id)
        return

    sessions[session_id]['is_running'] = True
    sessions[session_id]['chat_history'] = [{
        "resume": data.get("resume", ""),
        "jd": data.get("jd", ""),
        "question_type": data.get("question_type", "technical")
    }]

    audio_processor = StreamingAudioProcessor(
        session_id,
        on_transcript_callback=lambda text: process_user_transcript(session_id, text)
    )
    sessions[session_id]['audio_processor'] = audio_processor

    question = get_llm_response(
        session_id=session_id,
        resume_text=data.get("resume", ""),
        job_description=data.get("jd", ""),
        question_type=data.get("question_type", "technical")
    )

    if not question:
        question = (
            "Good morning. I'm Tara, Senior Technical Interviewer at Hiringhood. Thank you for taking the time to speak with us today. "
            "I'll be conducting your technical interview to assess your qualifications for this position. To begin, I'd like you to provide a comprehensive introduction about yourself, "
            "covering your educational background, professional experience, and key technical skills."
        )

    sessions[session_id]['chat_history'].append({"interviewer": question})
    audio_base64 = synthesize_speech(question)

    socketio.emit("ai_message", {
        "text": question,
        "audio": audio_base64,
        "question_number": 1,
        "interview_stage": "early",
        "should_continue": True
    }, room=session_id)


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    session_id = request.sid
    if session_id not in sessions or not sessions[session_id]['is_running']:
        return
    audio_b64 = data.get('audio', '')
    if not audio_b64:
        return
    audio_data = base64.b64decode(audio_b64)
    if sessions[session_id].get('audio_processor'):
        sessions[session_id]['audio_processor'].add_audio(audio_data)


@socketio.on("ai_speech_ended")
def handle_ai_speech_ended():
    session_id = request.sid
    if session_id in sessions and sessions[session_id]['is_running']:
        audio_processor = sessions[session_id].get('audio_processor')
        if audio_processor:
            if not audio_processor.is_running:
                audio_processor.start()
            else:
                audio_processor.unmute()


@socketio.on("stop_interview")
def stop_interview():
    session_id = request.sid
    if session_id in sessions and sessions[session_id]['is_running']:
        sessions[session_id]['is_running'] = False
        if sessions[session_id].get('audio_processor'):
            sessions[session_id]['audio_processor'].stop()
        socketio.emit("info", {"message": "Interview ended"}, room=session_id)


