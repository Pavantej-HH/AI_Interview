import json
import random
import time
import threading
from ..state import sessions, MIN_QUESTIONS, MAX_QUESTIONS
from ..sockets.socketio import socketio
from .llm import get_llm_response
from .tts import synthesize_speech
from .report import generate_dynamic_report


def get_question_count(chat_history):
    count = 0
    for entry in chat_history:
        if 'interviewer' in entry and 'resume' not in entry:
            count += 1
    return count


def check_stop_command(text):
    stop_phrases = [
        "stop the interview", "end the interview", "stop interview", "end interview",
        "finish interview", "conclude interview", "that's all", "i'm done", "no more questions"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in stop_phrases)


def create_conversational_feedback(evaluation, next_question):
    evaluation_lower = evaluation.lower() if evaluation else ""
    if any(word in evaluation_lower for word in ["excellent", "great", "good", "well", "correct", "strong", "impressive"]):
        transitions = ["Excellent. ", "That's very good. ", "Well explained. ", "Good understanding. ", "Perfect. "]
    elif any(word in evaluation_lower for word in ["okay", "decent", "fair", "partial", "some", "adequate"]):
        transitions = ["I see. ", "Understood. ", "Alright. ", "Thank you. "]
    elif any(word in evaluation_lower for word in ["unclear", "incomplete", "missing", "weak", "incorrect"]):
        transitions = ["Let me ask about... ", "Could you clarify... ", "I'd like to understand... ", "Please explain... "]
    else:
        transitions = ["Alright. ", "Thank you. ", "I understand. "]
    transition = random.choice(transitions)
    return f"{transition}{next_question}"


def clean_transcript(text):
    if not text:
        return text
    corrections = {
        r'\bfrustrated\b': 'first of all',
        r'\bbye\b(?!\s*bye)': 'by',
        r'\bsafe introduction\b': 'self introduction',
        r'\bepic opportunity\b': 'this opportunity',
        r'\bcoming to my place\b': 'coming to my background',
        r'\bworked has\b': 'worked as',
        r'\bworked ass\b': 'worked as',
        r'\bhave experience\b': 'have experience in',
        r'\bI am from\s+(?:my|the)\s+': 'I am from ',
        r'\breact js\b': 'React',
        r'\bnode js\b': 'Node.js',
        r'\bmongo db\b': 'MongoDB',
        r'\bpost gre sql\b': 'PostgreSQL',
        r'\bmy sql\b': 'MySQL',
        r'\brest api\b': 'REST API',
        r'\bgraph ql\b': 'GraphQL',
        r'\bci cd\b': 'CI/CD',
    }
    import re
    cleaned = text
    for pattern, replacement in corrections.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def process_user_transcript(session_id, user_text):
    try:
        user_text = clean_transcript(user_text)
        if not user_text or len(user_text) < 3:
            if sessions[session_id].get('audio_processor'):
                sessions[session_id]['audio_processor'].unmute()
            return

        if check_stop_command(user_text):
            questions_asked = get_question_count(sessions[session_id]['chat_history'])
            if questions_asked >= MIN_QUESTIONS:
                end_interview_naturally(session_id, questions_asked, user_initiated=True)
            else:
                audio_processor = sessions[session_id].get('audio_processor')
                if audio_processor:
                    audio_processor.mute()
                confirmation_msg = (
                    f"I understand you'd like to conclude the interview. However, we've only covered {questions_asked} questions. "
                    f"To provide a comprehensive assessment, I'd recommend answering at least {MIN_QUESTIONS - questions_asked} more question(s). "
                    f"Would you like to continue, or shall we conclude with the current assessment?"
                )
                audio_base64 = synthesize_speech(confirmation_msg)
                socketio.emit("ai_message", {
                    "text": confirmation_msg,
                    "audio": audio_base64,
                    "requires_confirmation": True
                }, room=session_id)
            return

        audio_processor = sessions[session_id].get('audio_processor')
        if audio_processor:
            audio_processor.mute()

        sessions[session_id]['chat_history'].append({"candidate": user_text})
        socketio.emit('user_transcript', {'text': user_text}, room=session_id)

        questions_asked = get_question_count(sessions[session_id]['chat_history'])
        if questions_asked >= MAX_QUESTIONS:
            end_interview_naturally(session_id, questions_asked)
            return

        next_response = get_llm_response(session_id=session_id, user_response=user_text)
        if not next_response:
            next_response = json.dumps({
                "evaluation": "Let's continue our discussion",
                "score": 0,
                "next_question": "Thank you for that response. Let's explore another technical area.",
                "should_continue": True,
                "interview_stage": "mid"
            })

        try:
            cleaned_response = next_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            response_data = json.loads(cleaned_response)

            evaluation = response_data.get("evaluation", "")
            score = response_data.get("score", 0)
            next_question = response_data.get("next_question", "Let's continue our discussion.")
            should_continue = response_data.get("should_continue", True)
            interview_stage = response_data.get("interview_stage", "mid")

            questions_asked = get_question_count(sessions[session_id]['chat_history'])
            sessions[session_id]['chat_history'].append({
                "interviewer": next_question,
                "evaluation": evaluation,
                "score": score
            })

            if not should_continue or questions_asked >= MAX_QUESTIONS:
                if questions_asked >= MIN_QUESTIONS:
                    return end_interview_naturally(session_id, questions_asked + 1)

            conversational_response = create_conversational_feedback(evaluation, next_question)
            audio_base64 = synthesize_speech(conversational_response)
            socketio.emit("ai_message", {
                "text": conversational_response,
                "audio": audio_base64,
                "evaluation": evaluation,
                "score": score,
                "question_number": questions_asked + 1,
                "interview_stage": interview_stage,
                "should_continue": should_continue
            }, room=session_id)

        except json.JSONDecodeError:
            questions_asked = get_question_count(sessions[session_id]['chat_history'])
            sessions[session_id]['chat_history'].append({"interviewer": next_response})
            audio_base64 = synthesize_speech(next_response)
            socketio.emit("ai_message", {
                "text": next_response,
                "audio": audio_base64,
                "question_number": questions_asked + 1
            }, room=session_id)

    except Exception as e:
        print(f"[PROCESS ERROR] {e}")
        import traceback
        traceback.print_exc()
        if sessions[session_id].get('audio_processor'):
            sessions[session_id]['audio_processor'].unmute()


def end_interview_naturally(session_id, questions_asked, user_initiated=False):
    try:
        audio_processor = sessions[session_id].get('audio_processor')
        if audio_processor:
            audio_processor.stop()
        sessions[session_id]['is_running'] = False

        if user_initiated:
            closing_message = (
                f"Thank you for your time and responses today. We've discussed {questions_asked} questions, which provides valuable insight into your capabilities. "
                f"I appreciate your openness in sharing your experience. This concludes our technical interview session."
            )
        else:
            closing_message = (
                f"Thank you for this comprehensive discussion. We've thoroughly covered {questions_asked} questions across various technical domains, "
                f"and I have a clear understanding of your expertise and approach to problem-solving. This concludes our technical interview session."
            )

        audio_base64 = synthesize_speech(closing_message)
        word_count = len(closing_message.split())
        estimated_duration = (word_count / 150) * 30
        buffer_time = 3
        total_wait_time = estimated_duration + buffer_time

        socketio.emit("ai_message", {
            "text": closing_message,
            "audio": audio_base64,
            "is_final": True
        }, room=session_id)

        def generate_and_send_report():
            report = generate_dynamic_report(session_id)
            if not report:
                return
            time.sleep(total_wait_time)
            socketio.emit("interview_complete", {"report": report}, room=session_id)

        threading.Thread(target=generate_and_send_report, daemon=True).start()

    except Exception as e:
        print(f"[END INTERVIEW ERROR] {e}")
        import traceback
        traceback.print_exc()


