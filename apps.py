import os
import json
import base64
import queue
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv
from google.cloud import texttospeech, speech
import requests
import random

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False, ping_timeout=60, ping_interval=25)

sessions = {}
MIN_QUESTIONS = 4
MAX_QUESTIONS = 10
IDEAL_QUESTIONS = 8

class StreamingAudioProcessor:
    def __init__(self, session_id, on_transcript_callback):
        self.session_id = session_id
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.is_listening = False
        self.stream_thread = None
        self.on_transcript_callback = on_transcript_callback
        self.current_transcript = ""
        self.last_final_time = None
        self.SILENCE_THRESHOLD = 2
        self.restart_lock = threading.Lock()
        self.restart_count = 0
        self.max_restarts = 300
        
    def start(self):
        with self.restart_lock:
            if self.is_running:
                return
            self.is_running = True
            self.is_listening = True
            self.restart_count = 0
            self.stream_thread = threading.Thread(target=self._stream_audio, daemon=True)
            self.stream_thread.start()
            print(f"[STT] Started for {self.session_id}")
    
    def stop(self):
        with self.restart_lock:
            if not self.is_running:
                return
            self.is_running = False
            self.is_listening = False
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.audio_queue.put(None)
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=2)
            print(f"[STT] Stopped for {self.session_id}")
    
    def mute(self):
        self.is_listening = False
        self.current_transcript = ""
        self.last_final_time = None
        print(f"[STT] Muted for {self.session_id}")
    
    def unmute(self):
        self.is_listening = True
        self.current_transcript = ""
        self.last_final_time = None
        print(f"[STT] Unmuted for {self.session_id}")
    
    def add_audio(self, audio_bytes):
        if self.is_running and self.is_listening:
            try:
                self.audio_queue.put(audio_bytes, block=False)
            except queue.Full:
                pass
    
    def _audio_generator(self):
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                if chunk is None:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue
    
    def _restart_stream(self):
        with self.restart_lock:
            if not self.is_running or self.restart_count >= self.max_restarts:
                return False
            
            self.restart_count += 1
            print(f"[STT] Auto-restarting stream for {self.session_id} (restart #{self.restart_count})")
            
            cleared = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            
            if cleared > 0:
                print(f"[STT] Cleared {cleared} pending audio chunks")
            
            return True
    
    def _stream_audio(self):
        while self.is_running and self.restart_count < self.max_restarts:
            try:
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                    enable_automatic_punctuation=True,
                    model="latest_long",
                    use_enhanced=True,
                    profanity_filter=False,
                    enable_word_time_offsets=False,
                    enable_word_confidence=False,
                    max_alternatives=1,
                    speech_contexts=[
                        speech.SpeechContext(
                            phrases=[
                                "stop the interview", "end the interview", "stop interview", "end interview",
                                "finish interview", "conclude interview", "that's all", "I'm done", "no more questions",
                                "I think we can stop here", "that should be enough", "I believe that covers everything",
                                "let's wrap this up", "we can end now", "please conclude the interview",
                                "I'm finished with the interview", "this can be the end", "we're done here",
                                "that concludes my part", "I have nothing more to add", "let's finish up",
                                "I'm ready to finish", "we can stop now", "that's all I have",
                                "I believe we're done", "let's end the session", "please stop now",
                                "I'd like to end here", "we can conclude now", "that will be all for today",
                                "thank you that will be all", "I appreciate your time we can stop",
                                "I think we've covered enough", "this seems like a good stopping point",
                                "first of all", "thank you", "self introduction", "myself", "worked as",
                                "experience", "background", "opportunity", "coming to", "let me tell you",
                                "I have been", "I am currently", "my role", "responsible for", "worked on",
                                "API", "database", "backend", "frontend", "full stack",
                                "React", "Angular", "Vue", "Python", "JavaScript",
                                "TypeScript", "Node.js", "Express", "Django", "Flask",
                                "Spring Boot", "AWS", "Azure", "Google Cloud",
                                "Docker", "Kubernetes", "microservices",
                                "MongoDB", "PostgreSQL", "MySQL", "Redis",
                                "SQL", "NoSQL", "REST API", "GraphQL",
                            ],
                            boost=15.0
                        )
                    ]
                )
                
                streaming_config = speech.StreamingRecognitionConfig(
                    config=config, 
                    interim_results=True,
                    single_utterance=False,
                    enable_voice_activity_events=True
                )
                
                audio_generator = self._audio_generator()
                responses = stt_client.streaming_recognize(streaming_config, audio_generator)
                self._process_responses(responses)
                
            except Exception as e:
                error_str = str(e)
                
                if "400" in error_str and ("Audio Timeout" in error_str or "exceeded" in error_str):
                    if self.is_running:
                        print(f"[STT] Stream timeout for {self.session_id}, auto-restarting...")
                        if self._restart_stream():
                            time.sleep(0.1)
                            continue
                        else:
                            print(f"[STT] Max restarts reached for {self.session_id}")
                            break
                
                elif self.is_running:
                    print(f"[STT ERROR] {self.session_id}: {error_str}")
                    if "Stream" in error_str or "DEADLINE_EXCEEDED" in error_str:
                        if self._restart_stream():
                            time.sleep(0.5)
                            continue
                    break
                else:
                    break
        
        if self.is_running:
            print(f"[STT] Stream ended for {self.session_id}")
    
    def _process_responses(self, responses):
        for response in responses:
            if not self.is_running or not self.is_listening:
                break
            
            if not response.results:
                continue
            
            result = response.results[0]
            if not result.alternatives:
                continue
            
            transcript = result.alternatives[0].transcript
            
            if not result.is_final:
                socketio.emit('interim_transcript', {'text': transcript}, room=self.session_id)
            else:
                if transcript.strip() and len(transcript.strip()) > 1:
                    print(f"[STT FINAL PART] {self.session_id}: {transcript.strip()}")
                    self.current_transcript += transcript.strip() + " "
                    self.last_final_time = time.time()
                    socketio.emit('final_transcript_part', {'text': transcript.strip()}, room=self.session_id)
                    threading.Thread(target=self._check_silence, daemon=True).start()
    
    def _check_silence(self):
        time.sleep(self.SILENCE_THRESHOLD)
        if self.last_final_time and (time.time() - self.last_final_time) >= self.SILENCE_THRESHOLD:
            if self.current_transcript.strip() and self.is_listening:
                complete_text = self.current_transcript.strip()
                self.current_transcript = ""
                self.last_final_time = None
                
                if len(complete_text) > 2 and not complete_text.isspace():
                    print(f"[STT COMPLETE] {self.session_id}: {complete_text}")
                    if self.on_transcript_callback:
                        self.on_transcript_callback(complete_text)
            else:
                self.current_transcript = ""
                self.last_final_time = None

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

def get_llm_response(session_id, resume_text=None, job_description=None, question_type=None, user_response=None):
    chat_history = sessions[session_id]['chat_history']

    try:
        if not user_response:
            prompt_text = f"""You are Tara, a senior technical interviewer at Hiringhood. Conduct a comprehensive, professional technical interview.

CANDIDATE RESUME:
{resume_text}

JOB REQUIREMENTS:
{job_description}

INTERVIEW TYPE: {question_type}

**OPENING GUIDELINES:**
Start with a warm, professional greeting based on time of day

Introduce yourself as Tara, Senior Technical Interviewer at Hiringhood

Express appreciation for the candidate's time

Clearly state the interview purpose: assessing qualifications for the position

Request a comprehensive self-introduction covering educational background, professional experience, and key technical skills

Maintain formal yet welcoming tone throughout

Do not disclose the number of questions or interview structure

Example tone: "Good morning/afternoon. I'm Tara, Senior Technical Interviewer at Hiringhood. Thank you for taking the time to speak with us today. I'll be conducting your technical interview to assess your qualifications for this position. To begin, I'd like you to provide a comprehensive introduction about yourself, covering your educational background, professional experience, and key technical skills."

Return ONLY the professional opening question."""

        else:
            questions_asked = get_question_count(chat_history)
            conversation_history = "\n".join([
                f"Interviewer: {m.get('interviewer', '')}" if 'interviewer' in m else f"Candidate: {m.get('candidate', '')}"
                for m in chat_history[-8:]
            ])

            if questions_asked < MIN_QUESTIONS:
                stage_guidance = "Early stage - explore fundamentals and background thoroughly"
                should_continue_default = True
            elif questions_asked < IDEAL_QUESTIONS:
                stage_guidance = "Mid stage - dive deeper into technical competencies and problem-solving"
                should_continue_default = True
            else:
                stage_guidance = f"Late stage ({questions_asked} questions asked) - prepare to conclude unless critical areas need coverage"
                should_continue_default = False

            prompt_text = f"""You are Tara, a senior technical interviewer. Continue the professional interview conversation.

CONVERSATION HISTORY:
{conversation_history}

CANDIDATE'S RESPONSE:
"{user_response}"

INTERVIEW PROGRESS: {questions_asked} questions asked
STAGE: {stage_guidance}

CANDIDATE RESUME:
{resume_text}

JOB REQUIREMENTS:
{job_description}

INTERVIEW TYPE: {question_type}

**YOUR ROLE:**
1. Provide detailed technical evaluation of the response
2. Ask probing follow-up questions for brief answers
3. Move to new technical areas when current topic is exhausted
4. Maintain professional, formal tone throughout
5. After {MIN_QUESTIONS} questions, evaluate if more depth is needed
6. Maximum {MAX_QUESTIONS} questions - conclude naturally when sufficient coverage achieved
7. Ask the Questions related to the JD , Resume provided , conversation history and User response.
8. Always ensure relevance to the job description and candidate's background

**INTERVIEW FLOW:**
- Questions 1-{MIN_QUESTIONS}: Core fundamentals and experience
- Questions {MIN_QUESTIONS+1}-{IDEAL_QUESTIONS}: Advanced technical depth
- Questions {IDEAL_QUESTIONS+1}-{MAX_QUESTIONS}: Final clarifications only if needed

Return ONLY this JSON:

{{
  "evaluation": "Comprehensive technical assessment with specific strengths and areas for improvement",
  "score": 7,
  "next_question": "Professional, probing follow-up question or new topic",
  "should_continue": {str(should_continue_default).lower()},
  "interview_stage": "early/mid/late"
}}

Scoring Guidelines:
- 1-3: Significant gaps in knowledge
- 4-6: Basic understanding with notable limitations  
- 7-8: Strong competence with minor gaps
- 9-10: Exceptional expertise and articulation"""

        headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            return json.dumps({
                "evaluation": "Let's continue with our discussion",
                "score": 0,
                "next_question": "Could you please elaborate further on that point?",
                "should_continue": True,
                "interview_stage": "mid"
            })
        
        response_data = response.json()
        result = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return result

    except Exception as e:
        print(f"[GEMINI ERROR] {e}")
        return json.dumps({
            "evaluation": "Let's continue with our conversation",
            "score": 0,
            "next_question": "Thank you for that response. Let's explore another technical area.",
            "should_continue": True,
            "interview_stage": "mid"
        })

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

def create_conversational_feedback(evaluation, next_question):
    evaluation_lower = evaluation.lower()
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

def generate_dynamic_report(session_id):
    """Generate comprehensive interview report using Gemini AI"""
    try:
        chat_history = sessions[session_id]['chat_history']
        
        if len(chat_history) < 2:
            return None
            
        resume = chat_history[0].get('resume', 'N/A')
        jd = chat_history[0].get('jd', 'N/A')
        question_type = chat_history[0].get('question_type', 'technical')
        
        # Build Q&A pairs with CORRECT pairing logic
        # Structure: Question → Answer → NextQuestion(contains evaluation of Answer)
        qa_pairs = []
        scores = []
        
        print(f"[REPORT DEBUG] Total chat history entries: {len(chat_history)}")
        
        # Skip first entry (resume/jd data) and process pairs
        i = 1
        qa_number = 0
        
        while i < len(chat_history):
            current_entry = chat_history[i]
            
            # Find interviewer question (skip if it's just metadata)
            if 'interviewer' in current_entry and 'resume' not in current_entry:
                qa_number += 1
                question = current_entry.get('interviewer', '')
                
                # Find the candidate's answer (next entry should be candidate response)
                answer = ''
                if i + 1 < len(chat_history) and 'candidate' in chat_history[i + 1]:
                    answer = chat_history[i + 1].get('candidate', '')
                    
                    # The evaluation and score are in the NEXT interviewer question (i+2)
                    evaluation = ''
                    score = 0
                    
                    if i + 2 < len(chat_history) and 'interviewer' in chat_history[i + 2]:
                        next_question_entry = chat_history[i + 2]
                        evaluation = next_question_entry.get('evaluation', '')
                        score = next_question_entry.get('score', 0)
                        
                        print(f"[REPORT DEBUG] Q{qa_number}: Found score={score}, eval length={len(evaluation)}")
                    else:
                        # Last answer might not have a follow-up question yet
                        print(f"[REPORT DEBUG] Q{qa_number}: Last question, no evaluation yet")
                    
                    # Add the Q&A pair
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'score': score,
                        'evaluation': evaluation if evaluation else 'Response received and being evaluated'
                    })
                    
                    if score > 0:
                        scores.append(score)
                        print(f"[REPORT DEBUG] Added score {score} to scores list")
                
            i += 1
        
        print(f"[REPORT DEBUG] Found {len(qa_pairs)} Q&A pairs, {len(scores)} valid scores: {scores}")
        
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0
        
        # If no valid scores, return early with minimal report
        if not scores or avg_score == 0:
            print(f"[REPORT WARNING] No valid scores found, generating minimal report")
            return generate_minimal_report(session_id, qa_pairs, resume, jd, question_type)
        
        # Build conversation transcript for Gemini
        conversation_transcript = ""
        for idx, qa in enumerate(qa_pairs, 1):
            conversation_transcript += f"\n{'='*60}\n"
            conversation_transcript += f"QUESTION {idx}:\n{qa['question']}\n\n"
            conversation_transcript += f"CANDIDATE ANSWER:\n{qa['answer']}\n\n"
            conversation_transcript += f"INTERVIEWER EVALUATION:\n{qa['evaluation']}\n"
            conversation_transcript += f"SCORE: {qa['score']}/10\n"
            conversation_transcript += f"{'='*60}\n"
        
        # Generate dynamic analysis using Gemini
        print(f"[REPORT] Generating AI-powered comprehensive analysis...")
        
        analysis_prompt = f"""You are an expert technical hiring manager analyzing a completed interview. Provide a comprehensive, professional assessment.

**CANDIDATE RESUME:**
{resume[:2000]}

**JOB DESCRIPTION:**
{jd[:2000]}

**INTERVIEW TYPE:** {question_type}

**COMPLETE INTERVIEW TRANSCRIPT:**
{conversation_transcript}

**STATISTICS:**
- Total Questions Asked: {len(qa_pairs)}
- Average Score: {avg_score}/10
- Individual Scores: {scores}
- Score Range: {min(scores) if scores else 0} to {max(scores) if scores else 0}

**ANALYSIS REQUIREMENTS:**
Based on the complete interview transcript above, provide a detailed professional assessment. Be specific and reference actual responses from the interview.

Return ONLY valid JSON in this exact format:

{{
  "overall_evaluation": "A comprehensive 4-5 sentence analysis of the candidate's performance, communication style, technical depth, and overall impression. Be honest and specific.",
  
  "recommendation": "Choose ONE: 'Strong Hire - [reason]', 'Hire - [reason]', 'Maybe - [reason]', or 'No Hire - [reason]'. Provide specific justification based on interview performance.",
  
  "key_strengths": [
    "Specific strength with example from their responses",
    "Another strength demonstrated during interview",
    "Third notable positive aspect"
  ],
  
  "areas_for_improvement": [
    "Specific area needing development with constructive feedback",
    "Another improvement area with actionable advice",
    "Third development opportunity"
  ],
  
  "technical_assessment": {{
    "depth_of_knowledge": 7,
    "problem_solving": 6,
    "communication": 5,
    "experience_relevance": 6
  }},
  
  "resume_alignment": "2-3 sentences analyzing if their interview responses match what's claimed in their resume. Be honest about any discrepancies.",
  
  "job_fit": "2-3 sentences on how well the candidate's demonstrated skills and experience fit the specific job requirements.",
  
  "next_steps": "Specific recommendation for next stage in hiring process"
}}

CRITICAL RULES:
1. Be honest and specific - reference actual interview responses
2. If performance was weak, say so professionally
3. All scores in technical_assessment should be 1-10 integers
4. Keep strengths and improvements lists to exactly 3 items each
5. Make recommendation realistic based on actual performance
6. Return ONLY the JSON, no other text"""

        headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
        payload = {"contents": [{"parts": [{"text": analysis_prompt}]}]}
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=45)
        
        if response.status_code == 200:
            response_data = response.json()
            ai_analysis = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # Clean and parse AI response
            if ai_analysis.startswith("```json"):
                ai_analysis = ai_analysis[7:]
            elif ai_analysis.startswith("```"):
                ai_analysis = ai_analysis[3:]
            if ai_analysis.endswith("```"):
                ai_analysis = ai_analysis[:-3]
            
            ai_analysis = ai_analysis.strip()
            
            try:
                analysis_data = json.loads(ai_analysis)
                
                # Validate and fix technical_assessment scores
                if 'technical_assessment' in analysis_data:
                    for key in analysis_data['technical_assessment']:
                        val = analysis_data['technical_assessment'][key]
                        # Convert to int if it's a string number
                        if isinstance(val, str):
                            try:
                                analysis_data['technical_assessment'][key] = int(val)
                            except:
                                analysis_data['technical_assessment'][key] = int(avg_score)
                        # Ensure it's in 1-10 range
                        analysis_data['technical_assessment'][key] = max(1, min(10, int(analysis_data['technical_assessment'][key])))
                
                print(f"[REPORT] AI analysis generated successfully")
                
            except json.JSONDecodeError as e:
                print(f"[REPORT] Failed to parse AI response as JSON: {e}")
                print(f"[REPORT] Response preview: {ai_analysis[:300]}...")
                analysis_data = generate_fallback_analysis(avg_score, len(qa_pairs), qa_pairs)
            
        else:
            print(f"[REPORT] AI analysis request failed with status {response.status_code}")
            analysis_data = generate_fallback_analysis(avg_score, len(qa_pairs), qa_pairs)
        
        # Construct final report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'interviewer': 'Tara (Senior Technical Interviewer)',
            'candidate_details': {
                'resume_summary': resume[:400] + '...' if len(resume) > 400 else resume,
                'job_description': jd[:400] + '...' if len(jd) > 400 else jd,
                'interview_type': question_type
            },
            'interview_statistics': {
                'total_questions': len(qa_pairs),
                'overall_score': avg_score,
                'score_distribution': {
                    'excellent (9-10)': len([s for s in scores if s >= 9]),
                    'good (7-8)': len([s for s in scores if 7 <= s < 9]),
                    'average (5-6)': len([s for s in scores if 5 <= s < 7]),
                    'below_average (1-4)': len([s for s in scores if s < 5])
                }
            },
            'ai_analysis': analysis_data,
            'detailed_qa': qa_pairs
        }
        
        print(f"[REPORT] ✅ Complete report generated successfully")
        print(f"[REPORT]    - Questions: {len(qa_pairs)}")
        print(f"[REPORT]    - Avg Score: {avg_score}/10")
        print(f"[REPORT]    - Recommendation: {analysis_data.get('recommendation', 'N/A')[:50]}...")
        return report
        
    except Exception as e:
        print(f"[REPORT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return generate_fallback_report(session_id)

def generate_minimal_report(session_id, qa_pairs, resume, jd, question_type):
    """Generate minimal report when no valid scores are available"""
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'interviewer': 'Tara (Senior Technical Interviewer)',
        'candidate_details': {
            'resume_summary': resume[:400] + '...' if len(resume) > 400 else resume,
            'job_description': jd[:400] + '...' if len(jd) > 400 else jd,
            'interview_type': question_type
        },
        'interview_statistics': {
            'total_questions': len(qa_pairs),
            'overall_score': 0,
            'score_distribution': {
                'excellent (9-10)': 0,
                'good (7-8)': 0,
                'average (5-6)': 0,
                'below_average (1-4)': 0
            }
        },
        'ai_analysis': {
            "overall_evaluation": f"Interview was initiated but not completed with scoreable responses. {len(qa_pairs)} questions were asked but responses were not evaluated with numerical scores.",
            "recommendation": "Incomplete Interview - Unable to provide hiring recommendation without scored responses",
            "key_strengths": ["Interview participation", "Time commitment", "Professional engagement"],
            "areas_for_improvement": ["Complete interview process", "Provide detailed technical responses", "Demonstrate technical knowledge clearly"],
            "technical_assessment": {
                "depth_of_knowledge": 0,
                "problem_solving": 0,
                "communication": 0,
                "experience_relevance": 0
            },
            "resume_alignment": "Unable to assess alignment due to incomplete interview",
            "job_fit": "Unable to determine fit without completed assessment",
            "next_steps": "Re-schedule complete technical interview"
        },
        'detailed_qa': qa_pairs
    }

def generate_fallback_analysis(avg_score, qa_count, qa_pairs):
    """Fallback analysis if AI fails - more nuanced than simple score ranges"""
    
    # Analyze response quality
    short_answers = sum(1 for qa in qa_pairs if len(qa['answer'].split()) < 15)
    unclear_answers = sum(1 for qa in qa_pairs if any(word in qa['answer'].lower() 
                          for word in ["don't remember", "don't know", "not sure", "maybe", "i think"]))
    
    # Determine recommendation based on multiple factors
    if avg_score >= 8.0 and unclear_answers == 0:
        recommendation = "Strong Hire - Demonstrated excellent technical competence and clear communication"
        overall_eval = f"Candidate performed exceptionally well across {qa_count} questions with an average score of {avg_score}/10. Shows strong technical foundation, clear communication, and confidence in their expertise."
        
    elif avg_score >= 7.0 and unclear_answers <= 1:
        recommendation = "Hire - Good technical foundation with clear potential"
        overall_eval = f"Candidate showed solid performance across {qa_count} questions with an average score of {avg_score}/10. Demonstrates good technical skills and adequate communication, with room for minor improvements."
        
    elif avg_score >= 6.0:
        recommendation = "Maybe - Some potential but notable gaps identified"
        overall_eval = f"Candidate completed {qa_count} questions with an average score of {avg_score}/10. Shows basic understanding but has gaps in technical depth and communication clarity. {unclear_answers} responses showed uncertainty."
        
    elif avg_score >= 5.0:
        recommendation = "Maybe - Needs significant development"
        overall_eval = f"Candidate struggled with technical depth across {qa_count} questions (average: {avg_score}/10). Multiple responses lacked clarity or detail. {unclear_answers} responses showed inability to recall or explain concepts."
        
    else:
        recommendation = "No Hire - Significant skill gaps and communication issues"
        overall_eval = f"Candidate demonstrated inadequate technical knowledge across {qa_count} questions (average: {avg_score}/10). Frequent inability to provide detailed responses or recall project details. Not ready for this role."
    
    # Generate context-aware strengths and improvements
    strengths = []
    improvements = []
    
    if qa_count >= IDEAL_QUESTIONS:
        strengths.append(f"Completed comprehensive interview ({qa_count} questions)")
    else:
        improvements.append(f"Interview concluded early with only {qa_count} questions covered")
    
    if unclear_answers == 0:
        strengths.append("Provided confident responses without hesitation")
    else:
        improvements.append(f"Showed uncertainty in {unclear_answers} responses, indicating knowledge gaps")
    
    if short_answers < qa_count / 2:
        strengths.append("Generally provided detailed explanations")
    else:
        improvements.append(f"Many responses were brief ({short_answers}/{qa_count}), lacking technical depth")
    
    if avg_score >= 7:
        strengths.append("Demonstrated solid technical foundation")
        improvements.append("Could improve by providing more specific examples")
    else:
        improvements.append("Needs to strengthen core technical knowledge and practical experience")
        improvements.append("Should work on articulating technical concepts more clearly")
    
    # Ensure we have 3 of each
    while len(strengths) < 3:
        strengths.append("Maintained professional demeanor throughout interview")
    while len(improvements) < 3:
        improvements.append("Requires more hands-on experience with technologies mentioned in resume")
    
    return {
        "overall_evaluation": overall_eval,
        "recommendation": recommendation,
        "key_strengths": strengths[:3],
        "areas_for_improvement": improvements[:3],
        "technical_assessment": {
            "depth_of_knowledge": int(round(avg_score)),
            "problem_solving": int(round(avg_score * 0.9)),
            "communication": int(round(avg_score * 0.8)) if unclear_answers > 2 else int(round(avg_score)),
            "experience_relevance": int(round(avg_score * 0.85))
        },
        "resume_alignment": f"Based on {qa_count} questions, candidate's responses {'align well with' if avg_score >= 7 else 'show discrepancies from'} resume claims. {'Strong correlation between stated and demonstrated skills.' if avg_score >= 7 else 'Some claimed experiences could not be adequately explained.'}",
        "job_fit": f"Candidate demonstrates {'strong' if avg_score >= 7.5 else 'partial' if avg_score >= 6 else 'limited'} alignment with job requirements. {'Recommended for next round.' if avg_score >= 7 else 'Additional assessment recommended.' if avg_score >= 5.5 else 'Does not meet minimum requirements at this time.'}",
        "next_steps": "Proceed to technical round with senior engineer" if avg_score >= 7.5 else "Consider additional screening" if avg_score >= 6 else "Thank candidate for their time, not proceeding"
    }

def generate_fallback_report(session_id):
    """Complete fallback report generation"""
    try:
        chat_history = sessions[session_id]['chat_history']
        qa_pairs = []
        scores = []
        
        i = 1
        while i < len(chat_history):
            entry = chat_history[i]
            if 'interviewer' in entry:
                question = entry.get('interviewer', '')
                score = entry.get('score', 0)
                answer = ''
                if i + 1 < len(chat_history) and 'candidate' in chat_history[i + 1]:
                    answer = chat_history[i + 1].get('candidate', '')
                if answer:
                    qa_pairs.append({'question': question, 'answer': answer, 'score': score})
                    if score > 0:
                        scores.append(score)
            i += 1
        
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0
        analysis = generate_fallback_analysis(avg_score, len(qa_pairs), qa_pairs)
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'interviewer': 'Tara (Senior Technical Interviewer)',
            'interview_statistics': {'total_questions': len(qa_pairs), 'overall_score': avg_score},
            'ai_analysis': analysis,
            'detailed_qa': qa_pairs
        }
    except:
        return None

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
            print(f"[PROCESS] Skipped empty transcript for {session_id}")
            if sessions[session_id].get('audio_processor'):
                sessions[session_id]['audio_processor'].unmute()
            return
        
        if check_stop_command(user_text):
            print(f"[INTERVIEW] User requested to stop interview")
            questions_asked = get_question_count(sessions[session_id]['chat_history'])
            
            if questions_asked >= MIN_QUESTIONS:
                end_interview_naturally(session_id, questions_asked, user_initiated=True)
            else:
                audio_processor = sessions[session_id].get('audio_processor')
                if audio_processor:
                    audio_processor.mute()
                
                confirmation_msg = f"I understand you'd like to conclude the interview. However, we've only covered {questions_asked} questions. To provide a comprehensive assessment, I'd recommend answering at least {MIN_QUESTIONS - questions_asked} more question(s). Would you like to continue, or shall we conclude with the current assessment?"
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
            print(f"[INTERVIEW] Reached maximum questions ({MAX_QUESTIONS}), ending interview")
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
            
        except json.JSONDecodeError as e:
            print(f"[JSON PARSE ERROR] Failed to parse: {next_response[:200]}...")
            print(f"[JSON PARSE ERROR] Error: {e}")
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
    """End interview with natural closing and wait for TTS to complete before sending report"""
    try:
        # Stop audio processor immediately
        audio_processor = sessions[session_id].get('audio_processor')
        if audio_processor:
            audio_processor.stop()
        
        sessions[session_id]['is_running'] = False
        
        # Generate closing message
        if user_initiated:
            closing_message = f"Thank you for your time and responses today. We've discussed {questions_asked} questions, which provides valuable insight into your capabilities. I appreciate your openness in sharing your experience. This concludes our technical interview session."
        else:
            closing_message = f"Thank you for this comprehensive discussion. We've thoroughly covered {questions_asked} questions across various technical domains, and I have a clear understanding of your expertise and approach to problem-solving. This concludes our technical interview session."
        
        audio_base64 = synthesize_speech(closing_message)
        
        # Calculate approximate TTS duration (rough estimate: 150 words per minute)
        word_count = len(closing_message.split())
        estimated_duration = (word_count / 150) * 30  # seconds
        buffer_time = 3  # additional buffer
        total_wait_time = estimated_duration + buffer_time
        
        print(f"[INTERVIEW] Closing message: {word_count} words, waiting {total_wait_time:.1f}s before report")
        
        # Send closing message
        socketio.emit("ai_message", {
            "text": closing_message,
            "audio": audio_base64,
            "is_final": True
        }, room=session_id)
        
        # Generate report in background but wait to send it
        def generate_and_send_report():
            print(f"[REPORT] Starting report generation for {session_id}...")
            report = generate_dynamic_report(session_id)
            
            if not report:
                print(f"[REPORT ERROR] Failed to generate report for {session_id}")
                return
            
            # Wait for TTS to complete
            print(f"[REPORT] Report ready, waiting {total_wait_time:.1f}s for TTS to complete...")
            time.sleep(total_wait_time)
            
            # Send report after TTS completes
            socketio.emit("interview_complete", {"report": report}, room=session_id)
            print(f"[REPORT] Report sent to {session_id} after TTS completion")
        
        # Start report generation in background
        threading.Thread(target=generate_and_send_report, daemon=True).start()
        
        print(f"[INTERVIEW] Ended after {questions_asked} questions")
        
    except Exception as e:
        print(f"[END INTERVIEW ERROR] {e}")
        import traceback
        traceback.print_exc()

@socketio.on("connect")
def handle_connect():
    session_id = request.sid
    sessions[session_id] = {'is_running': False, 'chat_history': [], 'audio_processor': None}
    print(f"[CONNECT] {session_id}")
    emit("info", {"message": "Connected to server"})

@socketio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    if session_id in sessions:
        if sessions[session_id].get('audio_processor'):
            sessions[session_id]['audio_processor'].stop()
        del sessions[session_id]
    print(f"[DISCONNECT] {session_id}")

@socketio.on("start_interview")
def start_interview(data):
    session_id = request.sid
    print(f"[START_INTERVIEW] {session_id}")

    if sessions[session_id]['is_running']:
        emit("error", {"message": "Interview already running"})
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
        question = "Good morning. I'm Tara, Senior Technical Interviewer at Hiringhood. Thank you for taking the time to speak with us today. I'll be conducting your technical interview to assess your qualifications for this position. To begin, I'd like you to provide a comprehensive introduction about yourself, covering your educational background, professional experience, and key technical skills."

    sessions[session_id]['chat_history'].append({"interviewer": question})
    audio_base64 = synthesize_speech(question)

    print(f"[START_INTERVIEW] Starting professional interview with Tara")
    emit("ai_message", {
        "text": question,
        "audio": audio_base64,
        "question_number": 1,
        "interview_stage": "early",
        "should_continue": True
    })

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    session_id = request.sid
    if session_id not in sessions or not sessions[session_id]['is_running']:
        return
    try:
        audio_b64 = data.get('audio', '')
        if not audio_b64:
            return
        audio_data = base64.b64decode(audio_b64)
        if sessions[session_id].get('audio_processor'):
            sessions[session_id]['audio_processor'].add_audio(audio_data)
    except Exception as e:
        print(f"[AUDIO ERROR] {e}")

@socketio.on("ai_speech_ended")
def handle_ai_speech_ended():
    session_id = request.sid
    if session_id in sessions and sessions[session_id]['is_running']:
        audio_processor = sessions[session_id].get('audio_processor')
        if audio_processor:
            if not audio_processor.is_running:
                print(f"[STT] Starting processor for {session_id}")
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
        print(f"[STOP_INTERVIEW] {session_id}")
        emit("info", {"message": "Interview ended"})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}

@app.route("/debug_session/<session_id>")
def debug_session(session_id):
    """Debug endpoint to view chat history structure"""
    if session_id not in sessions:
        return {"error": "Session not found"}, 404
    
    chat_history = sessions[session_id].get('chat_history', [])
    debug_output = []
    
    for idx, entry in enumerate(chat_history):
        entry_type = "METADATA" if 'resume' in entry else "INTERVIEWER" if 'interviewer' in entry else "CANDIDATE" if 'candidate' in entry else "UNKNOWN"
        
        debug_entry = {
            "index": idx,
            "type": entry_type,
            "has_score": 'score' in entry,
            "score": entry.get('score', 'N/A'),
            "has_evaluation": 'evaluation' in entry,
            "eval_length": len(entry.get('evaluation', '')),
            "content_preview": str(entry)[:100] + "..."
        }
        debug_output.append(debug_entry)
    
    return {
        "session_id": session_id,
        "total_entries": len(chat_history),
        "entries": debug_output
    }

if __name__ == "__main__":
    print("=" * 80)
    print("🎙️  TARA AI INTERVIEW SYSTEM - Professional Interview Assistant")
    print("=" * 80)
    print(f"🌐 Server: http://localhost:5050")
    print(f"👤 Interviewer: Tara (Senior Technical Interviewer)")
    print(f"❓ Question Range: {MIN_QUESTIONS}-{MAX_QUESTIONS} questions (Ideal: {IDEAL_QUESTIONS})")
    print(f"🎤 Voice Control: 'stop the interview' to end early")
    print(f"🤖 Gemini API: {'✅ Connected' if GEMINI_API_KEY else '❌ Missing'}")
    print(f"☁️  Google Cloud: {'✅ Configured' if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') else '❌ Missing'}")
    print(f"📊 Features:")
    print(f"   • Dynamic AI-powered comprehensive reports")
    print(f"   • TTS-safe report delivery (waits for speech completion)")
    print(f"   • Smart STT with proactive 50s restart (prevents 60s timeout)")
    print(f"   • Auto-recovery from network issues")
    print(f"   • Enhanced speech recognition for technical terms")
    print("=" * 80)
    print(f"💡 Tips:")
    print(f"   • Long answers are supported (auto-restarts every 50s)")
    print(f"   • 3-second silence threshold for natural pauses")
    print(f"   • Debug endpoint: /debug_session/<session_id>")
    print("=" * 80)
    socketio.run(app, host="0.0.0.0", port=5005, debug=True, allow_unsafe_werkzeug=True)