import json
import requests
from ..state import sessions, MIN_QUESTIONS, MAX_QUESTIONS, IDEAL_QUESTIONS
from ..config import get_env_vars


def get_llm_response(session_id, resume_text=None, job_description=None, question_type=None, user_response=None):
    env = get_env_vars()
    GEMINI_API_KEY = env['GEMINI_API_KEY']
    GEMINI_API_URL = env['GEMINI_API_URL']

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
            questions_asked = sum(1 for m in chat_history if 'interviewer' in m and 'resume' not in m)
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


