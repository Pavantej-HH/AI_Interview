import json
import time
from datetime import datetime
import requests
from ..state import sessions
from ..config import get_env_vars
from .tts import synthesize_speech
from ..sockets.socketio import socketio


def generate_dynamic_report(session_id):
    try:
        chat_history = sessions[session_id]['chat_history']
        if len(chat_history) < 2:
            return None

        resume = chat_history[0].get('resume', 'N/A')
        jd = chat_history[0].get('jd', 'N/A')
        question_type = chat_history[0].get('question_type', 'technical')

        qa_pairs = []
        scores = []

        i = 1
        qa_number = 0

        while i < len(chat_history):
            current_entry = chat_history[i]
            if 'interviewer' in current_entry and 'resume' not in current_entry:
                qa_number += 1
                question = current_entry.get('interviewer', '')
                answer = ''
                if i + 1 < len(chat_history) and 'candidate' in chat_history[i + 1]:
                    answer = chat_history[i + 1].get('candidate', '')
                    evaluation = ''
                    score = 0
                    if i + 2 < len(chat_history) and 'interviewer' in chat_history[i + 2]:
                        next_question_entry = chat_history[i + 2]
                        evaluation = next_question_entry.get('evaluation', '')
                        score = next_question_entry.get('score', 0)
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'score': score,
                        'evaluation': evaluation if evaluation else 'Response received and being evaluated'
                    })
                    if score > 0:
                        scores.append(score)
            i += 1

        avg_score = round(sum(scores) / len(scores), 2) if scores else 0
        if not scores or avg_score == 0:
            return generate_minimal_report(session_id, qa_pairs, resume, jd, question_type)

        conversation_transcript = ""
        for idx, qa in enumerate(qa_pairs, 1):
            conversation_transcript += f"\n{'='*60}\n"
            conversation_transcript += f"QUESTION {idx}:\n{qa['question']}\n\n"
            conversation_transcript += f"CANDIDATE ANSWER:\n{qa['answer']}\n\n"
            conversation_transcript += f"INTERVIEWER EVALUATION:\n{qa['evaluation']}\n"
            conversation_transcript += f"SCORE: {qa['score']}/10\n"
            conversation_transcript += f"{'='*60}\n"

        env = get_env_vars()
        headers = {"Content-Type": "application/json", "x-goog-api-key": env['GEMINI_API_KEY']}
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
  "key_strengths": ["...", "...", "..."],
  "areas_for_improvement": ["...", "...", "..."],
  "technical_assessment": {"depth_of_knowledge": 7, "problem_solving": 6, "communication": 5, "experience_relevance": 6},
  "resume_alignment": "2-3 sentences...",
  "job_fit": "2-3 sentences...",
  "next_steps": "..."
}}

CRITICAL RULES:
1. Be honest and specific
2. If performance was weak, say so professionally
3. technical_assessment scores are 1-10 integers
4. strengths and improvements lists are exactly 3 items each
5. Return ONLY the JSON, no other text"""

        payload = {"contents": [{"parts": [{"text": analysis_prompt}]}]}
        response = requests.post(env['GEMINI_API_URL'], headers=headers, json=payload, timeout=45)

        if response.status_code == 200:
            response_data = response.json()
            ai_analysis = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if ai_analysis.startswith("```json"):
                ai_analysis = ai_analysis[7:]
            elif ai_analysis.startswith("```"):
                ai_analysis = ai_analysis[3:]
            if ai_analysis.endswith("```"):
                ai_analysis = ai_analysis[:-3]
            ai_analysis = ai_analysis.strip()
            try:
                analysis_data = json.loads(ai_analysis)
                if 'technical_assessment' in analysis_data:
                    for key in analysis_data['technical_assessment']:
                        val = analysis_data['technical_assessment'][key]
                        if isinstance(val, str):
                            try:
                                analysis_data['technical_assessment'][key] = int(val)
                            except:
                                analysis_data['technical_assessment'][key] = int(avg_score)
                        analysis_data['technical_assessment'][key] = max(1, min(10, int(analysis_data['technical_assessment'][key])))
            except json.JSONDecodeError:
                analysis_data = generate_fallback_analysis(avg_score, len(qa_pairs), qa_pairs)
        else:
            analysis_data = generate_fallback_analysis(avg_score, len(qa_pairs), qa_pairs)

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

        return report

    except Exception as e:
        print(f"[REPORT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return generate_fallback_report(session_id)


def generate_minimal_report(session_id, qa_pairs, resume, jd, question_type):
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
    short_answers = sum(1 for qa in qa_pairs if len(qa['answer'].split()) < 15)
    unclear_answers = sum(1 for qa in qa_pairs if any(word in qa['answer'].lower() for word in ["don't remember", "don't know", "not sure", "maybe", "i think"]))

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

    strengths = []
    improvements = []
    if qa_count >= 8:
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
        "resume_alignment": f"Based on {qa_count} questions, candidate's responses {'align well with' if avg_score >= 7 else 'show discrepancies from'} resume claims.",
        "job_fit": f"Candidate demonstrates {'strong' if avg_score >= 7.5 else 'partial' if avg_score >= 6 else 'limited'} alignment with job requirements.",
        "next_steps": "Proceed to technical round with senior engineer" if avg_score >= 7.5 else "Consider additional screening" if avg_score >= 6 else "Thank candidate for their time, not proceeding"
    }


def generate_fallback_report(session_id):
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


