from flask import Blueprint, render_template
from ..state import sessions


bp = Blueprint('main', __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/health")
def health():
    return {"status": "ok", "sessions": len(sessions)}


@bp.route("/debug_session/<session_id>")
def debug_session(session_id):
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


