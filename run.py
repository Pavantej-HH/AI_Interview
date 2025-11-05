from ai_interview import create_app
from ai_interview.sockets.socketio import socketio


app = create_app()


if __name__ == "__main__":
    print("=" * 80)
    print("🎙️  TARA AI INTERVIEW SYSTEM - Professional Interview Assistant")
    print("=" * 80)
    print(f"🌐 Server: http://localhost:5050")
    print(f"👤 Interviewer: Tara (Senior Technical Interviewer)")
    print("=" * 80)
    socketio.run(app, host="0.0.0.0", port=5002, debug=True, allow_unsafe_werkzeug=True)


