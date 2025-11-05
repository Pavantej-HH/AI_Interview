import os
from flask import Flask
from flask_cors import CORS
from .config import load_config
from .sockets.socketio import socketio


def create_app() -> Flask:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(base_dir, '..', 'templates')

    app = Flask(__name__, template_folder=templates_path)

    load_config(app)

    CORS(app, resources={r"/*": {"origins": "*"}})

    socketio.init_app(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=False,
        engineio_logger=False,
        ping_timeout=60,
        ping_interval=25,
    )

    # Register Flask routes
    from .routes.main import bp as main_bp
    app.register_blueprint(main_bp)

    # Register Socket.IO events (import binds handlers)
    from .sockets import events  # noqa: F401

    return app


