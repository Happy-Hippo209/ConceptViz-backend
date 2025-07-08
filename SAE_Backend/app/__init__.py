# /root/autodl-tmp/learning/sae_backend/app/__init__.py

from flask import Flask
from app.routes import input_bp , explore_bp, validate_bp
from app.utils.errors import register_error_handlers

def create_app():
    app = Flask(__name__)
    
    # 注册蓝图
    app.register_blueprint(input_bp)
    app.register_blueprint(explore_bp)
    
    app.register_blueprint(validate_bp)
    
    # 注册错误处理
    register_error_handlers(app)
    
    return app