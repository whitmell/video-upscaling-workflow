# from flask import Flask

# def create_app():
#     app = Flask(__name__)
    
#     # Load configuration
#     app.config.from_mapping(
#         SECRET_KEY='your_secret_key',
#     )

#     # Register blueprints
#     from .routes import main as main_blueprint
#     app.register_blueprint(main_blueprint)

#     return app