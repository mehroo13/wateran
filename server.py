from flask import Flask, send_file
import requests
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/streamlit/<path:path>')
def proxy_streamlit(path):
    response = requests.get(f'http://localhost:8501/{path}')
    return response.content, response.status_code, response.headers.items()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80) 