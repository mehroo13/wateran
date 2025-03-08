from flask import Flask, send_file, send_from_directory, request
import requests
from werkzeug.middleware.proxy_fix import ProxyFix
import os

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/ads.txt')
def ads_txt():
    return 'google.com, pub-2264561932019289, DIRECT, f08c47fec0942fa0', 200, {'Content-Type': 'text/plain'}

@app.route('/robots.txt')
def robots_txt():
    return '''User-agent: *
Allow: /
Sitemap: https://your-domain.com/sitemap.xml''', 200, {'Content-Type': 'text/plain'}

@app.route('/sitemap.xml')
def sitemap():
    return '''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://your-domain.com/</loc>
        <changefreq>daily</changefreq>
        <priority>1.0</priority>
    </url>
</urlset>''', 200, {'Content-Type': 'application/xml'}

@app.route('/streamlit/<path:path>')
def proxy_streamlit(path):
    try:
        response = requests.get(f'http://localhost:8501/{path}')
        return response.content, response.status_code, response.headers.items()
    except requests.RequestException:
        return "Streamlit app is not available", 503

if __name__ == '__main__':
    # Check if running in production
    if os.environ.get('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=80, ssl_context='adhoc')
    else:
        app.run(host='0.0.0.0', port=80) 
