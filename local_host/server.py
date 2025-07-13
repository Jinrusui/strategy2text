#!/usr/bin/env python3
"""
Simple HTTP Server for AI Analysis Comparison Study
Serves HTML files and videos with proper CORS headers
"""

import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse, parse_qs
import json
import webbrowser
from datetime import datetime

class StudyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        # Handle CSV upload requests
        if self.path == '/upload-csv':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                # Save CSV file locally
                filename = data.get('fileName', f'study_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                csv_content = data.get('csvContent', '')
                
                # Create results directory if it doesn't exist
                os.makedirs('results', exist_ok=True)
                
                # Save the file with UTF-8 BOM for better Excel compatibility
                filepath = os.path.join('results', filename)
                with open(filepath, 'w', encoding='utf-8-sig') as f:
                    f.write(csv_content)
                
                print(f"✅ Saved study results: {filepath}")
                
                # Send success response
                response = {
                    'success': True,
                    'message': 'File saved successfully',
                    'filename': filename
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                print(f"❌ Error saving file: {e}")
                error_response = {
                    'success': False,
                    'error': str(e)
                }
                
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            super().do_POST()

    def log_message(self, format, *args):
        # Custom logging
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {format % args}")

def start_server(port=8000):
    """Start the HTTP server"""
    
    # Change to the directory containing the files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    try:
        with socketserver.TCPServer(("", port), StudyHTTPRequestHandler) as httpd:
            print(f"""
🚀 AI Analysis Comparison Study Server Started!

📋 Study URLs:
   HVA-X vs Highlights: http://localhost:{port}/hva-highlights-comparison.html
   HVA-X vs Grad-CAM:   http://localhost:{port}/hva-gradcam-comparison.html

📁 Files being served from: {os.getcwd()}
📊 Results will be saved to: {os.path.join(os.getcwd(), 'results')}

🌐 Share these URLs with participants:
   - Make sure this server is running
   - Participants can access via your IP address if on same network
   - For remote access, consider using ngrok or similar service

⚠️  Keep this terminal window open while conducting the study
🛑 Press Ctrl+C to stop the server
            """)
            
            # Try to open the browser automatically
            try:
                webbrowser.open(f'http://localhost:{port}/hva-highlights-comparison.html')
            except:
                pass
            
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Port {port} is already in use. Trying port {port + 1}...")
            start_server(port + 1)
        else:
            print(f"❌ Error starting server: {e}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default port 8000.")
    
    start_server(port) 