#!/usr/bin/env python3
"""
Prepare Netlify Deployment Script
This script creates a deployment-ready folder for Netlify hosting
"""

import os
import shutil
from pathlib import Path

def create_deployment_folder():
    """Create a deployment folder with all necessary files for Netlify"""
    
    # Create deployment directory
    deployment_dir = Path("netlify-deployment")
    if deployment_dir.exists():
        shutil.rmtree(deployment_dir)
    deployment_dir.mkdir()
    
    # Create videos subdirectory
    videos_dir = deployment_dir / "videos"
    videos_dir.mkdir()
    
    # Files to copy
    files_to_copy = [
        "hva-highlights-comparison.html",
        "hva-gradcam-comparison.html"
    ]
    
    video_files = [
        "demo.mp4",
        "HL_0.mp4", 
        "HL_1.mp4",
        "HL_2.mp4",
        "HL_3.mp4", 
        "HL_4.mp4",
        "gradcam_video.mp4"
    ]
    
    print("🚀 Preparing Netlify deployment folder...")
    
    # Copy HTML files
    for file in files_to_copy:
        if Path(file).exists():
            shutil.copy2(file, deployment_dir / file)
            print(f"✅ Copied {file}")
        else:
            print(f"❌ Warning: {file} not found")
    
    # Copy video files
    videos_source = Path("videos")
    if videos_source.exists():
        for video_file in video_files:
            source_path = videos_source / video_file
            if source_path.exists():
                shutil.copy2(source_path, videos_dir / video_file)
                print(f"✅ Copied videos/{video_file}")
            else:
                print(f"❌ Warning: videos/{video_file} not found")
    else:
        print("❌ Warning: videos/ directory not found")
    
    # Create index.html (landing page)
    index_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HVA-X Research Study</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            line-height: 1.6;
            color: #333;
        }
        .container {
            text-align: center;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .study-links {
            display: flex;
            gap: 30px;
            justify-content: center;
            flex-wrap: wrap;
            margin: 40px 0;
        }
        .study-link {
            display: inline-block;
            padding: 15px 30px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 500;
            transition: background 0.3s;
        }
        .study-link:hover {
            background: #2980b9;
        }
        .info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 30px 0;
        }
        .footer {
            margin-top: 50px;
            font-size: 14px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>HVA-X Research Study</h1>
        
        <div class="info">
            <p>Welcome to our research study comparing different AI explanation methods.</p>
            <p>Please click on one of the study links below to participate.</p>
        </div>
        
        <div class="study-links">
            <a href="hva-highlights-comparison.html" class="study-link">
                HVA-X vs Highlights Study
            </a>
            <a href="hva-gradcam-comparison.html" class="study-link">
                HVA-X vs Grad-CAM Study
            </a>
        </div>
        
        <div class="info">
            <p><strong>Instructions:</strong></p>
            <p>• Each study takes approximately 15-20 minutes to complete</p>
            <p>• Please use a desktop or laptop computer for the best experience</p>
            <p>• Make sure your audio is enabled for the video content</p>
            <p>• You can participate in both studies if you wish</p>
        </div>
        
        <div class="footer">
            <p>Thank you for participating in our research!</p>
        </div>
    </div>
</body>
</html>'''
    
    with open(deployment_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_content)
    print("✅ Created index.html landing page")
    
    # Create _redirects file for Netlify (optional)
    redirects_content = '''# Netlify redirects file
# Redirect root to index.html
/  /index.html  200
'''
    
    with open(deployment_dir / "_redirects", "w") as f:
        f.write(redirects_content)
    print("✅ Created _redirects file")
    
    print(f"\n🎉 Deployment folder ready: {deployment_dir.absolute()}")
    print("\nNext steps:")
    print("1. Visit https://www.netlify.com")
    print("2. Sign up for a free account")
    print("3. Drag and drop the 'netlify-deployment' folder to deploy")
    print("4. Get your public URLs and share with participants!")
    
    # Show folder structure
    print(f"\nFolder structure:")
    for root, dirs, files in os.walk(deployment_dir):
        level = root.replace(str(deployment_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

if __name__ == "__main__":
    create_deployment_folder() 