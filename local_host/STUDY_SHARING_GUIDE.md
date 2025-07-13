# 🚀 AI Analysis Comparison Study - Sharing Guide

## ✅ **Quick Start (For You)**

### 1. Start the Server
**Windows:**
```bash
# Double-click: start_study.bat
# OR in Command Prompt:
python server.py
```

**Mac/Linux:**
```bash
# In Terminal:
./start_study.sh
# OR:
python3 server.py
```

### 2. Access Your Study
The server will automatically open your browser to:
- **HVA-X vs Highlights**: http://localhost:8000/hva-highlights-comparison.html
- **HVA-X vs Grad-CAM**: http://localhost:8000/hva-gradcam-comparison.html

## 🌐 **Sharing with Participants**

### Option 1: Local Network Sharing (Recommended)
If participants are on the same WiFi/network:

1. **Find your IP address:**
   - **Windows**: Open Command Prompt → type `ipconfig` → look for "IPv4 Address"
   - **Mac**: System Preferences → Network → select WiFi → Advanced → TCP/IP
   - **Linux**: Terminal → type `ip addr show` or `ifconfig`

2. **Share these URLs with participants:**
   ```
   HVA-X vs Highlights: http://YOUR_IP_ADDRESS:8000/hva-highlights-comparison.html
   HVA-X vs Grad-CAM:   http://YOUR_IP_ADDRESS:8000/hva-gradcam-comparison.html
   ```
   
   **Example:** If your IP is `192.168.1.100`:
   ```
   http://192.168.1.100:8000/hva-highlights-comparison.html
   http://192.168.1.100:8000/hva-gradcam-comparison.html
   ```

### Option 2: Internet Sharing (Advanced)
For remote participants, use **ngrok** (free):

1. **Install ngrok**: https://ngrok.com/download
2. **Start your study server** (as above)
3. **In a new terminal:**
   ```bash
   ngrok http 8000
   ```
4. **Share the ngrok URL** (looks like: `https://abc123.ngrok.io`)

### Option 3: File Sharing
Send participants the entire folder and have them run it locally:

1. **Zip the entire project folder** including:
   - `videos/` folder with all video files
   - `hva-highlights-comparison.html`
   - `hva-gradcam-comparison.html`
   - `server.py`
   - `start_study.bat` (Windows) or `start_study.sh` (Mac/Linux)

2. **Send instructions:**
   - Extract the zip file
   - Run the start script for their operating system
   - Open the study URLs in their browser

## 📊 **Collecting Results**

### Automatic Collection
- Results are automatically saved to the `results/` folder
- Each participant's data is saved as a separate CSV file
- Files are named with timestamp and study type

### Manual Collection
- Participants can also download CSV files directly
- Files are saved to their Downloads folder
- Ask them to email you the CSV files

## 🎥 **Video Troubleshooting**

### If Videos Don't Play:
1. **Check file structure:**
   ```
   your-project/
   ├── videos/
   │   ├── demo_video.mp4 ✓
   │   ├── HL_0.mp4 ✓
   │   ├── HL_1.mp4 ✓
   │   ├── HL_2.mp4 ✓
   │   ├── HL_3.mp4 ✓
   │   ├── HL_4.mp4 ✓
   │   └── gradcam_BreakoutNoFrameskip-v4_dqn.mp4 ✓
   └── server.py
   ```

2. **Ensure server is running** (don't just open HTML files directly)

3. **Try different browsers** (Chrome, Firefox, Safari, Edge)

4. **Check browser console** for error messages (F12 → Console tab)

## 🔧 **Common Issues & Solutions**

### "Port already in use"
- The server will automatically try the next available port
- Or manually specify a different port: `python server.py 8001`

### "Connection refused"
- Make sure the server is still running
- Check that the port number matches in the URL

### Videos load slowly
- The Grad-CAM video has been optimized (now 2MB, H.264 format)
- Should load faster and work in more browsers
- If still having issues, try a different browser

### Results not saving
- Check that the `results/` folder exists
- Ensure the server has write permissions
- Check browser console for errors

## 📋 **Study Instructions for Participants**

**Send this to your participants:**

---

### AI Analysis Comparison Study - Participant Instructions

Thank you for participating in our research study!

**Time Required:** ~15 minutes

**What You'll Do:**
1. Watch a video of an AI playing a game
2. Review two different AI analysis methods
3. Rate each method using questionnaires
4. Your responses help us improve AI explanation systems

**Technical Requirements:**
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Stable internet connection
- Audio capability (optional, videos have no sound)

**Study Links:**
- **Study A**: [INSERT YOUR URL HERE]
- **Study B**: [INSERT YOUR URL HERE]

**Instructions:**
1. Click on one of the study links above
2. Choose your preferred language (English/中文)
3. Follow the step-by-step instructions
4. Complete all evaluation questions
5. Download your results at the end

**Need Help?**
- Contact: [YOUR EMAIL]
- Technical issues: Try refreshing the page or using a different browser

---

## 📈 **Data Analysis**

The CSV files contain:
- **Method**: Which analysis method was evaluated (HVA-X, Highlights, Grad-CAM)
- **Dimension**: The evaluation criteria (Clarity, Understandable, Completeness, Satisfaction, Useful, Accuracy, Improvement, Preference)
- **Score**: Participant's rating (1-7 scale)
- **Metadata**: Language, randomization order, completion time

**Note**: CSV files now use English-only headers to avoid character encoding issues, making them compatible with all spreadsheet software.

Import into Excel, R, Python, or your preferred analysis tool for statistical analysis.

## 🔐 **Privacy & Ethics**

- No personal information is collected
- Only study responses are recorded
- Participants can withdraw at any time
- Data is stored locally on your computer
- Follow your institution's IRB guidelines 