# DeepGuard-AI Server Instructions

## 1. Install Dependencies
Open a terminal in the `HACKSRM` directory and run:

```sh
pip install fastapi uvicorn websockets
# OR if using python3
python3 -m pip install fastapi uvicorn websockets
```

## 2. Run the Backend Server
Start the server that analyzes the video/audio streams:

```sh
python3 -m uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000
```

You should see: `Uvicorn running on http://127.0.0.1:8000`

## 3. Test the End-to-End Flow
1. **Reload Extension**: Go to `chrome://extensions`, find DeepGuard-AI, and click the refresh/reload icon.
2. **Open Test Page**: Open the file `test_call.html` in Chrome.
3. **Start Call**: Click "🔴 Start Mock Call".
4. **Check Logs**: Right-click > Inspect > Console.
   - You should see: `[DeepGuard] WebSocket Connected! Secure Pipeline Established.`
   - And streaming logs: `[DeepGuard] Frame Safe. Risk Score: ...`

## Troubleshooting
- If you see `WebSocket connection failed`, ensure the server is running on port 8000.
- If you don't see video, ensure camera permissions are granted.
