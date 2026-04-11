import sys
import os
import uvicorn

# Add the root directory to sys.path so we can import app.py
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from app import app
except ImportError:
    # Fallback if app.py is not in the expected location relative to this file
    sys.path.append(".")
    from app import app

def main():
    """Main entry point for the server script."""
    print("Starting AI Email Triage Server from server/app.py...")
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
