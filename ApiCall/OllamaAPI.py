import os
import sys
import json
import requests
from datetime import datetime

# Default parameters (can be overridden by environment variables or CLI args)
hostApi = os.environ.get("HOST_API", "http://localhost:11434")
model = os.environ.get("MODEL", "phi3:mini")
prompt = os.environ.get("PROMPT", "What is the capital of France?")
printRaw = os.environ.get("PRINT_RAW", "false").lower() == "true"
enableLogging = os.environ.get("ENABLE_LOGGING", "false").lower() == "true"

# Parse CLI args (optional, simple)
for arg in sys.argv[1:]:
    if arg.startswith("--hostApi="): hostApi = arg.split("=",1)[1]
    elif arg.startswith("--model="): model = arg.split("=",1)[1]
    elif arg.startswith("--prompt="): prompt = arg.split("=",1)[1]
    elif arg == "--printRaw": printRaw = True
    elif arg == "--enableLogging": enableLogging = True

logFile = os.path.join(os.path.dirname(__file__), "llm.log")

def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logFile, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")

if enableLogging:
    write_log(f"Run: hostApi='{hostApi}', model='{model}', prompt='{prompt}', printRaw={printRaw}")

body = {"model": model, "prompt": prompt}

try:
    resp = requests.post(f"{hostApi}/api/generate", json=body)
    if enableLogging:
        write_log(f"STATUS CODE: {resp.status_code}")
    print(f"STATUS CODE: {resp.status_code}")
    responseString = resp.text
    fullResponse = ""
    for line in responseString.splitlines():
        if line.strip():
            try:
                j = json.loads(line)
                if "response" in j:
                    fullResponse += j["response"]
            except Exception:
                pass
    if printRaw:
        print(responseString)
        if enableLogging:
            write_log(f"RAW RESPONSE: {responseString[:200]}...")
    else:
        print(fullResponse)
        if enableLogging:
            write_log(f"FULL RESPONSE: {fullResponse[:200]}...")
except Exception as e:
    print("ERROR:")
    print(e)
    if enableLogging:
        write_log(f"ERROR: {e}")
