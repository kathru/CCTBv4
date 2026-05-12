import hmac
import hashlib
import subprocess
from flask import Flask, request, abort

app = Flask(__name__)

# Segredo compartilhado com o GitHub (configure via env var em producao)
WEBHOOK_SECRET = "trading-bot-webhook-2026"

def verify_signature(payload, signature):
    mac = hmac.new(WEBHOOK_SECRET.encode(), payload, hashlib.sha256)
    expected = f"sha256={mac.hexdigest()}"
    return hmac.compare_digest(expected, signature or "")

@app.route("/webhook", methods=["POST"])
def webhook():
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(request.data, sig):
        abort(401)

    branch = request.json.get("ref", "")
    if branch != "refs/heads/master":
        return "Ignored (not master)", 200

    result = subprocess.run(
        "cd /home/ubuntu/claude-code-trading-bot && "
        "git pull origin master && "
        "source venv/bin/activate && "
        "pip install -r requirements.txt -q && "
        "sudo systemctl restart tradingbot",
        shell=True, capture_output=True, text=True, executable="/bin/bash"
    )
    print(result.stdout)
    print(result.stderr)
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
