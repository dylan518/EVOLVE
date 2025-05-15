#!/bin/bash



ENV_FILE="${1:-$(pwd)/keys.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌  Cannot find env-file: $ENV_FILE"; exit 1
fi

echo "[+] Running base setup script..."
chmod +x scripts/setup.sh
./scripts/setup.sh

git config --global user.name  "${GIT_USERNAME:-YourName}"
git config --global user.email "${GIT_EMAIL:-you@example.com}"

if ! command -v node &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
echo "$HF_TOKEN" | huggingface-cli login --token

