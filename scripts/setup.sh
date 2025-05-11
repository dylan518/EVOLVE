python3.10 -m venv env
source env/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt \
  --no-build-isolation --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu118
