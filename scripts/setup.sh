python3.10 -m venv env
source env/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt \
  --no-build-isolation --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu118


# install git lfs
sudo apt update
sudo apt install git-lfs -y
git lfs install
git lfs pull

#vllm reqs
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y binutils 