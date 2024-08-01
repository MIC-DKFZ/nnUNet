sudo apt update -y
sudo apt install tmux unzip -y
sudo ln -s /usr/bin/python3 /usr/bin/python
sudo apt install python3.12-venv -y
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws*

sudo apt install python3-pip -y
python3 -m venv ~/.virtualenvs/myenv
source ~/.virtualenvs/myenv/bin/activate
pip install virtualenvwrapper
echo -e "\nexport WORKON_HOME=\$HOME/.virtualenvs\nexport VIRTUALENVWRAPPER_PYTHON=\$HOME/.virtualenvs/myenv/bin/python\nsource \$HOME/.virtualenvs/myenv/bin/virtualenvwrapper.sh" >> ~/.bashrc
echo -e "\nexport nnUNet_raw=\$HOME/nnUNetData/nnUNet_raw\nexport nnUNet_preprocessed=\$HOME/nnUNetData/nnUNet_preprocessed\nexport nnUNet_results=\$HOME/nnUNetData/nnUNet_results" >> ~/.bashrc
source ~/.bashrc

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update -y
sudo apt install python3.11 -y

sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update -y
sudo apt install nvidia-driver-530 -y
# sudo reboot

# Visit: https://developer.nvidia.com/cuda-downloads
# Replace <distro> and <version> with the appropriate values
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda-repo-ubuntu2404-12-5-local_12.5.1-555.42.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-5-local_12.5.1-555.42.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update -y
sudo apt-get -y install cuda-toolkit-12-5
echo -e "\nexport PATH=/usr/local/cuda/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
# Installs cuDNN
wget https://developer.download.nvidia.com/compute/cudnn/9.2.1/local_installers/cudnn-local-repo-ubuntu2204-9.2.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.2.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.2.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt update -y
sudo apt install cudnn-cuda-12 -y


mkvirtualenv -p /usr/bin/python3.11 nnUNet
# workon nnUNet
git clone https://github.com/bunkerhillhealth/nnUNet.git
cd nnUNet; pip install --editable .; cd ~/

sudo apt install -y gcc g++ make libpq-dev

# https://stackoverflow.com/q/77490435
pip install "cython<3.0.0" wheel
pip install "pyyaml==5.4.1" --no-build-isolation

# Install blib
git clone https://github.com/bunkerhillhealth/bunkerhill
python -m pip install conan==1.59 --editable bunkerhill/blib

pip install line_profiler

