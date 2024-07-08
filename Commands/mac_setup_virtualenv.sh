#!/bin/bash

# Define the Python version you want to install
PYTHON_VERSION="3.12.1"

# Rest of the script remains the same
DIR=$(dirname "${BASH_SOURCE[0]}")
DIR=$(realpath "${DIR}")
BYellow='\033[1;93m'
On_Black='\033[40m'
NC='\033[0m' 
cd "$DIR"
cd ./..
 
check_dir="/Users/$USER/.pyenv"
if [ ! -d "$check_dir" ]; then
  echo -e "${BYellow}${On_Black}Installing pyenv since it was not found...${NC}"
  curl https://pyenv.run | bash
fi
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
 
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
 
echo " "
echo " "
echo "${BYellow}${On_Black}Reloading RC files...${NC}"
source ~/.bashrc
source ~/.zshrc
 
echo " "
echo " "
echo "${BYellow}${On_Black}Installing Python $PYTHON_VERSION…${NC}"
pyenv install "$PYTHON_VERSION"
 
echo " "
echo " "
echo "${BYellow}${On_Black}Installing virtual environment...${NC}"
~/.pyenv/versions/"$PYTHON_VERSION"/bin/python -m pip install virtualenv
 
echo " "
echo " "
echo "${BYellow}${On_Black}Creating virtual environment...${NC}"
~/.pyenv/versions/"$PYTHON_VERSION"/bin/python -m venv env
 
echo " "
echo " "
echo "${BYellow}${On_Black}Activating virtual environment...${NC}"
source env/bin/activate
 
echo " "
echo " "
echo "${BYellow}${On_Black}Installing virtual environment packages...${NC}"
pip install -r ./Commands/requirements_mac.txt
 
echo " "
echo " "
echo "${BYellow}${On_Black}Installing local packages...${NC}"
CUR_DIR=$(pwd)
echo $CUR_DIR > env/lib/python"$PYTHON_VERSION"/site-packages/localpackages.pth
