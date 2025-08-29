cd ./..

echo " "
echo " "
echo "${BYellow}${On_Black}Freezing virtual environment...${NC}"
source env/bin/activate

pip freeze > ./Commands/requirements.txt