
#!/bin/bash

echo "Generating .env with param dev for service python-air-quality"

aws ssm get-parameters-by-path --path "/dev/python-air-quality/" --with-decryption   --region ap-south-1 --query="Parameters[*].[Name, Value]"   --output text |
 while read line
 do
    name=$(echo ${line}} | cut -f 1 -d ' ' | sed -e "s/\/dev\/python-air-quality\///g")
    value=$(echo ${line} | cut -f 2 -d ' ')
    echo "${name}=${value}" >> .env
 done
ls -lart
python main.py
