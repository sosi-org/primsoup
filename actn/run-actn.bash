#!/usr/bin/env bash
function chk(){
#set -ex
#[[ ! -f ./p2-for-me ]]
if [[  -d ./p2-for-me ]]
then
# exists
return 0
#else
# does not exist
#return 0
fi

echo "INSTALLING THEM"

# brew install virtualenv
# virtualenv -v --python=python3 p3
virtualenv -v --python=python2 p2-for-me
# Although it seems python 2
source ./p2-for-me/bin/activate
pip install numpy
pip install matplotlib
}

chk

echo "Main script"
source ./p2-for-me/bin/activate

python action1.py
