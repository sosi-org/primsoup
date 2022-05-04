#!/usr/bin/env bash

# brew install virtualenv
# virtualenv -v --python=python3 p3
virtualenv -v --python=python2 p2-for-me
# Although it seems python 2
source ./p2-for-me/bin/activate
pip install numpy
pip install matplotlib

python actin1.py
