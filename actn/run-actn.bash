#!/usr/bin/env bash

set -xu

function chk_virtualenv(){
    # a solution based on `virutalenv`

    #set -ex
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

    virtualenv -v --python=python2 p2-for-me
    source ./p2-for-me/bin/activate
    pip install numpy
    pip install matplotlib
}

function chk_venv(){
    # a solution based on `venv` as opposed to `virutalenv`

    #set -ex
    if [[  -d ./p3-for-me ]]
    then
    echo "venv exists"
    return 0
    fi

    echo "INSTALLING THEM"
    rm -rf p3-for-me || :

    # venv is shipped with python3
    #python3 -m venv -v --python=python3 p3-for-me
    python3 -m venv p3-for-me
    source ./p3-for-me/bin/activate

    python --version
    # Python 3.9.12

    #pip install numpy
    #pip install matplotlib

    # python -m pip install \
    #    --trusted-host files.pythonhosted.org \
    #    --trusted-host pypi.org \
    #    --trusted-host pypi.python.org \
    #    [--proxy ...] [--user] <packagename>
    #
    #python -m pip install
    #   --trusted-host files.pythonhosted.org \
    #   --trusted-host pypi.org \
    #   --trusted-host pypi.python.org --user \
    #      numpy


    # For trusted sources: see  https://stackoverflow.com/questions/49324802/pip-always-fails-ssl-verification

    python -m \
        pip install \
            --trusted-host pypi.python.org \
            --trusted-host files.pythonhosted.org \
            --trusted-host pypi.org \
            --upgrade pip

    #python -m \
        pip install \
            --trusted-host pypi.python.org \
            --trusted-host files.pythonhosted.org \
            --trusted-host pypi.org \
            numpy

        pip install \
            --trusted-host pypi.python.org \
            --trusted-host files.pythonhosted.org \
            --trusted-host pypi.org \
            matplotlib
}
# chk_virtualenv
chk_venv

echo "Main script"
source ./p3-for-me/bin/activate

python --version

# python action1.py

python action2.py
