#!/usr/bin/env bash

VENVNAME= Computer_Vision02 

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# problems when installing from requirements.txt
pip install ipython
pip install jupyter
pip install networkx
pip install sklearn

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install requirements.txt

deactivate
echo "build $VENVNAME"
