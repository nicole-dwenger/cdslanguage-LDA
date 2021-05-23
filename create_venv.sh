#!/usr/bin/env bash

VENVNAME=venv_LDA

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# requirements
test -f requirements.txt && pip install -r requirements.txt

# load spacy language model
python3 -m spacy download en_core_web_sm

deactivate
echo "build $VENVNAME"
