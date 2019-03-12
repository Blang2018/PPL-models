#!/usr/bin/env nextflow

process createVirtualEnv {
  cache true
  output:
    file 'venvpath.txt' into venvPath
    file 'modelspath.txt' into modelsPath
"""
#!/bin/bash
mkdir .virtualenvs
python3 -m venv ./.virtualenvs
source ./.virtualenvs/bin/activate
pip3 install pymc3
pip3 install matplotlib
echo `pwd`/.virtualenvs/bin/ > venvpath.txt
echo `pwd`/../../../ > modelspath.txt
"""
}


process runPyMC3 {
  input:
    file 'venvpath.txt' from venvPath
    file 'modelspath.txt' from modelsPath
  output:
    file '*.csv'
"""
source `cat venvpath.txt`activate
mkdir deliverables
python3 `cat modelspath.txt`challenger.py 1 1000 `cat modelspath.txt`
"""
}
