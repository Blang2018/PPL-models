#!/usr/bin/env nextflow

deliverableDir = 'deliverables/' + workflow.scriptName.replace('.nf','')

process createVirtualEnv {
  cache true
  output:
    file 'rootpath.txt' into rootPath
    file '.virtualenvs' into venv
"""
#!/bin/bash
mkdir -p .virtualenvs/mbench
python3.6 -m venv ./.virtualenvs/mbench
source ./.virtualenvs/mbench/bin/activate
pip install pymc3 matplotlib
echo `pwd`/../../../ > rootpath.txt
"""
}

pyModels = Channel.fromPath('models/*.py')

process performPyMC3Inference {
  input:
    file 'rootpath.txt' from rootPath
    file pyModel from pyModels
    file '.virtualenvs' from venv
  output:
    file '*.csv' into posteriorSamples
  publishDir deliverableDir, mode: 'copy', overwrite: true
"""
source .virtualenvs/mbench/bin/activate
mkdir deliverables
python3.6 $pyModel 1 1000 `cat rootpath.txt`
mv deliverables/*.csv ./
"""
}

