#!/usr/bin/env nextflow

params.nSeeds = 1
params.nDraws = 100

chains = [1]
seeds = (1..params.nSeeds).collect{it}


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
    val 'nDraws' from params.nDraws
    file 'rootpath.txt' from rootPath
    file pyModel from pyModels
    file '.virtualenvs' from venv
    each seed from seeds
    each chain from chains
  output:
    file 'results/inference' into posteriorSamples
  publishDir deliverableDir, mode: 'copy', overwrite: true
"""
source .virtualenvs/mbench/bin/activate
mkdir -p results/inference/samples
python3.6 $pyModel $chain $nDraws $seed `cat rootpath.txt`
mv *.csv results/inference/samples
"""
}

process analysisCode { 
  input:
    val gitRepoName from 'nedry'
    val gitUser from 'alexandrebouchard'
    val codeRevision from '4c6ddf0de0027ad88d73ef6634d1e70cc9f94bfe'
    val snapshotPath from "${System.getProperty('user.home')}/w/nedry"
  output:
    file 'code' into analysisCode
  script:
    template 'buildRepo.sh'
}

analysisCode.into {
  essCode
  aggregateEssCode
  aggregateDensityCode
} 

process computeESS {
  conda 'csvkit'
  input:
    file 'results/inference' from posteriorSamples
    file essCode
  output:
    file 'results' into essResults
  publishDir deliverableDir, mode: 'copy', overwrite: true
  """
  mkdir results/ess
  outputDir=results/ess
  allESSName=ess-all.csv
  echo "ess" >> \$outputDir/\$allESSName
  for sampleFile in results/inference/samples/*.csv; do
    output=\$outputDir/ess-\$(basename \$sampleFile)
    code/bin/ess  \
      --experimentConfigs.saveStandardStreams false \
      --experimentConfigs.managedExecutionFolder false \
      --inputFile \$sampleFile \
      --burnInFraction 0.5 \
      --moment 2 \
      --output \$output
    csvcut -c ess \$output | grep -v ess >> \$outputDir/\$allESSName
  done
  """
}


