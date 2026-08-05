#!/bin/bash -l
#$ -N entity_analysis
#$ -P rise-phishing
#$ -pe omp 8
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /project/rise-phishing/kiva-prediction/nlp/entity_analysis.qlog

cd /project/rise-phishing/kiva-prediction/nlp
/project/rise-phishing/kiva-prediction/.venv/bin/python entity_analysis.py
