#!/bin/bash -l
#$ -N readability_analysis
#$ -P rise-phishing
#$ -pe omp 1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /project/rise-phishing/kiva-prediction/nlp/readability_analysis.qlog

cd /project/rise-phishing/kiva-prediction/nlp
/project/rise-phishing/kiva-prediction/.venv/bin/python readability_analysis.py
