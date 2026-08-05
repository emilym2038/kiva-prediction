#!/bin/bash -l
#$ -N emotion_analysis_cpu
#$ -P rise-phishing
#$ -pe omp 8
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /project/rise-phishing/kiva-prediction/nlp/emotion_analysis.qlog

cd /project/rise-phishing/kiva-prediction/nlp
export OMP_NUM_THREADS=8
/project/rise-phishing/kiva-prediction/.venv/bin/python emotion_analysis.py
