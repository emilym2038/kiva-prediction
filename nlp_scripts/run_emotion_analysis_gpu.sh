#!/bin/bash -l
#$ -N emotion_analysis_gpu
#$ -P rise-phishing
#$ -pe omp 4
#$ -l gpus=1
#$ -q a100,h200,l40s,a40
#$ -l h_rt=24:00:00
#$ -j y
#$ -o /project/rise-phishing/kiva-prediction/nlp/emotion_analysis.qlog

cd /project/rise-phishing/kiva-prediction/nlp
/project/rise-phishing/kiva-prediction/.venv/bin/python emotion_analysis.py
