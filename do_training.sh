#!/bin/bash

# 주피터 실행시간 테스트
echo "주피터 실행 날짜: $(date +%Y)년   $(date +%m)월 $(date +%d)일 "
echo "주피어 실행 시간: $(date +%H) 시  $(date +%M) 분  $(date +%S)초"

# 주피터 노트북 실행 코드
nohup jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root &