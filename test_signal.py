# -*- coding: utf-8 -*-
"""신호 감지 테스트 스크립트"""
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_detector import test_detector

if __name__ == "__main__":
    test_detector()

