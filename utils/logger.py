import sys
import os
import datetime
import logging

class OutputLogger:
    """콘솔과 파일에 동시에 출력하는 로거"""
    
    def __init__(self, log_file_path, mode=None):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        mode_str = f" (모드: {mode})" if mode else ""
        self.log_file.write(f"===== 실행 시작{mode_str}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n\n")
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.write(f"\n===== 실행 종료: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        self.log_file.close()

def setup_logging(output_dir):
    """로깅 설정 및 시작"""
    # 출력 디렉토리에서 모드 추출
    mode = None
    if "_binary" in output_dir:
        mode = "binary"
    elif "_multi" in output_dir:
        mode = "multi"
    
    log_file_path = os.path.join(output_dir, 'log.txt')
    sys.stdout = OutputLogger(log_file_path, mode)
    
    # Pyradiomics 로거 설정
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)
    
    return sys.stdout

def close_logging():
    """로깅 종료"""
    if isinstance(sys.stdout, OutputLogger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal