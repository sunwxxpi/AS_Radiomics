import sys
import os
import datetime
import logging

_root_handler = None

# 이 로거들은 자기 stderr 핸들러로 한 번, root 로 전파돼 또 한 번 찍는다. 핸들러를 떼면 log.txt 에도 남는 root 쪽 한 줄만 남는다.
# root 를 INFO 로 낮춘 탓에 새로 새어 나오는 진행 로그는 레벨로 막는다. radiomics 는 케이스별 특징 추출 실패 사유만 남기면 된다.
THIRD_PARTY_LOG_LEVELS = {
    'radiomics': logging.ERROR,
    'nibabel': logging.WARNING,
    'nibabel.global': logging.WARNING,
}

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
    global _root_handler

    # 출력 디렉토리에서 모드 추출
    mode = None
    if "_binary" in output_dir:
        mode = "binary"
    elif "_multi" in output_dir:
        mode = "multi"
    
    log_file_path = os.path.join(output_dir, 'log.txt')
    sys.stdout = OutputLogger(log_file_path, mode)

    # logging 출력도 OutputLogger 를 거쳐야 화면과 로그 파일에 함께 남는다.
    root_logger = logging.getLogger()
    if _root_handler is not None:
        root_logger.removeHandler(_root_handler)
    _root_handler = logging.StreamHandler(sys.stdout)
    _root_handler.setLevel(logging.INFO)
    _root_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    root_logger.addHandler(_root_handler)
    # root 기본 레벨이 WARNING 이라 낮추지 않으면 INFO 레코드가 핸들러까지 오지 않는다.
    root_logger.setLevel(logging.INFO)

    for name, level in THIRD_PARTY_LOG_LEVELS.items():
        third_party = logging.getLogger(name)
        for handler in list(third_party.handlers):
            third_party.removeHandler(handler)
        third_party.setLevel(level)

    return sys.stdout

def close_logging():
    """로깅 종료"""
    global _root_handler

    # 파일이 닫힌 뒤 핸들러가 쓰지 않도록 먼저 떼어낸다.
    if _root_handler is not None:
        logging.getLogger().removeHandler(_root_handler)
        _root_handler = None

    if isinstance(sys.stdout, OutputLogger):
        # 로그 파일 쓰기가 실패해도 stdout 은 되돌린다. 안 되돌리면 인터프리터 종료 때 다시 터져 원래 예외를 덮는다.
        try:
            sys.stdout.close()
        finally:
            sys.stdout = sys.stdout.terminal