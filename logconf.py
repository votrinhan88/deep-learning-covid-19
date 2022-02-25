import logging
import logging.handlers
from datetime import datetime
from pytz import timezone, utc

def initLogger(logger, project_path):
    log_path = project_path + 'logs/log.log'
    
    # Format logger
    logger.setLevel(logging.DEBUG)
    def customTime(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("Asia/Ho_Chi_Minh")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()
    logging.Formatter.converter = customTime

    format_str = '%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)

    # Removing existing handlers to prevent duplication
    if list(logger.handlers) == []:
        # Create handlers:
        # FileHandler writes to a disk file
        # StreamHandler writes as a "return" output
        fileHandler = logging.FileHandler(log_path)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)
        
        logger.debug('[CMPL] Initialized logger')