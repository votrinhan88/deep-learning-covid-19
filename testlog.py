import logging
from . import logconf

logger = logging.getLogger(__name__)

def testlog(project_path):
    logconf.initLogger(logger, project_path = project_path)
    logger.debug('testlog')