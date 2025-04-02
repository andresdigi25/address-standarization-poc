import logging
import sys
from .config import get_settings

settings = get_settings()

def setup_logging():
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging() 