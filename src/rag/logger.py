import logging
import sys

# Create logger
logger = logging.getLogger("alzheimer-rag")
logger.setLevel(logging.INFO)

# Create console handler
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler
logger.addHandler(handler)
