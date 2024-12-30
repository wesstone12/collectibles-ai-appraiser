import time
import random
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10, exponential_base=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay

            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logger.error(f"Max retries ({max_retries}) reached. Last error: {str(e)}")
                        raise

                    # Calculate next delay with jitter
                    delay = min(delay * exponential_base, max_delay)
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = delay + jitter

                    logger.warning(f"Search failed (attempt {retries}/{max_retries}). "
                                 f"Retrying in {sleep_time:.2f} seconds... Error: {str(e)}")
                    time.sleep(sleep_time)
                    delay = min(delay * exponential_base, max_delay)

        return wrapper
    return decorator 