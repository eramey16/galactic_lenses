import logging
import argparse
import time
import numpy as np

if __name__=='__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=None, help="Number to print")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger=logging.getLogger(__name__)
    sleeptime = 5 + 60*np.random.rand() # Sleep 5-65 seconds
    time.sleep(sleeptime)

    logger.info(f"Number {args.num} printing at {(time.time()-start):.2f} seconds")
    