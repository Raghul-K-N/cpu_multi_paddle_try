# main.py
import os
import sys
import time
import logging
import multiprocessing as mp
from datetime import datetime
from ocr_worker import ocr_worker

INPUT_DIR = "./pepsico_attachments"
OUTPUT_DIR = "./ocr_output"
LOG_DIR = "./logs"
NUM_WORKERS = 10
QUEUE_SIZE = 5  # bounded queue
TOTAL_CPUS = os.cpu_count() or 4  # auto-detect available vCPUs
threads_per_paddle = 5
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Sentinel value to signal workers to shut down cleanly
SENTINEL = None


def setup_logging():
    """Create a new timestamped log file for each run and return the logger."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"ocr_run_{timestamp}.log")

    logger = logging.getLogger("ocr_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — captures everything
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file created: {log_filename} | fname=")
    return logger, log_filename


def collect_input_files(input_dir, logger):
    """Collect and validate input files, filtering by supported extensions."""
    all_entries = os.listdir(input_dir)
    logger.info(f"Total entries found in input directory: {len(all_entries)} | fname=SCAN")
    logger.debug(f"All entries: {all_entries} | fname=SCAN")

    files = []
    skipped = []
    for entry in sorted(all_entries):
        full_path = os.path.join(input_dir, entry)
        if not os.path.isfile(full_path):
            logger.debug(f"Skipping non-file entry: {entry} | fname={entry}")
            skipped.append(entry)
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Skipping unsupported file type '{ext}': {entry} | fname={entry}")
            skipped.append(entry)
            continue
        file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
        logger.debug(f"Queuing file: {entry} ({file_size_mb:.2f} MB) | fname={entry}")
        files.append(full_path)

    logger.info(
        f"Files to process: {len(files)}, Skipped: {len(skipped)} | fname=SCAN"
    )
    return files


def main():
    run_start = time.time()

    logger, log_filename = setup_logging()
    logger.info(f"========== OCR Pipeline Starting ========== | fname=INIT")
    logger.info(f"Input directory : {os.path.abspath(INPUT_DIR)} | fname=INIT")
    logger.info(f"Output directory: {os.path.abspath(OUTPUT_DIR)} | fname=INIT")
    threads_per_worker = threads_per_paddle
    logger.info(f"Total vCPUs     : {TOTAL_CPUS} | fname=INIT")
    logger.info(f"Num workers     : {NUM_WORKERS} | fname=INIT")
    logger.info(f"Threads/worker  : {threads_per_worker} | fname=INIT")
    logger.info(f"Queue size      : {QUEUE_SIZE} | fname=INIT")
    logger.info(f"Supported exts  : {SUPPORTED_EXTENSIONS} | fname=INIT")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validate input directory
    if not os.path.isdir(INPUT_DIR):
        logger.error(f"Input directory does not exist: {INPUT_DIR} | fname=INIT")
        sys.exit(1)

    # Collect files before starting workers
    files = collect_input_files(INPUT_DIR, logger)
    if not files:
        logger.warning(f"No processable files found. Exiting. | fname=INIT")
        sys.exit(0)

    # Use spawn to avoid fork + MKL weirdness
    mp.set_start_method("spawn", force=True)
    logger.info(f"Multiprocessing start method set to 'spawn' | fname=INIT")

    task_queue = mp.Queue(maxsize=QUEUE_SIZE)
    logger.debug(f"Task queue created with maxsize={QUEUE_SIZE} | fname=INIT")

    # Start workers
    workers = []
    logger.info(f"Spawning {NUM_WORKERS} worker processes... | fname=INIT")
    for i in range(NUM_WORKERS):
        p = mp.Process(
            target=ocr_worker,
            args=(task_queue, OUTPUT_DIR, i, log_filename, threads_per_worker),
            name=f"Worker-{i}",
        )
        p.start()
        logger.info(f"Worker-{i} started (PID={p.pid}) | fname=INIT")
        workers.append(p)

    # Enqueue tasks
    logger.info(f"Enqueuing {len(files)} file(s)... | fname=ENQUEUE")
    enqueue_start = time.time()
    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        logger.debug(f"Putting file {idx}/{len(files)} into queue | fname={fname}")
        task_queue.put(fpath)
        logger.debug(f"File enqueued successfully ({idx}/{len(files)}) | fname={fname}")
    enqueue_elapsed = time.time() - enqueue_start
    logger.info(f"All {len(files)} files enqueued in {enqueue_elapsed:.3f}s | fname=ENQUEUE")

    # Send sentinel values to signal workers to stop
    logger.info(f"Sending {NUM_WORKERS} sentinel(s) to shut down workers | fname=SHUTDOWN")
    for i in range(NUM_WORKERS):
        task_queue.put(SENTINEL)
    logger.debug(f"All sentinels sent | fname=SHUTDOWN")

    # Wait for workers to finish
    logger.info(f"Waiting for workers to finish... | fname=SHUTDOWN")
    for p in workers:
        p.join()
        exit_code = p.exitcode
        status = "OK" if exit_code == 0 else f"FAILED (exit={exit_code})"
        logger.info(f"Worker {p.name} (PID={p.pid}) joined — {status} | fname=SHUTDOWN")

    run_elapsed = time.time() - run_start
    avg_time = run_elapsed / len(files) if files else 0

    logger.info(f"========== OCR Pipeline Complete ========== | fname=DONE")
    logger.info(f"Total files processed : {len(files)} | fname=DONE")
    logger.info(f"Total wall-clock time : {run_elapsed:.2f}s | fname=DONE")
    logger.info(f"Average time per file : {avg_time:.2f}s | fname=DONE")
    logger.info(f"Log saved to          : {log_filename} | fname=DONE")


if __name__ == "__main__":
    main()
