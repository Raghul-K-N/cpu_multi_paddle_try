# ocr_worker.py
import os
import sys
import time
import logging
import traceback


def _setup_worker_logger(worker_id, log_filename):
    """Attach this worker process to the shared log file."""
    logger = logging.getLogger(f"ocr_pipeline.worker{worker_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

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
    return logger


def ocr_worker(task_queue, output_dir, worker_id, log_filename):
    # 1) Set thread limits INSIDE spawned worker (before any numpy/MKL import)
    os.environ["OMP_NUM_THREADS"] = "10"
    os.environ["MKL_NUM_THREADS"] = "10"
    os.environ["NUMEXPR_NUM_THREADS"] = "10"

    logger = _setup_worker_logger(worker_id, log_filename)
    logger.info(f"Worker-{worker_id} process started (PID={os.getpid()}) | fname=INIT")
    logger.info(
        f"Thread env — OMP={os.environ.get('OMP_NUM_THREADS')}, "
        f"MKL={os.environ.get('MKL_NUM_THREADS')}, "
        f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')} | fname=INIT"
    )

    # 2) Initialize PaddleOCR ONCE — measure time
    logger.info(f"Worker-{worker_id} initializing PaddleOCR model... | fname=INIT")
    model_init_start = time.time()
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            enable_mkldnn=True,
            device="cpu",
            show_log=False,  # suppress paddle's own verbose logs
        )
        model_init_elapsed = time.time() - model_init_start
        logger.info(
            f"Worker-{worker_id} PaddleOCR model initialized in "
            f"{model_init_elapsed:.2f}s | fname=INIT"
        )
    except Exception as e:
        model_init_elapsed = time.time() - model_init_start
        logger.critical(
            f"Worker-{worker_id} FAILED to initialize PaddleOCR after "
            f"{model_init_elapsed:.2f}s: {e} | fname=INIT"
        )
        logger.debug(f"Traceback:\n{traceback.format_exc()} | fname=INIT")
        return  # cannot continue without model

    files_processed = 0
    files_failed = 0
    total_ocr_time = 0.0

    # 3) Processing loop — exits on sentinel (None)
    logger.info(f"Worker-{worker_id} entering processing loop | fname=INIT")
    while True:
        # --- Fetch next task ---
        logger.debug(f"Worker-{worker_id} waiting for next task from queue | fname=QUEUE_WAIT")
        try:
            file_path = task_queue.get(timeout=30)
        except Exception:
            logger.warning(
                f"Worker-{worker_id} timed out waiting for queue (30s). Exiting. | fname=TIMEOUT"
            )
            break

        # --- Sentinel check ---
        if file_path is None:
            logger.info(f"Worker-{worker_id} received sentinel. Shutting down. | fname=SENTINEL")
            break

        fname = os.path.basename(file_path)
        logger.info(f"Worker-{worker_id} picked up task | fname={fname}")

        # --- Validate file ---
        if not os.path.isfile(file_path):
            logger.error(f"Worker-{worker_id} file not found: {file_path} | fname={fname}")
            files_failed += 1
            continue

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(
            f"Worker-{worker_id} starting OCR ({file_size_mb:.2f} MB) | fname={fname}"
        )

        # --- Run OCR ---
        ocr_start = time.time()
        try:
            result = ocr.ocr(file_path)
            ocr_elapsed = time.time() - ocr_start
            total_ocr_time += ocr_elapsed
            logger.info(
                f"Worker-{worker_id} OCR completed in {ocr_elapsed:.2f}s | fname={fname}"
            )
        except Exception as e:
            ocr_elapsed = time.time() - ocr_start
            logger.error(
                f"Worker-{worker_id} OCR FAILED after {ocr_elapsed:.2f}s: {e} | fname={fname}"
            )
            logger.debug(f"Traceback:\n{traceback.format_exc()} | fname={fname}")
            files_failed += 1
            continue

        # --- Parse results ---
        if result is None:
            logger.warning(f"Worker-{worker_id} OCR returned None result | fname={fname}")
            files_failed += 1
            continue

        line_count = 0
        page_count = 0
        text_lines = []
        try:
            for page in result:
                if page is None:
                    logger.warning(
                        f"Worker-{worker_id} encountered None page (page {page_count}) | fname={fname}"
                    )
                    page_count += 1
                    continue
                page_count += 1
                for line in page:
                    text_lines.append(line[1][0])
                    line_count += 1
            logger.info(
                f"Worker-{worker_id} extracted {line_count} lines from {page_count} page(s) | fname={fname}"
            )
        except Exception as e:
            logger.error(
                f"Worker-{worker_id} error parsing OCR result: {e} | fname={fname}"
            )
            logger.debug(f"Traceback:\n{traceback.format_exc()} | fname={fname}")
            files_failed += 1
            continue

        # --- Write output ---
        out_file = os.path.join(output_dir, f"{fname}.txt")
        write_start = time.time()
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                f.write("\n".join(text_lines) + "\n")
            write_elapsed = time.time() - write_start
            logger.info(
                f"Worker-{worker_id} output written to {out_file} "
                f"({write_elapsed:.4f}s) | fname={fname}"
            )
        except Exception as e:
            logger.error(
                f"Worker-{worker_id} failed to write output: {e} | fname={fname}"
            )
            logger.debug(f"Traceback:\n{traceback.format_exc()} | fname={fname}")
            files_failed += 1
            continue

        files_processed += 1
        total_elapsed = time.time() - ocr_start
        logger.info(
            f"Worker-{worker_id} completed (OCR={ocr_elapsed:.2f}s, "
            f"total={total_elapsed:.2f}s) | fname={fname}"
        )

    # --- Worker summary ---
    avg_ocr = (total_ocr_time / files_processed) if files_processed > 0 else 0
    logger.info(
        f"Worker-{worker_id} summary — processed={files_processed}, "
        f"failed={files_failed}, total_ocr_time={total_ocr_time:.2f}s, "
        f"avg_ocr_per_file={avg_ocr:.2f}s, "
        f"model_init_time={model_init_elapsed:.2f}s | fname=SUMMARY"
    )
    logger.info(f"Worker-{worker_id} exiting | fname=SHUTDOWN")
