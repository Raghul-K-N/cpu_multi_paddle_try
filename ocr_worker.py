# ocr_worker.py
import os
import sys
import time
import logging
import traceback
import json
import requests
import numpy as np


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


LLM_API_URL = "http://34.162.53.116:8080/v1/fetch-invoice-keyfields"
LLM_API_KEY = "Llama-api/version-0.1"
LLM_TIMEOUT = 800


def adapt_paddle_result(pipeline_output):
    """
    Converts PaddleOCR v3 .predict() output (flat box [xmin, ymin, xmax, ymax])
    into the legacy polygon format [[x,y], [x,y], [x,y], [x,y]] compatible
    with merge_text_with_spaces.

    Returns: List of lists — each page is a list of [polygon_box, [text, confidence]]
    """
    formatted_pages = []

    for page_data in pipeline_output:
        res = page_data
        raw_boxes = res.get('rec_boxes', [])
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])

        if isinstance(raw_boxes, np.ndarray):
            raw_boxes = raw_boxes.tolist()
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        page_items = []
        for i in range(len(texts)):
            box = raw_boxes[i]
            text = texts[i]
            conf = scores[i]

            if isinstance(box[0], (int, float)):
                x_min, y_min, x_max, y_max = box
                formatted_box = [
                    [x_min, y_min], [x_max, y_min],
                    [x_max, y_max], [x_min, y_max],
                ]
            else:
                formatted_box = box

            page_items.append([formatted_box, [text, conf]])

        formatted_pages.append(page_items)

    return formatted_pages


def merge_text_with_spaces(results, logger=None, confidence_threshold=0.8):
    """
    Merge text segments with proper spacing based on bounding box positions.

    Args:
        results: PaddleOCR results in legacy format — list of pages,
                 each page is a list of [polygon_box, [text, confidence]].
        logger:  Optional logger for debug output.
        confidence_threshold: Minimum confidence score to include text.

    Returns:
        list: Merged text lines with spatial layout preserved.
    """
    if not results:
        return []

    global_x_left = 99999
    all_merged_lines = []

    for page_num, sublist in enumerate(results, start=1):
        if not sublist:
            continue

        all_merged_lines.append(
            f"{'=' * 34}<Page {page_num} - START>{'=' * 32}"
        )

        lines = {}
        for each in sublist:
            box = each[0]
            text = each[1][0]
            confidence = each[1][1]

            if confidence < confidence_threshold:
                continue

            top_left, top_right, bottom_right, bottom_left = box
            max_top = max(top_left[1], top_right[1])
            max_bottom = max(bottom_left[1], bottom_right[1])
            y_center = (max_top + max_bottom) / 2
            line_key = round(y_center / 10) * 10

            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append({
                'text': text,
                'box': box,
                'x_left': min(point[0] for point in box),
                'x_right': max(point[0] for point in box),
                'confidence': confidence,
            })
            global_x_left = min(global_x_left, min(point[0] for point in box))

        merged_lines = []
        for y_pos in sorted(lines.keys()):
            line_segments = lines[y_pos]
            line_segments.sort(key=lambda x: x['x_left'])

            merged_text = ""
            for i, segment in enumerate(line_segments):
                text = segment['text']
                if i == 0:
                    gap = segment['x_left'] - global_x_left
                    if gap > 10:
                        no_of_spaces = round(gap / 10)
                        merged_text += " " * no_of_spaces
                if i > 0:
                    prev_segment = line_segments[i - 1]
                    gap = segment['x_left'] - prev_segment['x_right']
                    if gap > 10:
                        no_of_spaces = round(gap / 10)
                        merged_text += " " * no_of_spaces
                # Special handling for colons and numbers
                if text.startswith(':'):
                    merged_text = merged_text.rstrip() + text
                elif text[0].isdigit() and merged_text and not merged_text.endswith(' '):
                    merged_text += " " + text
                else:
                    merged_text += text

            if merged_text.strip():
                merged_lines.append(merged_text)

        merged_lines.append(
            f"{'=' * 34}<Page {page_num} - END>{'=' * 32}"
        )
        all_merged_lines.extend(merged_lines)

        if logger:
            kept = sum(len(lines[k]) for k in lines)
            logger.debug(
                f"Page {page_num}: {len(sublist)} segments, {kept} kept after confidence filter"
            )

    return all_merged_lines


def call_llm(text_lines, filename, logger, worker_id):
    """Send OCR text lines to the LLM API and return the JSON response."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {"lines": text_lines, "filename": filename}

    logger.info(f"Worker-{worker_id} calling LLM API ({len(text_lines)} lines) | fname={filename}")
    llm_start = time.time()
    try:
        resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        llm_elapsed = time.time() - llm_start
        logger.info(
            f"Worker-{worker_id} LLM response status={resp.status_code} "
            f"in {llm_elapsed:.2f}s | fname={filename}"
        )
        resp.raise_for_status()
        json_response = resp.json()

        # Log every key-value from the response
        invoice_data = json_response.get("invoice_data", {})
        raw_response = json_response.get("raw_response", {})
        logger.info(f"Worker-{worker_id} LLM invoice_data fields: | fname={filename}")
        for k, v in invoice_data.items():
            logger.info(f"  {k}: {v} | fname={filename}")
        logger.info(f"Worker-{worker_id} LLM raw_response fields: | fname={filename}")
        for k, v in raw_response.items():
            logger.info(f"  {k}: {v} | fname={filename}")

        return json_response
    except requests.exceptions.Timeout:
        llm_elapsed = time.time() - llm_start
        logger.error(
            f"Worker-{worker_id} LLM request timed out after {llm_elapsed:.2f}s | fname={filename}"
        )
    except requests.exceptions.RequestException as e:
        llm_elapsed = time.time() - llm_start
        logger.error(
            f"Worker-{worker_id} LLM request failed after {llm_elapsed:.2f}s: {e} | fname={filename}"
        )
        logger.debug(f"Traceback:\n{traceback.format_exc()} | fname={filename}")
    except Exception as e:
        llm_elapsed = time.time() - llm_start
        logger.error(
            f"Worker-{worker_id} LLM unexpected error after {llm_elapsed:.2f}s: {e} | fname={filename}"
        )
        logger.debug(f"Traceback:\n{traceback.format_exc()} | fname={filename}")
    return None


def ocr_worker(task_queue, output_dir, worker_id, log_filename, threads_per_worker=10):
    # 1) Set thread limits INSIDE spawned worker (before any numpy/MKL import)
    #    threads_per_worker = total_cpus // num_workers to avoid over-subscription
    tpw = str(threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = tpw
    os.environ["MKL_NUM_THREADS"] = tpw
    os.environ["NUMEXPR_NUM_THREADS"] = tpw
    os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

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
            lang="de",
            enable_mkldnn=True,
            device="cpu",
            mkldnn_cache_capacity=15,
            cpu_threads=threads_per_worker
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
            result = ocr.predict(file_path)
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

        # Filter out None pages and count
        page_count = 0
        valid_pages = []
        for page in result:
            if page is None:
                logger.warning(
                    f"Worker-{worker_id} encountered None page (page {page_count}) | fname={fname}"
                )
            else:
                valid_pages.append(page)
            page_count += 1

        # Use merge_text_with_spaces for spatially-aware text merging
        try:
            texts = adapt_paddle_result(valid_pages)
            text_lines = merge_text_with_spaces(texts, logger=logger)
            logger.info(
                f"Worker-{worker_id} merge_text_with_spaces produced {len(text_lines)} lines "
                f"from {len(valid_pages)} page(s) | fname={fname}"
            )
        except Exception as e:
            logger.error(
                f"Worker-{worker_id} error in merge_text_with_spaces: {e} | fname={fname}"
            )
            logger.debug(f"Traceback:\n{traceback.format_exc()} | fname={fname}")
            files_failed += 1
            continue

        if not text_lines:
            logger.warning(f"Worker-{worker_id} no text extracted after merge | fname={fname}")
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

        # --- Call LLM API ---
        llm_response = call_llm(text_lines, fname, logger, worker_id)
        if llm_response is not None:
            # Save LLM response alongside OCR output
            llm_out_file = os.path.join(output_dir, f"{fname}.llm.json")
            try:
                with open(llm_out_file, "w", encoding="utf-8") as lf:
                    json.dump(llm_response, lf, indent=2, ensure_ascii=False)
                logger.info(
                    f"Worker-{worker_id} LLM result saved to {llm_out_file} | fname={fname}"
                )
            except Exception as e:
                logger.error(
                    f"Worker-{worker_id} failed to write LLM result: {e} | fname={fname}"
                )
        else:
            logger.warning(f"Worker-{worker_id} LLM returned no result | fname={fname}")

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
