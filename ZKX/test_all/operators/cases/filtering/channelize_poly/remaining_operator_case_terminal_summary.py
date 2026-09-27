"""Render Remaining operator case per-case terminal tables from the completed main CSV."""

import atexit
import csv
import json
import math
import pathlib
import shlex


_REGISTERED = []


def verify_input_content_evidence(row, expected_names):
    """Reject digest-only evidence and validate the public readable-input contract."""
    content = json.loads(row.get("input_content_json", "[]"))
    parameters = json.loads(row.get("case_parameters_json", "{}"))
    if [item.get("input_name") for item in content] != list(expected_names):
        raise ValueError("input content names mismatch")
    if not parameters:
        raise ValueError("case parameters missing")
    for item in content:
        required = ("input_name", "shape", "dtype", "element_count",
                    "generator", "preview_heacuda_api_tail2")
        if any(item.get(key) in (None, "", "NA") for key in required):
            raise ValueError(f"incomplete input content row: {item}")
        count = int(item["element_count"])
        preview = item["preview_heacuda_api_tail2"].split(";")
        if len(preview) != min(count, 6):
            raise ValueError(f"input preview count mismatch: {item['input_name']}")
        expected_indices = _preview_indices(count)
        actual_indices = [int(value.split(":", 1)[0][2:]) for value in preview]
        if actual_indices != expected_indices:
            raise ValueError(f"input preview indices mismatch: {item['input_name']}")


def _table(title, columns, rows):
    values = [[str(row.get(key, "NA")) for key, _ in columns] for row in rows]
    widths = [max([len(label)] + [len(row[index]) for row in values])
              for index, (_, label) in enumerate(columns)]
    line = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    body = [title, line,
            "| " + " | ".join(label.ljust(widths[index])
                                for index, (_, label) in enumerate(columns)) + " |",
            line]
    body.extend("| " + " | ".join(row[index].ljust(widths[index])
                                    for index in range(len(columns))) + " |"
                for row in values)
    body.append(line)
    return "\n".join(body)


def _kind(path):
    name = path.name
    if name.startswith("operator_main_results_"):
        return "main_csv"
    if name.startswith("operator_timing_samples_"):
        return "timing_csv"
    if "memory_trace" in name:
        return "memory_csv"
    if name.startswith("operator_dtype_evidence_"):
        return "dtype_csv"
    if name.endswith("_report.json"):
        return "report_json"
    if name.endswith("_summary.txt"):
        return "summary_txt"
    if name.endswith("_full.log"):
        return "full_log"
    if name.endswith("_probe_raw.json"):
        return "probe_raw_json"
    return "related_file"


def _resource_summaries(directory):
    paths = list(directory.glob("*memory_trace*.csv"))
    if len(paths) != 1:
        return {}
    with paths[0].open(encoding="utf-8", newline="") as handle:
        trace = list(csv.DictReader(handle))
    summaries = {}
    for device in ("CPU", "GPU"):
        device_rows = sorted((row for row in trace if row.get("device") == device),
                             key=lambda row: int(row.get("phase_index", "0")))
        for prefix, column in (("cpu_heap", "cpu_live_heap_bytes"),
                               ("rss", "rss_bytes"), ("gpu", "gpu_used_bytes")):
            values = [int(row[column]) for row in device_rows
                      if row.get(column, "NA") not in ("", "NA")]
            if values:
                summaries[(device, prefix)] = {
                    "before": values[0], "after": values[-1], "peak": max(values),
                    "delta": values[-1] - values[0],
                }
    return summaries


def _backfill_resources(main_path, fieldnames, rows):
    summaries = _resource_summaries(main_path.parent)
    changed = False
    for row in rows:
        device = row.get("device", "NA")
        for prefix in ("cpu_heap", "rss", "gpu"):
            summary = summaries.get((device, prefix))
            if not summary:
                continue
            for field in ("before", "after", "peak", "delta"):
                key = f"{prefix}_{field}_bytes"
                if row.get(key, "NA") in ("", "NA"):
                    row[key] = str(summary[field])
                    changed = True
    if changed:
        with main_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)


def _case_parameters(directory):
    logs = list(directory.glob("operator_*_full.log"))
    if len(logs) != 1:
        return {}
    first = logs[0].read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if not first or not first[0].startswith("command="):
        return {}
    tokens = shlex.split(first[0][len("command="):])
    result = {}
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:]
        # Every Remaining operator case probe option is emitted as a key/value pair.  An empty
        # string (for example detrend --breakpoints "") must remain empty and
        # must not be misreported as a boolean flag.
        value = ""
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1]
            index += 1
        if key not in {"output", "warmup", "measured"}:
            result[key] = value
        index += 1
    return result


def _preview_indices(count):
    return list(range(count)) if count <= 6 else [0, 1, 2, 3, count - 2, count - 1]


def _format_value(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def _modular_preview(count, seed, dtype, taps=False):
    integral = dtype.startswith("INT")
    values = []
    for index in _preview_indices(count):
        raw = ((index * 7 + seed * 3) % 9 - 4) if taps \
            else ((index * 17 + seed) % 23 - 11)
        value = raw if integral else raw * (0.125 if taps else 0.25)
        values.append(f"i={index}:{_format_value(value)}")
    return ";".join(values)


def _formula_preview(count, dtype, formula, scale=0.25):
    integral = dtype.startswith("INT")
    values = []
    for index in _preview_indices(count):
        raw = formula(index)
        value = raw if integral else raw * scale
        values.append(f"i={index}:{_format_value(value)}")
    return ";".join(values)


def _fixed_preview(values):
    return ";".join(f"i={index}:{_format_value(value)}"
                    for index, value in enumerate(values))


def _actual_preview(count, formula):
    return ";".join(
        f"i={index}:{_format_value(formula(index))}"
        for index in _preview_indices(count))


def _input_content(operator, parameters, dtype):
    def integer(key):
        return int(parameters[key])

    def row(name, shape, count, generator, preview):
        return {"input_name": name, "shape": shape, "dtype": dtype,
                "element_count": str(count), "generator": generator,
                "preview_heacuda_api_tail2": preview}

    if operator == "channelize_poly":
        seed = integer("seed")
        x_count, h_count = integer("x-count"), integer("h-count")
        return [
            row("x", f"[{x_count}]", x_count,
                f"modular_signal(seed={seed},integral={str(dtype.startswith('INT')).lower()})",
                _modular_preview(x_count, seed, dtype)),
            row("h", f"[{h_count}]", h_count,
                f"modular_taps(seed={seed + 1},integral={str(dtype.startswith('INT')).lower()})",
                _modular_preview(h_count, seed + 1, dtype, taps=True)),
        ]
    if operator in {"hilbert", "hilbert2", "resample", "decimate"}:
        rows, cols, seed = integer("rows"), integer("cols"), integer("seed")
        count = rows * cols
        content = [row(
            "x", f"[{rows}x{cols}]", count,
            f"modular_signal(seed={seed},integral={str(dtype.startswith('INT')).lower()})",
            _modular_preview(count, seed, dtype))]
        if operator == "decimate" and parameters.get("custom-coefficients") == "true":
            content.append(row("h", "[5]", 5, "fixed_custom_coefficients",
                               "i=0:1;i=1:2;i=2:1;i=3:-1;i=4:1"))
        return content
    if operator == "correlate2d":
        seed = integer("seed")
        r1, c1, r2, c2 = (integer(key) for key in ("rows1", "cols1", "rows2", "cols2"))
        inputs = []
        for name, rows, cols, salt in (("in1", r1, c1, 3), ("in2", r2, c2, 7)):
            count = rows * cols
            inputs.append(row(name, f"[{rows}x{cols}]", count,
                              f"correlate_modular(seed={seed},salt={salt},integral={str(dtype.startswith('INT')).lower()})",
                              _formula_preview(count, dtype,
                                               lambda index, salt=salt: (index * 17 + seed + salt) % 11 - 5)))
        return inputs
    if operator == "firwin2":
        return [
            row("freq", "[3]", 3, "fixed_frequency_breakpoints", _fixed_preview([0, 4, 8])),
            row("gain", "[3]", 3, "fixed_gain_breakpoints", _fixed_preview([0, 1, 0])),
        ]
    if operator == "firfilter2":
        rows, cols, seed = integer("rows"), integer("cols"), integer("seed")
        count = rows * cols
        return [
            row("x", f"[{rows}x{cols}]", count,
                f"firfilter_modular(seed={seed},integral={str(dtype.startswith('INT')).lower()})",
                _formula_preview(count, dtype, lambda index: (index * 13 + seed) % 9 - 4)),
            row("b", "[3]", 3, "fixed_filter_coefficients", _fixed_preview([1, 1, 1])),
        ]
    if operator == "resample_poly":
        rows, cols, seed = integer("rows"), integer("cols"), integer("seed")
        count = rows * cols
        content = [row(
            "x", f"[{rows}x{cols}]", count,
            f"resample_poly_modular(seed={seed},integral={str(dtype.startswith('INT')).lower()})",
            _formula_preview(count, dtype, lambda index: (index * 17 + seed) % 11 - 5))]
        if parameters.get("filter-mode") == "custom":
            content.append(row("h", "[5]", 5, "fixed_custom_coefficients",
                               _fixed_preview([1, 2, 3, 2, 1])))
        return content
    if operator == "upfirdn":
        rank, rows, cols = integer("rank"), integer("rows"), integer("cols")
        taps, seed = integer("taps"), integer("seed")
        x_count = rows * (cols if rank == 2 else 1)
        return [
            row("h", f"[{taps}]", taps,
                f"upfirdn_modular(seed={seed + 1},integral={str(dtype.startswith('INT')).lower()})",
                _formula_preview(taps, dtype, lambda index: (index * 17 + seed + 1) % 9 - 4)),
            row("x", f"[{rows}x{cols}]" if rank == 2 else f"[{rows}]", x_count,
                f"upfirdn_modular(seed={seed},integral={str(dtype.startswith('INT')).lower()})",
                _formula_preview(x_count, dtype, lambda index: (index * 17 + seed) % 9 - 4)),
        ]
    if operator == "detrend":
        rank, rows, cols = integer("rank"), integer("rows"), integer("cols")
        seed = integer("seed")
        count = rows * (cols if rank == 2 else 1)
        integral = dtype.startswith("INT")
        def detrend_value(index):
            value = (((index * 13 + seed) % 9) - 4) * 0.25 + 0.05 * index
            return int(value) if integral else value
        return [row(
            "x", f"[{rows}x{cols}]" if rank == 2 else f"[{rows}]", count,
            f"trend_plus_modular_noise(seed={seed},integral={str(integral).lower()})",
            _actual_preview(count, detrend_value))]
    if operator == "freq_shift":
        count, seed = integer("count"), integer("seed")
        integral = dtype.startswith("INT")
        def freq_shift_value(index):
            value = (((index * 17 + seed) % 11) - 5) * 0.25 + (index % 97) * 0.03125
            return int(value) if integral else value
        return [row(
            "x", f"[{count}]", count,
            f"modular_noise_plus_ramp(seed={seed},integral={str(integral).lower()})",
            _actual_preview(count, freq_shift_value))]
    if operator == "lfilter_zi":
        order, leading = integer("order"), integer("leading")
        b_count, a_count = order + 1, order + 1 + leading
        b = [1] + [0] * order
        a = [0] * a_count
        a[leading], a[leading + 1] = 2, -1
        return [
            row("b", f"[{b_count}]", b_count, "fixed_lfilter_zi_numerator",
                _actual_preview(b_count, lambda index: b[index])),
            row("a", f"[{a_count}]", a_count,
                f"fixed_lfilter_zi_denominator(leading_zeros={leading})",
                _actual_preview(a_count, lambda index: a[index])),
        ]
    if operator == "sosfilt":
        rank, rows, cols = integer("rank"), integer("rows"), integer("cols")
        sections, axis, seed = integer("sections"), integer("axis"), integer("seed")
        x_count = rows * (cols if rank == 2 else 1)
        integral = dtype.startswith("INT")
        content = [
            row("sos", f"[{sections}x6]", sections * 6,
                "repeated_sos_section_[1,1,0,1,0,0]",
                _actual_preview(sections * 6,
                                lambda index: (1 if index % 6 in (0, 1, 3) else 0))),
            row("x", f"[{rows}x{cols}]" if rank == 2 else f"[{rows}]", x_count,
                f"sosfilt_modular(seed={seed},integral={str(integral).lower()})",
                _formula_preview(x_count, dtype,
                                 lambda index: (index * 13 + seed) % 9 - 4)),
        ]
        if parameters.get("zi") == "true":
            zi_count = sections * 2 * (cols if rank == 2 and axis == 0
                                       else rows if rank == 2 else 1)
            zi_shape = (f"[{sections}x2x{cols}]" if axis == 0
                        else f"[{sections}x{rows}x2]") if rank == 2 \
                else f"[{sections}x2]"
            content.append(row("zi", zi_shape, zi_count, "constant_initial_state(1)",
                               _actual_preview(zi_count, lambda _index: 1)))
        return content
    if operator == "wiener":
        rank, rows, cols = integer("rank"), integer("rows"), integer("cols")
        seed = integer("seed")
        count = rows * (cols if rank == 2 else 1)
        integral = dtype.startswith("INT")
        def wiener_value(index):
            z = ((index * 17 + seed) % 11) - 5
            return z if integral else z * 0.25 + (index % 7) * 0.0625
        return [row(
            "x", f"[{rows}x{cols}]" if rank == 2 else f"[{rows}]", count,
            f"wiener_modular_plus_periodic_offset(seed={seed},integral={str(integral).lower()})",
            _actual_preview(count, wiener_value))]
    if operator in {"csd", "stft"}:
        seed = integer("seed")
        integral = dtype.startswith("INT")
        def spectral_value(index, input_seed):
            z = ((index * 17 + input_seed) % 11) - 5
            return z if integral else z * 0.25 + (index % 13) * 0.03125
        if operator == "stft":
            count = integer("count")
            return [row(
                "x", f"[{count}]", count,
                f"spectral_modular_plus_periodic_offset(seed={seed},integral={str(integral).lower()})",
                _actual_preview(count, lambda index: spectral_value(index, seed)))]
        nx, ny = integer("nx"), integer("ny")
        return [
            row("x", f"[{nx}]", nx,
                f"spectral_modular_plus_periodic_offset(seed={seed},integral={str(integral).lower()})",
                _actual_preview(nx, lambda index: spectral_value(index, seed))),
            row("y", f"[{ny}]", ny,
                f"spectral_modular_plus_periodic_offset(seed={seed + 1},integral={str(integral).lower()})",
                _actual_preview(ny, lambda index: spectral_value(index, seed + 1))),
        ]
    if operator == "istft":
        bins, frames, seed = integer("bins"), integer("frames"), integer("seed")
        count = bins * frames
        integral = dtype.startswith("INT")
        def complex_value(index):
            real_raw = ((index * 17 + seed) % 11) - 5
            imag_raw = ((index * 7 + seed // 3) % 9) - 4
            real = real_raw if integral else real_raw * 0.25 + (index % 13) * 0.03125
            imag = imag_raw if integral else imag_raw * 0.125 - (index % 7) * 0.015625
            return f"({_format_value(real)},{_format_value(imag)})"
        return [{
            "input_name": "z", "shape": f"[{bins}x{frames}]",
            "dtype": f"Complex{dtype}", "element_count": str(count),
            "generator": (f"frequency_major_complex_modular(seed={seed},"
                          f"component_dtype={dtype},integral={str(integral).lower()})"),
            "preview_heacuda_api_tail2": _actual_preview(count, complex_value),
        }]
    if operator == "lombscargle":
        samples, frequencies, seed = (integer(key)
                                      for key in ("samples", "frequencies", "seed"))
        integral = dtype.startswith("INT")
        def time_value(index):
            return ((index * 3 + index // 7) % 101) if integral \
                else index * 0.125 + (index % 7) * 0.0078125
        def signal_value(index):
            z = ((index * 17 + seed) % 11) - 5
            return z if integral else z * 0.25 + (index % 13) * 0.03125
        def frequency_value(index):
            return index + 1 if integral else 0.05 + (index + 1) * 0.035
        return [
            row("time", f"[{samples}]", samples,
                f"deterministic_sample_times(integral={str(integral).lower()})",
                _actual_preview(samples, time_value)),
            row("value", f"[{samples}]", samples,
                f"modular_observations(seed={seed},integral={str(integral).lower()})",
                _actual_preview(samples, signal_value)),
            row("frequency", f"[{frequencies}]", frequencies,
                f"deterministic_frequency_grid(integral={str(integral).lower()})",
                _actual_preview(frequencies, frequency_value)),
        ]
    if operator == "vectorstrength":
        events, periods, seed = (integer(key) for key in ("events", "periods", "seed"))
        integral = dtype.startswith("INT")
        def event_value(index):
            return ((index * 3 + index // 7 + seed % 5) % 101) if integral \
                else index * 0.125 + (index % 7) * 0.0078125
        def period_value(index):
            return index + 3 if integral else 0.5 + (index + 1) * 0.25
        content = [row(
            "events", f"[{events}]", events,
            f"deterministic_events(seed={seed},integral={str(integral).lower()})",
            _actual_preview(events, event_value))]
        scalar = parameters.get("period-mode") == "scalar"
        content.append(row(
            "period" if scalar else "periods", "scalar" if scalar else f"[{periods}]",
            periods, f"deterministic_period_grid(integral={str(integral).lower()})",
            _actual_preview(periods, period_value)))
        return content
    if operator in {"morlet", "morlet2"}:
        frequency = parameters["frequency"]
        scale = parameters["scale"]
        return [
            row("frequency", "scalar", 1, "configured_typed_scalar",
                f"i=0:{frequency}"),
            row("scale", "scalar", 1, "configured_typed_scalar",
                f"i=0:{scale}"),
        ]
    if operator == "qmf":
        count, seed = integer("elements"), integer("seed")
        integral = dtype.startswith("INT")
        def qmf_value(index):
            return ((index * 7 + seed % 11) % 97 - 48) if integral \
                else math.sin(index * 0.03125) + (seed % 7) * 0.015625
        return [row(
            "hk", f"[{count}]", count,
            f"qmf_sinusoid_or_modular(seed={seed},integral={str(integral).lower()})",
            _actual_preview(count, qmf_value))]
    if operator in {"gauss_spline", "quadratic"}:
        count, seed = integer("elements"), integer("seed")
        integral = dtype.startswith("INT")
        def bspline_value(index):
            return ((index + seed % 5) % 7 - 3) if integral \
                else ((index % 257) - 128) / 32.0 + (seed % 3) / 16.0
        return [row(
            "x", f"[{count}]", count,
            f"bspline_modular_or_piecewise_ramp(seed={seed},integral={str(integral).lower()})",
            _actual_preview(count, bspline_value))]
    if operator == "unit_impulse":
        shape, idx = integer("shape"), integer("idx")
        mode = parameters["idx-mode"]
        return [{
            "input_name": "request_spec", "shape": "scalar_parameters",
            "dtype": "NO_TYPED_BUSINESS_INPUT", "element_count": "1",
            "generator": "unit_impulse_request(dtype_argument_ignored=true)",
            "preview_heacuda_api_tail2": (
                f"i=0:shape={shape},idx={idx},mode={mode},request_variant={dtype}"),
        }]
    return []


def _enrich_dtype_evidence(main_path, main_rows):
    paths = list(main_path.parent.glob("operator_dtype_evidence_*.csv"))
    if len(paths) != 1 or not main_rows:
        return
    path = paths[0]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return
    parameters = _case_parameters(main_path.parent)
    operator = main_rows[0].get("operator_name", main_rows[0].get("target", ""))
    dtype = main_rows[0].get("dtype", "NA")
    content = _input_content(operator, parameters, dtype)
    for field in ("input_content_json", "case_parameters_json"):
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        row["input_content_json"] = json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        row["case_parameters_json"] = json.dumps(
            parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    report_paths = list(main_path.parent.glob("operator_*_report.json"))
    if len(report_paths) == 1:
        report = json.loads(report_paths[0].read_text(encoding="utf-8"))
        report["input_content"] = content
        report["case_parameters"] = parameters
        report_paths[0].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_paths = list(main_path.parent.glob("operator_*_summary.txt"))
    if len(summary_paths) == 1:
        with summary_paths[0].open("a", encoding="utf-8") as handle:
            handle.write("input_content_json=" + json.dumps(
                content, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.write("case_parameters_json=" + json.dumps(
                parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def render_main_csv(main_path):
    main_path = pathlib.Path(main_path)
    with main_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not rows or "_remaining_operator_case_" not in rows[0].get("run_id", ""):
        return ""
    _backfill_resources(main_path, fieldnames, rows)
    _enrich_dtype_evidence(main_path, rows)
    blocks = [_table(
        "[remaining_operator_case] identity and status",
        [("target", "target"), ("scale_id", "scale_id"), ("device", "device"),
         ("backend", "backend"), ("dtype", "dtype"), ("warmup_runs", "warmup"),
         ("measured_runs", "measured"), ("timing_scope", "timing_scope"),
         ("status", "status"), ("error_code", "error_code")], rows)]
    blocks.append(_table(
        "[remaining_operator_case] timing detail",
        [("device", "device"), ("mean_ms", "mean_ms"), ("p50_ms", "p50_ms"),
         ("p95_ms", "p95_ms"), ("p99_ms", "p99_ms"), ("min_ms", "min_ms"),
         ("max_ms", "max_ms"), ("std_ms", "std_ms"), ("cv", "cv"),
         ("cpu_gpu_speedup", "speedup")], rows))
    gpu = next((row for row in rows if row.get("device") == "GPU"), rows[-1])
    blocks.append(_table(
        "[remaining_operator_case] accuracy detail",
        [("accuracy_reference", "accuracy_reference"), ("mse", "mse"),
         ("rmse", "rmse"), ("relative_l2", "relative_l2"),
         ("relative_linf", "relative_linf"), ("exact_match", "exact_match"),
         ("mismatch_count", "mismatch_count"), ("semantic_check", "semantic_check"),
         ("accuracy_status", "status")], [gpu]))
    resources = []
    for row in rows:
        for prefix, label in (("cpu_heap", "cpu_heap"), ("rss", "rss"),
                              ("gpu", "gpu_memory")):
            if row.get(f"{prefix}_before_bytes", "NA") != "NA":
                resources.append({
                    "device": row.get("device", "NA"), "metric": label,
                    "before": row.get(f"{prefix}_before_bytes", "NA"),
                    "after": row.get(f"{prefix}_after_bytes", "NA"),
                    "peak": row.get(f"{prefix}_peak_bytes", "NA"),
                    "delta": row.get(f"{prefix}_delta_bytes", "NA"),
                })
    blocks.append(_table(
        "[remaining_operator_case] resource detail",
        [("device", "device"), ("metric", "metric"), ("before", "before_bytes"),
         ("after", "after_bytes"), ("peak", "peak_bytes"),
         ("delta", "delta_bytes")], resources))
    timing_paths = list(main_path.parent.glob("operator_timing_samples_*.csv"))
    io_rows = []
    if len(timing_paths) == 1:
        with timing_paths[0].open(encoding="utf-8", newline="") as handle:
            timing_rows = list(csv.DictReader(handle))
        for device in ("CPU", "GPU"):
            sample = next((row for row in timing_rows if row.get("device") == device), None)
            if sample:
                io_rows.append({"device": device, "input_digest": sample.get("input_digest", "NA"),
                                "output_digest": sample.get("output_digest", "NA")})
    blocks.append(_table("[remaining_operator_case] input/output identity",
                         [("device", "device"), ("input_digest", "input_digest"),
                          ("output_digest", "output_digest")], io_rows))
    dtype_paths = list(main_path.parent.glob("operator_dtype_evidence_*.csv"))
    dtype_rows = []
    if len(dtype_paths) == 1:
        with dtype_paths[0].open(encoding="utf-8", newline="") as handle:
            dtype_rows = list(csv.DictReader(handle))
    input_rows = []
    parameters = "{}"
    if dtype_rows:
        try:
            input_rows = json.loads(dtype_rows[0].get("input_content_json", "[]"))
        except json.JSONDecodeError:
            input_rows = []
        parameters = dtype_rows[0].get("case_parameters_json", "{}")
    blocks.append(_table("[remaining_operator_case] input content",
                         [("input_name", "input_name"), ("shape", "shape"),
                          ("dtype", "dtype"), ("element_count", "elements"),
                          ("generator", "generator"),
                          ("preview_heacuda_api_tail2", "values(head4/tail2)")], input_rows))
    blocks.append(_table("[remaining_operator_case] case parameters",
                         [("parameters", "parameters")],
                         [{"parameters": parameters}]))
    blocks.append(_table("[remaining_operator_case] dtype evidence",
                         [("requested_dtype", "requested_dtype"),
                          ("config_input_dtype", "config_input_dtype"),
                          ("output_dtype", "output_dtype"),
                          ("actual_input_digest", "actual_input_digest"),
                          ("status", "status")], dtype_rows))
    files = [{"kind": _kind(path), "path": str(path)}
             for path in sorted(main_path.parent.iterdir()) if path.is_file()]
    blocks.append(_table("[remaining_operator_case] evidence files",
                         [("kind", "kind"), ("path", "path")], files))
    return "\n\n".join(blocks)


def _emit_registered():
    for main_path in _REGISTERED:
        try:
            text = render_main_csv(main_path)
            if not text:
                continue
            print(text)
            for log_path in main_path.parent.glob("operator_*_full.log"):
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(text + "\n")
        except Exception as error:
            print(f"[remaining_operator_case] terminal summary unavailable error={error}")


def register_main_csv(path):
    path = pathlib.Path(path)
    if path.name.startswith("operator_main_results_") and path not in _REGISTERED:
        _REGISTERED.append(path)


atexit.register(_emit_registered)
