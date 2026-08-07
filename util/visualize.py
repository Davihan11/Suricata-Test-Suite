#!/usr/bin/python3
"""
Author(s):  Dávid Hanko <david.hanko@cesnet.cz>

Copyright: (C) 2026 CESNET, z.s.p.o.
SPDX-License-Identifier: BSD-3-Clause

Generic visualisation of Suricata counter values from stats/eve.json files.

Reads one or more eve.json (or eve-stats.json) files, filters the per-line
``stats`` records with a universal jq filter, extracts the requested counter
paths and plots them over time (using ``.stats.uptime`` as the x-axis).

Example:
    python3 util/visualize.py -i stats.json -p flow.memuse -p stream.memuse \\
        -f 'uptime > 30'

Multiple input files can be given (each ``-i`` adds another file); records from
all of them are merged and plotted together.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import jq
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# jq filter that selects only the "stats" event records from a JSON-lines file.
_STATS_SELECTOR = "select(.event_type == \"stats\")"


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Suricata counter values from stats/eve.json files. "
            "Each input file is treated as JSON-lines; only records with "
            "event_type == 'stats' are considered."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        required=True,
        metavar="FILE",
        help=(
            "Input stats/eve.json file (JSON-lines). May be given multiple "
            "times to merge several files into one plot."
        ),
    )
    parser.add_argument(
        "-p",
        "--path",
        action="append",
        required=True,
        metavar="PATH",
        help=(
            "Counter path relative to .stats, e.g. 'flow.memuse' or "
            "'capture.dpdk.imissed'. May be given multiple times to plot "
            "several counters."
        ),
    )
    parser.add_argument(
        "-f",
        "--filter",
        default="",
        metavar="JQ",
        help=(
            "Universal jq filter applied to each stats record (e.g. "
            "'uptime > 30'). The filter is evaluated against the .stats "
            "object; records for which it does not match are skipped."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="Optional output file for the plot (e.g. graph.png).",
    )
    parser.add_argument(
        "-x",
        "--x-axis",
        default="uptime",
        metavar="PATH",
        help=(
            "Counter path (relative to .stats) used as the x-axis. "
            "Default: 'uptime'."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        help=(
            "Plot the difference between consecutive samples instead of the "
            "raw cumulative counter value. Suricata stats counters are "
            "cumulative (they only increase), so plotting them directly gives "
            "a straight line; --delta shows the per-interval rate instead."
        ),
    )
    parser.add_argument(
        "--by-multiplier",
        action="store_true",
        help=(
            "Plot a summary value of each counter against the traffic "
            "multiplier instead of against uptime. The input must be a test "
            "directory containing multiplier_*/eve-stats.json subdirectories. "
            "For each multiplier the counter's final value is plotted (or, "
            "with --delta, its peak per-interval rate)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _read_stats_records(path: Path) -> List[dict]:
    """Read all ``stats`` event records from a JSON-lines file."""
    records: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line in %s", path)
                continue
            if obj.get("event_type") == "stats":
                records.append(obj)
    return records


def _apply_filter(records: List[dict], filter_expr: str) -> List[dict]:
    """Keep only records whose ``.stats`` matches the given jq filter."""
    if not filter_expr:
        return records
    # jq parses a leading bare identifier (e.g. ``uptime > 30``) as a function
    # call, not a field access. Prepend ``.`` so the common ``uptime > 30``
    # form works as ``.uptime > 30`` without requiring the user to type it.
    filter_expr = _normalize_filter(filter_expr)
    compiled = jq.compile(filter_expr)
    filtered: List[dict] = []
    for rec in records:
        stats = rec.get("stats")
        if stats is None:
            continue
        try:
            result = compiled.input(stats).first()
        except Exception as exc:  # jq raises on invalid input for the filter
            logger.warning("jq filter failed on a record: %s", exc)
            continue
        # jq returns None for a non-matching select() and False/0 for a
        # boolean/arithmetic expression that evaluates to false.
        if result is None or result is False or result == 0:
            continue
        filtered.append(rec)
    return filtered


def _normalize_filter(filter_expr: str) -> str:
    """Prepend ``.`` to a leading bare identifier in a jq filter.

    ``uptime > 30`` becomes ``.uptime > 30`` so field access works without the
    user having to type the leading dot. Filters that already start with a
    jq construct (``.``, ``[``, ``(``, ``{``, ``select``, etc.) are left as-is.
    """
    stripped = filter_expr.lstrip()
    if not stripped:
        return filter_expr
    first = stripped[0]
    if first.isalpha() and not stripped.startswith(("select", "if", "reduce")):
        return "." + filter_expr
    return filter_expr


def _extract_series(records: List[dict], path: str) -> List[Tuple[float, float]]:
    """Extract (x, y) points for a counter path across all records."""
    compiled = jq.compile(f".stats.{path}")
    points: List[Tuple[float, float]] = []
    for rec in records:
        try:
            value = compiled.input(rec).first()
        except Exception as exc:
            logger.warning("Failed to extract %r: %s", path, exc)
            continue
        if value is None:
            continue
        points.append(value)
    return points


def _to_delta(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Convert a cumulative series into per-interval deltas.

    Suricata stats counters are cumulative (they only increase), so plotting
    them directly yields a straight line. This converts each point to the
    difference from the previous sample, i.e. the amount accumulated in that
    interval (a proxy for the rate). The first point is dropped since it has
    no predecessor.
    """
    deltas: List[Tuple[float, float]] = []
    for i in range(1, len(points)):
        x_prev, y_prev = points[i - 1]
        x_cur, y_cur = points[i]
        deltas.append((x_cur, y_cur - y_prev))
    return deltas


def _find_multiplier_dirs(test_dir: Path) -> List[Tuple[float, Path]]:
    """Find ``multiplier_*/eve-stats.json`` subdirectories under a test dir.

    Returns a list of ``(multiplier, eve_stats_path)`` sorted by multiplier.
    """
    found: List[Tuple[float, Path]] = []
    for sub in sorted(test_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("multiplier_"):
            continue
        try:
            multiplier = float(sub.name.split("_", 1)[1])
        except ValueError:
            logger.warning("Skipping non-numeric multiplier dir: %s", sub.name)
            continue
        stats_path = sub / "eve-stats.json"
        if stats_path.is_file():
            found.append((multiplier, stats_path))
    found.sort(key=lambda item: item[0])
    return found


def _summarize_series(points: List[Tuple[float, float]], use_delta: bool) -> float:
    """Reduce a counter series to a single summary value.

    Without ``use_delta`` returns the final (last) cumulative value. With
    ``use_delta`` returns the peak per-interval rate (max delta), which is a
    better proxy for the sustained throughput at a given multiplier.
    """
    if not points:
        return float("nan")
    if use_delta:
        deltas = _to_delta(points)
        return max((d[1] for d in deltas), default=float("nan"))
    return float(points[-1][1])


def _plot(
    series: List[Tuple[str, List[Tuple[float, float]]]],
    x_label: str,
    title: str | None,
    output: str | None,
) -> None:
    fig, ax = plt.subplots()
    for name, points in series:
        if not points:
            logger.warning("No data points for %r", name)
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, label=name, marker="o", markersize=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel("counter value")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    if output:
        fig.savefig(output)
        logger.info("Plot saved to %s", output)
    else:
        plt.show()


def _main_by_multiplier(args: argparse.Namespace) -> int:
    """Plot a summary value of each counter against the traffic multiplier.

    Each ``-i`` argument is a test directory containing ``multiplier_*/``
    subdirectories with an ``eve-stats.json`` each. For every multiplier the
    requested counters are reduced to a single summary value (final value, or
    peak rate with ``--delta``) and plotted against the multiplier.
    """
    series: List[Tuple[str, List[Tuple[float, float]]]] = []
    for input_path in args.input:
        test_dir = Path(input_path)
        if not test_dir.is_dir():
            logger.error("Test directory not found: %s", test_dir)
            return 1
        multipliers = _find_multiplier_dirs(test_dir)
        if not multipliers:
            logger.error(
                "No multiplier_*/eve-stats.json found under %s", test_dir
            )
            return 1
        logger.info(
            "Found %d multiplier runs under %s", len(multipliers), test_dir
        )

        label = test_dir.name if len(args.input) > 1 else ""
        for counter_path in args.path:
            xs: List[float] = []
            ys: List[float] = []
            for multiplier, stats_path in multipliers:
                records = _read_stats_records(stats_path)
                if args.filter:
                    records = _apply_filter(records, args.filter)
                points = list(
                    zip(
                        _extract_series(records, args.x_axis),
                        _extract_series(records, counter_path),
                    )
                )
                summary = _summarize_series(points, args.delta)
                xs.append(multiplier)
                ys.append(summary)
            name = f"{counter_path} ({label})" if label else counter_path
            series.append((name, list(zip(xs, ys))))

    if not series:
        logger.error("No data to plot.")
        return 1

    _plot(series, "multiplier", args.title, args.output)
    return 0


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.by_multiplier:
        return _main_by_multiplier(args)

    # Process each input file independently so that, when several files are
    # given, their series are labelled separately (e.g. comparing two
    # multipliers produces distinguishable lines per file).
    series: List[Tuple[str, List[Tuple[float, float]]]] = []
    for input_path in args.input:
        path = Path(input_path)
        if not path.is_file():
            logger.error("Input file not found: %s", path)
            return 1
        records = _read_stats_records(path)
        logger.info("Read %d stats records from %s", len(records), path)
        if not records:
            logger.warning("No stats records in %s; skipping.", path)
            continue

        # Apply the universal jq filter to this file's records.
        if args.filter:
            before = len(records)
            records = _apply_filter(records, args.filter)
            logger.info(
                "Filter %r kept %d/%d records in %s",
                args.filter,
                len(records),
                before,
                path,
            )
        if not records:
            logger.warning("No records remain in %s after filtering; skipping.", path)
            continue

        # Extract the x-axis series once per file.
        x_points = _extract_series(records, args.x_axis)
        if not x_points:
            logger.warning(
                "Could not extract x-axis path %r from %s; skipping.",
                args.x_axis,
                path,
            )
            continue

        # One series per requested path, labelled with the source file so that
        # multiple inputs are distinguishable on the same plot. Use the parent
        # directory name (e.g. "multiplier_2.5") when present, since input
        # files are often all named eve-stats.json.
        label = path.parent.name if len(args.input) > 1 else ""
        for counter_path in args.path:
            y_points = _extract_series(records, counter_path)
            if len(y_points) != len(x_points):
                logger.warning(
                    "Path %r produced %d points but x-axis has %d in %s; "
                    "pairing by index may be misaligned.",
                    counter_path,
                    len(y_points),
                    len(x_points),
                    path,
                )
            points = list(zip(x_points, y_points))
            if args.delta:
                points = _to_delta(points)
            name = f"{counter_path} ({label})" if label else counter_path
            series.append((name, points))

    if not series:
        logger.error("No data to plot from the given input files.")
        return 1

    _plot(series, args.x_axis, args.title, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
