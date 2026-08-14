"""
Author(s):  Matyáš Sedmidubský <matyas.sedmidubsky@cesnet.cz>

Copyright: (C) 2026 CESNET, z.s.p.o.
SPDX-License-Identifier: BSD-3-Clause
"""

import logging
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Sequence, Tuple

import pytest
from lbr_testsuite.executable import executable, remote_executor

logger = logging.getLogger(__name__)


class TrexMode(Enum):
    STL = 0
    ASTF = 1
    STF = 2


PcapList = Sequence[Tuple[str, int | float]]


def merge_pcaps(
    pcap_paths: list[Path],
    weights: list[float],
    out_path: Path,
    max_packets: int = 100_000,
) -> Path:
    """
    Merges several pcaps into one by interleaving their packets
    proportionally to `weights` (weighted round-robin).

    Packets are read and written one at a time (streaming) so that large
    pcaps are not loaded fully into memory.

    A short pcap with a large weight would otherwise be exhausted quickly and
    become under-represented relative to its expected weight. To avoid this,
    exhausted streams are restarted (looped back to the beginning) until the
    merged output reaches `max_packets` packets, so every source keeps
    contributing proportionally for the whole merged file.

    Returns `out_path` with the merged pcap written.
    """
    from scapy.all import PcapReader, PcapWriter

    if len(pcap_paths) != len(weights):
        raise ValueError("pcap_paths and weights must have the same length")
    if any(w <= 0 for w in weights):
        raise ValueError("all weights must be positive")
    if max_packets <= 0:
        raise ValueError("max_packets must be positive")

    total_w = sum(weights)
    if total_w <= 0:
        raise ValueError("sum of weights must be positive")

    # weighted round-robin: each source emits packets per round proportional to
    # its share of the total weight, scaled so the smallest positive quota emits 1
    quotas = [w / total_w for w in weights]
    min_q = min(q for q in quotas if q > 0)
    # zero-weight sources emit nothing; others emit at least 1 packet per round
    per_round = [0 if q <= 0 else max(1, round(q / min_q)) for q in quotas]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def open_readers() -> list:
        return [PcapReader(str(p)) for p in pcap_paths]

    readers = open_readers()
    iters = [iter(r) for r in readers]
    # sources that are empty (or become empty on restart) are skipped entirely
    empty = [False] * len(iters)
    total = 0
    try:
        with PcapWriter(str(out_path), append=False, sync=True) as writer:
            while total < max_packets:
                # if every source is exhausted/empty, there is nothing left to
                # write; stop to avoid an infinite loop
                if all(per_round[si] == 0 or empty[si] for si in range(len(iters))):
                    break
                for si, it in enumerate(iters):
                    if per_round[si] == 0 or empty[si]:
                        continue
                    for _ in range(per_round[si]):
                        if total >= max_packets:
                            break
                        try:
                            pkt = next(it)
                        except StopIteration:
                            # restart the exhausted stream so it keeps
                            # contributing proportionally to its weight
                            readers[si].close()
                            readers[si] = PcapReader(str(pcap_paths[si]))
                            iters[si] = iter(readers[si])
                            # point the local `it` at the fresh iterator so the
                            # rest of this round reads from the new reader
                            # instead of the just-closed one
                            it = iters[si]
                            try:
                                pkt = next(it)
                            except StopIteration:
                                # empty pcap: nothing to contribute, skip it
                                empty[si] = True
                                break
                        writer.write(pkt)
                        total += 1
    finally:
        for r in readers:
            r.close()

    logger.info(
        "Merged %d pcaps into %s (%d packets, weights=%s)",
        len(pcap_paths),
        out_path.name,
        total,
        weights,
    )
    return out_path


def send_to_remote(source: Path, hostname: str, destination: Path | None = None):
    if destination is None:
        destination = source

    logger.debug("Sending %s to remote %s:%s", source, hostname, destination)
    subprocess.run(
        [
            "rsync",
            "-z",
            "--checksum",
            "--update",
            str(source),
            f"{os.environ['USER']}@{hostname}:{str(destination)}",
        ],
        check=True,
    )


def mkdir_remote(dir: Path, hostname: str):
    logger.debug("Creating remote directory %s on %s", dir, hostname)
    executor = remote_executor.RemoteExecutor(host=hostname, user=os.environ["USER"])
    mkdir = executable.Tool(
        f"mkdir -p '{str(dir)}' && chmod 777 '{str(dir)}'",
        executor=executor,
        sudo=True,
    )
    mkdir.run()


def get_trex_mac(hostname: str, pci: str, trex_version: str):
    logger.debug(
        "Getting TRex MAC for %s on %s (version %s)", pci, hostname, trex_version
    )
    executor = remote_executor.RemoteExecutor(host=hostname, user=os.environ["USER"])
    get_mac = executable.Tool(
        f"cd /opt/trex/{trex_version} && ./dpdk_setup_ports.py -t | grep {pci[5:]} | awk '{{print $8}}'",
        executor=executor,
        sudo=True,
    )
    stdout, _ = get_mac.run()

    stdout = str(stdout).strip()
    assert len(stdout) == 17, f"Couldn't get MAC address for {pci}"
    logger.debug("TRex MAC for %s: %s", pci, stdout)
    return stdout


def str_to_trex_mode(mode: str) -> TrexMode | None:
    match mode.lower():
        case "astf":
            return TrexMode.ASTF
        case "stf":
            return TrexMode.STF
        case "stl":
            return TrexMode.STL
        case _:
            return None


def get_trex_mode(request, available_modes) -> TrexMode:
    """
    Selects a TRex mode out of `available_modes` based on the
    `--prefer-trex-mode` and `--force-trex-mode` flags.

    `available_modes: List[TrexMode]` should be in descending order by priority.

    Automatically skips tests with no usable TRex modes.
    """
    if (
        not isinstance(available_modes, list)
        or len(available_modes) < 1
        or not isinstance(available_modes[0], TrexMode)
    ):
        raise ValueError("available_modes must be a list of at least one TrexMode")

    forced_mode = request.config.getoption("--force-trex-mode")
    if forced_mode is not None:
        mode_enum = str_to_trex_mode(forced_mode)
        if mode_enum is None:
            raise ValueError(f"{forced_mode} is not a valid TRex mode")

        if mode_enum in available_modes:
            logger.debug("Using forced TRex mode: %s", forced_mode)
            return mode_enum
        else:
            pytest.skip(f"{forced_mode} is not supported by this test")

    preferred_mode = request.config.getoption("--prefer-trex-mode")
    if preferred_mode is not None:
        mode_enum = str_to_trex_mode(preferred_mode)
        if mode_enum is None:
            raise ValueError(f"{preferred_mode} is not a valid TRex mode")

        if mode_enum in available_modes:
            logger.debug("Using preferred TRex mode: %s", preferred_mode)
            return mode_enum
        else:
            logger.debug(
                "Preferred TRex mode %s unavailable, falling back to %s",
                preferred_mode,
                available_modes[0].name,
            )
            return available_modes[0]

    logger.debug("Using default TRex mode: %s", available_modes[0].name)
    return available_modes[0]
