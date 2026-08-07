"""
Author(s):  Dávid Hanko <david.hanko@cesnet.cz>

Copyright: (C) 2026 CESNET, z.s.p.o.
SPDX-License-Identifier: BSD-3-Clause

Suricata testing module.

Tests Suricata throughput under heavy IP fragmentation/defragmentation.
Heavy fragmentation collapses throughput because every packet must be
reassembled before inspection. Parametrised over IPv4 vs IPv6 and in-order
vs out-of-order fragment arrival. The defrag tracker/memcap settings are
tuned via param.py.
"""

import pytest
import signal
import logging

from pathlib import Path
from typing import List

from lbr_testsuite import trex
from util.suricata_manager import Suricata_manager
from util.suri_util import TestInfo, get_drop_rate
from util.pcap_gen import gen_fragmentation
from assets.trex.traffic_profiles.ad_hoc_stl_trex_profile import AdHocStlProfile
from conftest import kill_pytest, get_trex_multi, suri_interface_bind, Suri_conf
from util.multiplier_iterator import multiplier_iterator_create
from util.test_runner import TrexTestRun

logger = logging.getLogger(__name__)

FRAG_SIZE = 1400
NUM_FRAGS = 4
NUM_FLOWS = 1000


@pytest.mark.parametrize(
    "rules_config",
    [
        {"name": "norules", "path": "/dev/null/"},
        {"name": "rules", "path": "/var/lib/suricata/rules/suricata.rules"},
    ],
    ids=["norules", "rules"],
)
def test_fragmentation(
    request: pytest.FixtureRequest,
    trex_generators: dict,
    result_path: str,
    suricata_tmp_stats_path: str,
    utilized_programs_info: dict,
    params: dict,
    suri_conf: Suri_conf,
    get_settings_file: str,
    get_traffic_duration: int,
    get_heatup_duration: int,
    rules_config: dict,
    get_target_mac: str,
    get_target_vlan: int,
    b_search: dict | None,
):
    trex_manager: trex.TRexManager = trex.TRexManager(
        trex.TRexMachinesPool(trex_generators)
    )

    suri_daemon: Suricata_manager = Suricata_manager(
        request,
        suricata_tmp_stats_path,
        interface=suri_interface_bind(request)[0],
        capture_mode=suri_interface_bind(request)[1],
        conf_file=suri_conf.conf_file.with_params(params).build(),
        rules_file=rules_config["path"],
    )
    signal.signal(signal.SIGINT, kill_pytest)

    test_info = TestInfo(
        result_path=result_path,
        traffic_duration=get_traffic_duration,
        heatup_duration=get_heatup_duration,
        suricata_path_to_bin=suri_daemon.get_path_to_binary(),
        suricata_rules_paths=[suri_daemon.rules_file],
        suricata_config_path=suri_daemon.conf_file,
        utilized_programs_info=utilized_programs_info,
    )

    tmp_dir = Path(request.node.path).parent / "tmp"
    pcap_path = gen_fragmentation(
        tmp_dir,
        frag_size=FRAG_SIZE,
        num_frags=NUM_FRAGS,
        ipv6=False,
        out_of_order=False,
        num_flows=NUM_FLOWS,
    )

    trex_client = AdHocStlProfile(
        [(pcap_path, 1.0)], trex_manager, request, get_target_mac, get_target_vlan
    )

    test_variant_name = f"{suri_conf.test_name}_{rules_config['name']}"
    trex_multipliers: List[float] = get_trex_multi(
        get_settings_file, suri_conf.server, suri_conf.pcie, test_variant_name
    )

    tester = TrexTestRun(trex_client, suri_daemon, test_info, params, request)

    mult_iter = multiplier_iterator_create(b_search, trex_multipliers)
    for multiplier in mult_iter:
        logger.progress(
            f"multiplier {multiplier:.4f} | ipv4 inorder | params={params}"
        )
        tester.execute(multiplier)
        mult_iter.set_result(get_drop_rate())
        logger.info("Run ended.")

    if mult_iter.result is not None:
        logger.progress(
            f"Maximum multiplier found is: {mult_iter.result:.4f}. | ipv4 inorder | params={params}\n\n"
        )
    else:
        logger.progress(
            f"Enumeration complete. | ipv4 inorder | params={params}\n\n"
        )
