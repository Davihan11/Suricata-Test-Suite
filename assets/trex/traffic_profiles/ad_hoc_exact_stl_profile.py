"""
Author(s):  Dávid Hanko <david.hanko@cesnet.cz>

Copyright: (C) 2026 CESNET, z.s.p.o.
SPDX-License-Identifier: BSD-3-Clause

TRex profile for exact packet count STL transmission using STLTXSingleBurst.
"""

from lbr_testsuite.trex import TRexManager
from lbr_trex_client.interactive.trex.stl.trex_stl_client import STLClient
from pytest import FixtureRequest

# Must import from trex.stl (not lbr_trex_client.interactive.trex.stl)
# because STLClient.add_streams() checks isinstance against trex.stl classes
from trex.stl.trex_stl_packet_builder_scapy import STLPktBuilder
from trex.stl.trex_stl_streams import STLStream, STLTXSingleBurst

from .trex_client_manager import BaseTrexClientManager, PcapList, TrexMode


class AdHocExactStlProfile(BaseTrexClientManager, pcaps=[]):
    def __init__(
        self,
        pcaps: PcapList,
        manager: TRexManager,
        request: FixtureRequest,
        target_mac: str,
        target_vlan: int = 0,
    ):
        self.profile_pcaps = pcaps
        self.request = request
        super().__init__(manager, request, target_mac, target_vlan, mode=TrexMode.STL)

    def run(self, blocking=True) -> None:
        """
        Start traffic using STLStream with STLTXSingleBurst mode
        instead of the default push_remote loop.
        """
        client: STLClient = self.stl_generator.get_handler()

        base_pps = self.request.config.getoption("--trex-pps")
        total_pkts = self.request.config.getoption("--trex-total-packets")

        # scale the send rate by the traffic multiplier so that binary search
        # (and multiplier enumeration) can vary the speed; the packet count
        # stays fixed
        multiplier = self.multiplier if self.multiplier is not None else 1.0
        pps = base_pps * multiplier

        streams = []
        for i, pcap in enumerate(self.pcaps):
            pcap_path = str(self.PCAP_PATH_PREFIX / pcap[0])
            stream = STLStream(
                name=f"S{i}",
                packet=STLPktBuilder(pkt=pcap_path),
                mode=STLTXSingleBurst(pps=pps, total_pkts=total_pkts),
            )
            streams.append(stream)

        client.add_streams(streams, ports=[0])
        # STLTXSingleBurst sends exactly `total_pkts` packets and stops on its
        # own, so the duration is irrelevant (like in STL mode). Use -1
        # (unlimited) so the burst is never truncated at low multipliers.
        client.start(ports=[0], duration=-1)

        if blocking:
            self.wait_on_traffic()
