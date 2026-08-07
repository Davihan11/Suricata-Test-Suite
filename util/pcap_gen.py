"""
Author(s):  Dávid Hanko <david.hanko@cesnet.cz>

Copyright: (C) 2026 CESNET, z.s.p.o.
SPDX-License-Identifier: BSD-3-Clause

Synthetic PCAP generator for Suricata performance tests.

Generates stateless (STL) traffic as PCAP files that are replayed by TRex.
Unlike the real-traffic profiles (HTTP/HTTPS/NFS), these give precise control
over packet sizes, flow counts, fragmentation and encapsulation, which is what
the synthetic performance tests need.

All generators write into a caller-supplied output directory (usually
``tmp/``) and return the path to the generated PCAP. The returned PCAP is then
replayed via an STL profile (see ``assets/trex/traffic_profiles/synthetic_stl_trex_profile.py``).

The generators are intentionally pure (no Suricata/TRex dependencies) so they
can be unit-tested and reused.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from scapy.all import (
    Ether,
    IP,
    IPv6,
    TCP,
    UDP,
    Raw,
    GRE,
    wrpcap,
    fragment,
    fragment6,
)

logger = logging.getLogger(__name__)

# Default L2/L3 addressing used across all synthetic pcaps. These are
# deliberately in the RFC 5737 / documentation ranges so they never collide
# with real traffic on the test network.
DEFAULT_SRC_MAC = "02:00:00:00:00:01"
DEFAULT_DST_MAC = "02:00:00:00:00:02"
DEFAULT_SRC_IP = "10.0.0.1"
DEFAULT_DST_IP = "10.0.0.2"
DEFAULT_SRC_IP6 = "2001:db8::1"
DEFAULT_DST_IP6 = "2001:db8::2"
DEFAULT_SPORT = 12345
DEFAULT_DPORT = 80

# Ethernet frame overhead (without payload): preamble is not captured, so the
# on-wire frame is 14 (L2) + 20 (IPv4 hdr) + 20 (TCP hdr) = 54 bytes minimum.
# We use these to pad packets to an exact total frame size.
L2_IPV4_TCP_OVERHEAD = 14 + 20 + 20  # 54
L2_IPV4_UDP_OVERHEAD = 14 + 20 + 8  # 42
L2_IPV6_TCP_OVERHEAD = 14 + 40 + 20  # 74
L2_IPV6_UDP_OVERHEAD = 14 + 40 + 8  # 62


def _pad(pkt, total_size: int) -> bytes:
    """Pad a packet's payload so the whole frame is exactly *total_size* bytes."""
    raw = bytes(pkt)
    if len(raw) > total_size:
        raise ValueError(
            f"packet is {len(raw)}B but target size is {total_size}B; "
            "increase the target size or reduce headers"
        )
    return raw + b"\x00" * (total_size - len(raw))


def _flow_ip(net: str, flow_id: int) -> str:
    """
    Map a flow_id to a unique IPv4 address inside a /16 subnet.

    ``net`` must be a ``"A.B.0.0"`` /16 prefix. The flow_id is split into the
    third and fourth octets, giving up to 65536 unique addresses per /16.
    """
    a, b, _, _ = net.split(".")
    return f"{a}.{b}.{flow_id // 256 % 256}.{flow_id % 256}"


def _write(pcap_path: Path, packets: Iterable[bytes]) -> Path:
    """Write raw frames to a PCAP file, creating the parent dir if needed."""
    pcap_path = Path(pcap_path)
    pcap_path.parent.mkdir(parents=True, exist_ok=True)
    # wrpcap accepts Ether frames; we pass raw bytes wrapped in Ether.
    frames = [Ether(raw) for raw in packets]
    wrpcap(str(pcap_path), frames)
    logger.info("Wrote %d packets to %s", len(frames), pcap_path)
    return pcap_path


# ---------------------------------------------------------------------------
# Packet sizes
# ---------------------------------------------------------------------------

def gen_packet_sizes(
    out_dir: Path,
    sizes: Iterable[int] = (64, 512, 1400, 9000),
    packets_per_size: int = 1000,
    proto: str = "tcp",
) -> dict[int, Path]:
    """
    Generate one PCAP per fixed packet size.

    Args:
        out_dir: Directory to write the PCAPs into.
        sizes: Total on-wire frame sizes in bytes (64B, 512B, 1400B, jumbo 9000B).
        packets_per_size: Number of packets per size.
        proto: "tcp" or "udp".

    Returns:
        Mapping of frame size -> path to the generated PCAP.
    """
    out_dir = Path(out_dir)
    result: dict[int, Path] = {}
    for size in sizes:
        packets = []
        for i in range(packets_per_size):
            if proto == "tcp":
                pkt = (
                    Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                    / IP(src=DEFAULT_SRC_IP, dst=DEFAULT_DST_IP)
                    / TCP(sport=DEFAULT_SPORT, dport=DEFAULT_DPORT, flags="S")
                )
            else:
                pkt = (
                    Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                    / IP(src=DEFAULT_SRC_IP, dst=DEFAULT_DST_IP)
                    / UDP(sport=DEFAULT_SPORT, dport=DEFAULT_DPORT)
                )
            packets.append(_pad(pkt, size))
        path = _write(out_dir / f"packet_size_{size}.pcap", packets)
        result[size] = path
    return result


# ---------------------------------------------------------------------------
# Flow churn (many short flows)
# ---------------------------------------------------------------------------

def gen_many_short_flows(
    out_dir: Path,
    num_flows: int = 100_000,
    packets_per_flow: int = 5,
    src_net: str = "10.1.0.0",
    dst_net: str = "10.2.0.0",
) -> Path:
    """
    Generate a PCAP with *num_flows* distinct 5-tuples, each carrying
    *packets_per_flow* packets. This stresses flow hash inserts, allocations
    and expiry evictions.

    Each flow uses a unique (src_ip, dst_ip, sport, dport) tuple. The flows are
    emitted sequentially so that the first flow's packets are all sent before
    the next flow starts (worst case for flow table churn).
    """
    out_dir = Path(out_dir)
    packets = []
    for flow_id in range(num_flows):
        src_ip = _flow_ip(src_net, flow_id)
        dst_ip = _flow_ip(dst_net, flow_id)
        sport = DEFAULT_SPORT + (flow_id % 1000)
        dport = DEFAULT_DPORT + (flow_id % 1000)
        for _ in range(packets_per_flow):
            pkt = (
                Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                / IP(src=src_ip, dst=dst_ip)
                / TCP(sport=sport, dport=dport, flags="S")
            )
            packets.append(bytes(pkt))
    return _write(out_dir / f"many_short_flows_{num_flows}.pcap", packets)


# ---------------------------------------------------------------------------
# Single flow vs many flows
# ---------------------------------------------------------------------------

def gen_single_flow(
    out_dir: Path,
    num_packets: int = 1_000_000,
) -> Path:
    """
    Generate a PCAP with a single 5-tuple pounded repeatedly. This isolates
    single-NIC-queue behaviour: one flow hashes to one queue, so only one
    worker thread does the work.
    """
    out_dir = Path(out_dir)
    pkt = (
        Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
        / IP(src=DEFAULT_SRC_IP, dst=DEFAULT_DST_IP)
        / TCP(sport=DEFAULT_SPORT, dport=DEFAULT_DPORT, flags="S")
    )
    packets = [bytes(pkt)] * num_packets
    return _write(out_dir / f"single_flow_{num_packets}.pcap", packets)


def gen_many_flows(
    out_dir: Path,
    num_flows: int = 100_000,
    packets_per_flow: int = 10,
    src_net: str = "10.3.0.0",
    dst_net: str = "10.4.0.0",
) -> Path:
    """
    Generate a PCAP with *num_flows* parallel flows, interleaved so they spread
    across all NIC queues. Each flow carries *packets_per_flow* packets.
    """
    out_dir = Path(out_dir)
    # Build one packet per flow, then round-robin so all flows progress together.
    flow_packets: list[bytes] = []
    for flow_id in range(num_flows):
        src_ip = _flow_ip(src_net, flow_id)
        dst_ip = _flow_ip(dst_net, flow_id)
        sport = DEFAULT_SPORT + (flow_id % 1000)
        dport = DEFAULT_DPORT + (flow_id % 1000)
        pkt = (
            Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
            / IP(src=src_ip, dst=dst_ip)
            / TCP(sport=sport, dport=dport, flags="S")
        )
        flow_packets.append(bytes(pkt))

    packets = []
    for _ in range(packets_per_flow):
        packets.extend(flow_packets)
    return _write(out_dir / f"many_flows_{num_flows}.pcap", packets)


# ---------------------------------------------------------------------------
# Fragmentation
# ---------------------------------------------------------------------------

def gen_fragmentation(
    out_dir: Path,
    frag_size: int = 1400,
    num_frags: int = 4,
    ipv6: bool = False,
    out_of_order: bool = False,
    num_flows: int = 1000,
) -> Path:
    """
    Generate a PCAP of fragmented IP packets.

    Args:
        out_dir: Output directory.
        frag_size: Fragment payload size in bytes.
        num_frags: Number of fragments per original packet.
        ipv6: Use IPv6 instead of IPv4.
        out_of_order: Shuffle fragment order (out-of-order reassembly).
        num_flows: Number of distinct fragmented packets (each a unique flow).
    """
    out_dir = Path(out_dir)
    packets: list[bytes] = []
    for flow_id in range(num_flows):
        src_ip = _flow_ip("10.5.0.0", flow_id)
        dst_ip = _flow_ip("10.6.0.0", flow_id)
        payload = b"A" * (frag_size * num_frags)

        if ipv6:
            base = (
                Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                / IPv6(src=f"2001:db8:{flow_id % 65536:x}::1", dst=f"2001:db8:{flow_id % 65536:x}::2")
                / TCP(sport=DEFAULT_SPORT, dport=DEFAULT_DPORT)
                / Raw(load=payload)
            )
            frags = fragment6(base, fragSize=frag_size)
        else:
            base = (
                Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                / IP(src=src_ip, dst=dst_ip)
                / TCP(sport=DEFAULT_SPORT, dport=DEFAULT_DPORT)
                / Raw(load=payload)
            )
            frags = fragment(base, fragsize=frag_size)

        if out_of_order:
            # Reverse the fragment order to force out-of-order reassembly.
            frags = list(reversed(frags))
        packets.extend(bytes(f) for f in frags)

    tag = "ipv6" if ipv6 else "ipv4"
    order = "ooo" if out_of_order else "inorder"
    return _write(
        out_dir / f"frag_{tag}_{order}_frag{frag_size}_n{num_frags}.pcap", packets
    )


# ---------------------------------------------------------------------------
# VPN / encapsulation
# ---------------------------------------------------------------------------

def gen_vpn(
    out_dir: Path,
    encap: str = "gre",
    num_flows: int = 1000,
    packets_per_flow: int = 10,
) -> Path:
    """
    Generate a PCAP of encapsulated (VPN) traffic.

    Args:
        out_dir: Output directory.
        encap: One of "gre", "ipip", "esp", "wireguard".
        num_flows: Number of distinct inner flows.
        packets_per_flow: Packets per inner flow.
    """
    out_dir = Path(out_dir)
    packets: list[bytes] = []
    for flow_id in range(num_flows):
        inner_src = _flow_ip("192.168.0.0", flow_id)
        inner_dst = _flow_ip("192.168.1.0", flow_id)
        inner = (
            IP(src=inner_src, dst=inner_dst)
            / TCP(sport=DEFAULT_SPORT, dport=DEFAULT_DPORT, flags="S")
        )
        for _ in range(packets_per_flow):
            if encap == "gre":
                pkt = (
                    Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                    / IP(src=DEFAULT_SRC_IP, dst=DEFAULT_DST_IP)
                    / GRE(proto=0x0800)
                    / inner
                )
            elif encap == "ipip":
                pkt = (
                    Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                    / IP(src=DEFAULT_SRC_IP, dst=DEFAULT_DST_IP, proto=4)
                    / inner
                )
            elif encap == "esp":
                # ESP is encrypted; we only model the outer IP + ESP header with
                # a fixed SPI and opaque payload (no real crypto needed for a
                # CPU-cost test).
                pkt = (
                    Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                    / IP(src=DEFAULT_SRC_IP, dst=DEFAULT_DST_IP, proto=50)
                    / Raw(load=b"\x00\x00\x00\x01" + b"\x00" * 32)
                )
            elif encap == "wireguard":
                # WireGuard uses UDP port 51820 with an opaque encrypted payload.
                pkt = (
                    Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                    / IP(src=DEFAULT_SRC_IP, dst=DEFAULT_DST_IP)
                    / UDP(sport=51820, dport=51820)
                    / Raw(load=b"\x04" + b"\x00" * 64)
                )
            else:
                raise ValueError(f"unknown encapsulation: {encap}")
            packets.append(bytes(pkt))
    return _write(out_dir / f"vpn_{encap}.pcap", packets)


# ---------------------------------------------------------------------------
# Signature / regex matching
# ---------------------------------------------------------------------------

def gen_signature_traffic(
    out_dir: Path,
    num_flows: int = 1000,
    packets_per_flow: int = 10,
    poison: str = "EVILPATTERN",
) -> Path:
    """
    Generate HTTP traffic carrying a distinctive *poison* string in the URI,
    designed to match a regex rule. Used together with a large ruleset to
    stress signature/regex matching.
    """
    out_dir = Path(out_dir)
    packets: list[bytes] = []
    for flow_id in range(num_flows):
        src_ip = _flow_ip("10.7.0.0", flow_id)
        dst_ip = _flow_ip("10.8.0.0", flow_id)
        http_req = (
            f"GET /{poison}/{flow_id} HTTP/1.1\r\n"
            f"Host: www.example{flow_id % 100}.com\r\n"
            "User-Agent: Mozilla/5.0\r\n\r\n"
        ).encode()
        for _ in range(packets_per_flow):
            pkt = (
                Ether(src=DEFAULT_SRC_MAC, dst=DEFAULT_DST_MAC)
                / IP(src=src_ip, dst=dst_ip)
                / TCP(sport=DEFAULT_SPORT, dport=80, flags="PA")
                / Raw(load=http_req)
            )
            packets.append(bytes(pkt))
    return _write(out_dir / "signature_traffic.pcap", packets)
