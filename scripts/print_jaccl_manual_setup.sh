#!/usr/bin/env bash
set -euo pipefail

NODE0="${NODE0:-node-0.local}"
NODE1="${NODE1:-node-1.local}"
TB_IFACE_NODE0="${TB_IFACE_NODE0:-en4}"
TB_IFACE_NODE1="${TB_IFACE_NODE1:-en2}"
TB_IP_NODE0="${TB_IP_NODE0:-192.168.0.1}"
TB_IP_NODE1="${TB_IP_NODE1:-192.168.0.2}"
TB_MASK="${TB_MASK:-255.255.255.252}"

cat <<EOF
Run these locally with sudo to prepare the dedicated Thunderbolt link for JACCL.

On ${NODE0}:
  sudo ifconfig bridge0 down
  sudo ifconfig ${TB_IFACE_NODE0} inet ${TB_IP_NODE0} netmask ${TB_MASK}
  sudo route change ${TB_IP_NODE1} -interface ${TB_IFACE_NODE0}

On ${NODE1}:
  sudo ifconfig bridge0 down
  sudo ifconfig ${TB_IFACE_NODE1} inet ${TB_IP_NODE1} netmask ${TB_MASK}
  sudo route change ${TB_IP_NODE0} -interface ${TB_IFACE_NODE1}
EOF
