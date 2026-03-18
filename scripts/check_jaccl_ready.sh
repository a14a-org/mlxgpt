#!/usr/bin/env bash
set -euo pipefail

NODE0="${NODE0:-node-0.local}"
NODE1="${NODE1:-node-1.local}"
RDMA_DEV_NODE0="${RDMA_DEV_NODE0:-rdma_en4}"
RDMA_DEV_NODE1="${RDMA_DEV_NODE1:-rdma_en2}"
TB_IFACE_NODE0="${TB_IFACE_NODE0:-en4}"
TB_IFACE_NODE1="${TB_IFACE_NODE1:-en2}"
TB_IP_NODE0="${TB_IP_NODE0:-192.168.0.1}"
TB_IP_NODE1="${TB_IP_NODE1:-192.168.0.2}"

check_node() {
  local host="$1"
  local rdma_device="$2"
  local iface="$3"
  local expected_ip="$4"

  echo "== ${host} =="
  ssh "$host" "
    set -euo pipefail
    printf 'rdma_ctl: '
    rdma_ctl status
    printf 'rdma_device_present: '
    if ibv_devices | awk 'NR > 2 {print \$1}' | grep -qx '${rdma_device}'; then
      echo yes
    else
      echo no
    fi
    printf 'rdma_port_state: '
    state=\$(ibv_devinfo -d '${rdma_device}' | awk '/state:/ {print \$2; exit}')
    if [ -n \"\$state\" ]; then
      echo \"\$state\"
    else
      echo unknown
    fi
    printf 'iface_ipv4: '
    if ifconfig '${iface}' | grep -q 'inet ${expected_ip} '; then
      echo '${expected_ip}'
    else
      ip=\$(ifconfig '${iface}' | awk '/inet / {print \$2; exit}')
      if [ -n \"\$ip\" ]; then
        echo \"\$ip\"
      else
        echo missing
      fi
    fi
    printf 'iface_status: '
    if ifconfig '${iface}' | grep -q 'status: active'; then
      echo active
    else
      echo inactive
    fi
  "
  echo
}

ready=0

check_node "${NODE0}" "${RDMA_DEV_NODE0}" "${TB_IFACE_NODE0}" "${TB_IP_NODE0}"
check_node "${NODE1}" "${RDMA_DEV_NODE1}" "${TB_IFACE_NODE1}" "${TB_IP_NODE1}"

if ssh "${NODE0}" "ifconfig '${TB_IFACE_NODE0}' | grep -q 'inet ${TB_IP_NODE0} '" \
  && ssh "${NODE1}" "ifconfig '${TB_IFACE_NODE1}' | grep -q 'inet ${TB_IP_NODE1} '" \
  && ssh "${NODE0}" "ibv_devices | awk 'NR > 2 {print \$1}' | grep -qx '${RDMA_DEV_NODE0}'" \
  && ssh "${NODE1}" "ibv_devices | awk 'NR > 2 {print \$1}' | grep -qx '${RDMA_DEV_NODE1}'" \
  && ssh "${NODE0}" "ibv_devinfo -d '${RDMA_DEV_NODE0}' | awk '/state:/ {print \$2; exit}' | grep -qx 'PORT_ACTIVE'" \
  && ssh "${NODE1}" "ibv_devinfo -d '${RDMA_DEV_NODE1}' | awk '/state:/ {print \$2; exit}' | grep -qx 'PORT_ACTIVE'"; then
  ready=1
fi

if [ "$ready" -eq 1 ]; then
  echo "JACCL readiness: READY"
else
  echo "JACCL readiness: NOT READY"
  echo "Confirm the Thunderbolt IP setup and make sure the selected RDMA ports report PORT_ACTIVE."
fi
