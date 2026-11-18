
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone

from src.event_manager import SecurityEvent, record_event


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)



POWERSHELL_SCRIPT = r"""
$diskMetadata = @{}

Get-CimInstance -ClassName Win32_DiskDrive |
ForEach-Object {
    $disk = Get-Disk -Number $_.Index -ErrorAction SilentlyContinue

    if ($disk) {
        $diskMetadata[$_.PNPDeviceID] = @{
            BusType = [string]$disk.BusType
            IsBoot = [bool]$disk.IsBoot
            IsSystem = [bool]$disk.IsSystem
        }
    }
}

$devices = @(
    Get-PnpDevice -PresentOnly |
    Where-Object {
        $_.Class -in @("DiskDrive", "WPD") -or
        (
            $_.Class -eq "USB" -and
            $_.FriendlyName -notmatch "Host Controller|Root Hub"
        )
    } |
    ForEach-Object {
        $metadata = $diskMetadata[$_.InstanceId]

        [PSCustomObject]@{
            Class = $_.Class
            FriendlyName = $_.FriendlyName
            InstanceId = $_.InstanceId
            BusType = if ($metadata) {
                $metadata.BusType
            } else {
                $null
            }
            IsBoot = if ($metadata) {
                $metadata.IsBoot
            } else {
                $null
            }
            IsSystem = if ($metadata) {
                $metadata.IsSystem
            } else {
                $null
            }
        }
    }
)

ConvertTo-Json -InputObject $devices -Compress
"""



def get_connected_devices():
    """Read currently present USB-related devices from Windows."""

    if platform.system() != "Windows":
        raise RuntimeError(
            "This monitor currently supports Windows only."
        )

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            POWERSHELL_SCRIPT,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )

    output = result.stdout.strip()

    if not output:
        return {}

    devices = json.loads(output)

    if isinstance(devices, dict):
        devices = [devices]

    return {
        device["InstanceId"]: device
        for device in devices
        if device.get("InstanceId")
    }


def record_usb_event(action, device):
    """Save a USB-related device event to the security-events database."""

    device_class = device.get("Class", "Unknown")
    device_name = device.get("FriendlyName", "Unknown device")

    event = SecurityEvent(
        user_id="SYSTEM",
        event_type=f"usb_device_{action.lower()}",
        description=(
            f"USB-related device {action.lower()}: "
            f"{device_name} (class: {device_class})"
        ),
        risk_level="low",
    )

    try:
        saved = record_event(event)

        logging.info(
            "Security event saved | %s | %s",
            saved["event_id"],
            event.event_type,
        )

    except Exception:
        logging.exception(
            "Failed to save USB security event"
        )


def monitor_usb(poll_interval=3):
    """Log newly connected and removed USB-related devices."""

    logging.info(
        "USB and portable-device monitoring started."
    )

    try:
        previous = get_connected_devices()

        logging.info(
            "Baseline captured: %d USB-related devices already connected.",
            len(previous),
        )

        while True:
            current = get_connected_devices()

            # Detect newly connected devices.
            for device_id in current.keys() - previous.keys():
                device = current[device_id]

                logging.warning(
                    "DEVICE CONNECTED | %s | %s | %s",
                    device.get("Class"),
                    device.get("FriendlyName"),
                    datetime.now(timezone.utc).isoformat(),
                )

                record_usb_event("CONNECTED", device)

            # Detect disconnected devices.
            for device_id in previous.keys() - current.keys():
                device = previous[device_id]

                logging.warning(
                    "DEVICE DISCONNECTED | %s | %s | %s",
                    device.get("Class"),
                    device.get("FriendlyName"),
                    datetime.now(timezone.utc).isoformat(),
                )

                record_usb_event("DISCONNECTED", device)

            previous = current
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logging.info("Device monitoring stopped.")

    except (
        subprocess.SubprocessError,
        json.JSONDecodeError,
        OSError,
    ) as error:
        logging.error(
            "Device monitoring error: %s",
            error,
        )


if __name__ == "__main__":
    monitor_usb()
