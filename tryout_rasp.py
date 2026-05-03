import socket
import subprocess

def check_raspberry_pi_connection(hostname="raspberrypi", timeout=2):
    """Check if Raspberry Pi is reachable on the network"""
    try:
        # Try to ping the device
        result = subprocess.run(
            ["ping", "-c", "1", hostname],
            timeout=timeout,
            capture_output=True
        )
        if result.returncode == 0:
            print(f"✓ {hostname} is connected!")
            return True
        else:
            print(f"✗ {hostname} is not reachable")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {hostname} did not respond")
        return False

    def list_connected_devices(subnet="192.168.1"):
        """Scan subnet and list connected devices"""
        connected = []
        for i in range(1, 255):
            ip = f"{subnet}.{i}"
            result = subprocess.run(
                ["ping", "-c", "1", ip],
                timeout=1,
                capture_output=True
            )
            if result.returncode == 0:
                connected.append(ip)
        return connected
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    check_raspberry_pi_connection()