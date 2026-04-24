"""
UDP communication bridge for simulator state and control.

Handles low-level UDP send/receive operations for:
- Receiving simulation state from esmini
- Sending control commands to esmini
"""

import socket


class UdpBridge:
    """Manages UDP sockets for bidirectional communication with the simulator."""
    
    def __init__(self, listen_ip: str, listen_port: int, send_ip: str, 
                 send_port: int, timeout_s: float):
        """
        Initialize UDP bridge with separate sockets for receive and send.
        
        Args:
            listen_ip: IP address to bind for receiving state
            listen_port: Port to bind for receiving state
            send_ip: IP address to send control commands to
            send_port: Port to send control commands to
            timeout_s: Receive timeout in seconds
        """
        # Receive socket (state from esmini -> python)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind((listen_ip, listen_port))
        self.rx.settimeout(timeout_s)

        # Send socket (control python -> esmini)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = (send_ip, send_port)

    def recv_packet(self) -> bytes:
        """
        Receive a UDP packet containing simulation state.
        
        Returns:
            Raw packet bytes
            
        Raises:
            socket.timeout: If no packet is received within the timeout period
        """
        packet, _addr = self.rx.recvfrom(65535)
        return packet

    def send_packet(self, payload: bytes):
        """
        Send a UDP packet containing control commands.
        
        Args:
            payload: Control command data as bytes
        """
        self.tx.sendto(payload, self.send_addr)

    def close(self):
        """Close both UDP sockets."""
        self.rx.close()
        self.tx.close()
