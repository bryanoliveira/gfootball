import abc
import atexit
import json
import numpy as np
import socket

DEFAULT_BUFFER_SIZE = 1024
server_socket = None
client_socket = None


def disconnect():
    global server_socket, client_socket
    if server_socket is not None:
        server_socket.close()
    if client_socket is not None:
        client_socket.close()
    print("CLOSED SOCKETS")


atexit.register(disconnect)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Messenger(abc.ABC):
    def sendall(self, data: str, prefix="", wait_for_ack=True):
        assert len(prefix) <= 1
        data = prefix + data
        self.client_socket.sendall(data.encode("utf-8"))
        if wait_for_ack:
            ack = self.client_socket.recv(DEFAULT_BUFFER_SIZE)
            if ack != b"ACK":
                raise Exception("ACK NOT RECEIVED")

    def recvall(self, prefix="", send_ack=True) -> str:
        assert len(prefix) <= 1
        received_prefix = self.client_socket.recv(len(prefix.encode("utf-8"))).decode("utf-8")
        assert (
            received_prefix == prefix
        ), f"Expected prefix {prefix}, got {received_prefix}"

        data = b""
        while True:
            part = self.client_socket.recv(DEFAULT_BUFFER_SIZE)
            data += part
            if len(part) < DEFAULT_BUFFER_SIZE:
                # No more data, or the remaining data is less than buffer size
                break
        if send_ack:
            self.client_socket.sendall(b"ACK")
        return data.decode("utf-8")


class Server(Messenger):
    def __init__(self, port=6000):
        global server_socket, client_socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.server_socket.bind(("0.0.0.0", port))
            server_socket = self.server_socket
        except OSError as e:
            print(e)
            print("Could not bind to port ", port)
            exit(1)
        self.server_socket.listen(1)
        print("AWAITING CONNECTION ON PORT", port, "...")
        self.client_socket, _ = self.server_socket.accept()
        client_socket = self.client_socket
        print("CONNECTED TO CLIENT", self.client_socket.getpeername())


class Client(Messenger):
    def __init__(self, ip="127.0.0.1", port=6000):
        global client_socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("CONNECTING TO SERVER", ip, port)
        while True:
            try:
                self.client_socket.connect((ip, port))
                break
            except ConnectionRefusedError:
                continue

        print("CONNECTED TO SERVER", self.client_socket.getpeername())
