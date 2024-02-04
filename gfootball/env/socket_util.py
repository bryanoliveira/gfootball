import abc
import atexit
import json
import lz4.frame
import numpy as np
import socket

MAX_BUFFER_SIZE = 16 * 1024**2
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
    def sendall(self, data: str):
        # encode data
        data = data.encode("utf-8")
        # compress it
        data = lz4.frame.compress(data)
        # send length
        self.client_socket.sendall(len(data).to_bytes(4, byteorder="big"))
        # print(f"sent {len(data)} bytes")
        # send data
        self.client_socket.sendall(data)

    def recvall(self) -> str:
        # receive length
        length = int.from_bytes(self.client_socket.recv(4), byteorder="big")
        # receive data
        data = b""
        while len(data) < length:
            packet = self.client_socket.recv(
                min(length - len(data), MAX_BUFFER_SIZE)
            )
            if not packet:
                break  # Connection closed
            data += packet

        # decompress it
        data = lz4.frame.decompress(data)
        # decode it
        data = data.decode("utf-8")
        return data


class Server(Messenger):
    def __init__(self, port=6000, verbose=False):
        global server_socket, client_socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
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
        self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
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
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                break
            except ConnectionRefusedError:
                continue

        print("CONNECTED TO SERVER", self.client_socket.getpeername())
