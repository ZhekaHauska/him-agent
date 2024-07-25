#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
import socket
import time
import json
import atexit
import pygame

HOST = "127.0.0.1"
PORT = 5555
TIMEOUT = 600


class ToyDHTMVis:
    def __init__(
            self,
            window_size=(800, 600),
            host=HOST,
            port=PORT
    ):
        self.host = host
        self.port = port

        self.window_size = window_size

        atexit.register(self.close)

        # create server
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection = None
        while True:
            try:
                self.server.bind((self.host, self.port))
                break

            except OSError:
                self.port += 1
                if self.port > 49151:
                    self.port = 1024

        self.run()

    def run(self):
        # initialize the pygame module
        pygame.init()
        pygame.font.init()
        font = pygame.font.SysFont('arial', 14)
        pygame.display.set_caption("Toy DHTM")

        # create a surface on screen that has the size of 240 x 180
        screen = pygame.display.set_mode(self.window_size)
        # controls
        key_actions = (
            (pygame.K_SPACE, "step"),
        )
        # define a variable to control the main loop
        running = True
        # kind of camera position
        pos = 0
        events = dict()
        info = 'No info'
        # main loop
        while running:
            # event handling, gets all event from the event queue
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # change the value to False, to exit the main loop
                    running = False
                    self.close()
                if event.type == pygame.KEYDOWN:
                    for k, a in key_actions:
                        if event.key == k:
                            action = a

            if self.connection is None:
                info = "Waiting for new connection."
            else:
                if action == 'step':
                    self._send_string('step')
                    data = self._get_json_dict()

                    if data['type'] == 'events':
                        events = data
                        info = str(events)
                    elif data['type'] == 'close':
                        info = "Connection closed."
                        self.connection = None

            # update text info
            txt = font.render(
                info,
                False,
                (255, 255, 255)
            )
            screen.fill((0, 0, 0))
            screen.blit(txt, (self.window_size[0] // 3, 0))
            pygame.display.flip()

            if self.connection is None:
                self.connect()
                info = "Connected"

    def _send_as_json(self, dictionary):
        message_json = json.dumps(dictionary)
        self._send_string(message_json)

    def _get_json_dict(self):
        data = self._get_data()
        return json.loads(data)

    def _send_string(self, string):
        message = len(string).to_bytes(4, "little") + bytes(string.encode())
        self.connection.sendall(message)

    def _get_data(self):
        try:
            data = None
            while not data:
                data = self.connection.recv(4)
                time.sleep(0.000001)

            length = int.from_bytes(data, "little")
            string = ""
            while (
                    len(string) != length
            ):  # TODO: refactor as string concatenation could be slow
                string += self.connection.recv(length).decode()

            return string
        except socket.timeout as e:
            print("toy_dhtm timed out", e)

        return None

    def close(self):
        self._send_string('close')
        print("close message sent")
        time.sleep(1.0)
        self.connection.close()
        try:
            atexit.unregister(self.close)
        except Exception as e:
            print("exception unregistering close method", e)

    def connect(self):
        self.server.listen(1)
        self.connection, addr = self.server.accept()
        print(
            f'connected {addr}'
        )

        data = self._get_json_dict()
        if data['type'] == 'hello':
            self._send_string('toy_dhtm')
        else:
            self.close()


if __name__ == '__main__':
    vis_sever = ToyDHTMVis()
