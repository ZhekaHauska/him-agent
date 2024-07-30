#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
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

from hima.experiments.cognitive_maps.vis.entities import LearningHistory, TransitionGraph

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

        # initialize the pygame module
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Toy DHTM")
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.window_size)

        # controls
        self.key_actions = (
            (pygame.K_SPACE, "step"),
            (pygame.K_LEFT, "scroll_left"),
            (pygame.K_RIGHT, "scroll_right")
        )
        # define a variable to control the main loop
        self.running = True
        self.events = list()

        sprites = (
            pygame.image.load('assets/obs_state.png').convert_alpha(),
            pygame.image.load('assets/hidden_state.png').convert_alpha(),
            pygame.image.load('assets/predicted_state_perm.png').convert_alpha(),
            pygame.image.load('assets/predicted_state_temp.png').convert_alpha(),
            pygame.image.load('assets/aliased_state.png').convert_alpha(),
        )
        self.history = LearningHistory(
            (self.window_size[0], self.window_size[1]//2),
            (100, 100),
            sprites
        )
        self.graph = TransitionGraph(
            (self.window_size[0], self.window_size[1] // 2),
            [sprites[-1]],
            50,
            1,
            0.99,
            2.0,
            10,
            0.01
        )

    def run(self):
        while self.running:
            self.clock.tick(60)
            # event handling, gets all event from the event queue
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # change the value to False, to exit the main loop
                    self.running = False
                    self.close()
                if event.type == pygame.KEYDOWN:
                    for k, a in self.key_actions:
                        if event.key == k:
                            action = a

            if self.connection is not None:
                if action == 'step':
                    if len(self.events) == 0:
                        self._send_string('step')
                        data = self._get_json_dict()

                        if data['type'] == 'events':
                            events = data['events']
                            self.events.extend(events)
                        elif data['type'] == 'close':
                            self.connection = None

                    if len(self.events) > 0:
                        self._handle_event(self.events.pop(0))
                elif action == 'scroll_left':
                    self.history.scroll(-1)
                elif action == 'scroll_right':
                    self.history.scroll(+1)

            self.update()
            pygame.display.flip()

            if self.connection is None:
                self.connect()

        pygame.quit()

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

    def update(self):
        self.history.update()
        self.graph.update()
        self.screen.blit(self.graph.canvas, (0, 0))
        self.screen.blit(self.history.canvas, (0, self.graph.canvas.get_size()[1]))

    def _handle_event(self, event):
        self.history.handle(event)
        self.graph.handle(event)


if __name__ == '__main__':
    ToyDHTMVis().run()
