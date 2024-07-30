#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pygame as pg
import pygame.transform
from pygame.sprite import Sprite
from pygame.transform import scale
from pygame import Surface
import numpy as np
from math import copysign


ANIMATION_SPEED = 2  # 1 is normal speed
ALPHA_INACTIVE = 0.5
ALPHA_ACTIVE = 1.0
COLORS = {
    'text': (255, 255, 255)
}
# left, right, up, down
ACTIONS = ['l', 'r', 'u', 'd']


class State(Sprite):
    def __init__(
            self,
            info: tuple[str, float, float, float, tuple[int, int, int]],
            image: Surface,
            position: tuple[int, int],
            bsize: int,
            *groups,
            additional_info: list = None,
    ):
        """
            info: one float number is for relative size
             and the other for the shift relative to the center
        """
        super().__init__(*groups)
        # rescale to fit size
        original_size = image.get_rect().size
        ratio = original_size[0] / original_size[1]
        if ratio < 1:
            size = (int(bsize * ratio), bsize)
        else:
            size = (bsize, int(bsize/ratio))

        self.image = scale(image, size)
        self.rect = self.image.get_rect()
        self.rect.center = position

        self.decay = (1 / 100) * ANIMATION_SPEED

        text_size = int(bsize * info[1])
        text_vpos = int(size[1]//2 + bsize * info[2])
        text_hpos = int(size[0]//2 + bsize * info[3])

        text = pg.font.SysFont(
            'arial', text_size
        ).render(info[0], True, info[-1])
        text_rect = text.get_rect()
        text_rect.center = (text_hpos, text_vpos)

        self.image.blit(text, text_rect)
        if additional_info is not None:
            for txt in additional_info:
                text_size = int(bsize * txt[1])
                text_vpos = int(size[1] // 2 + bsize * txt[2])
                text_hpos = int(size[0] // 2 + bsize * txt[3])
                text = pg.font.SysFont(
                    'arial', text_size
                ).render(txt[0], True, txt[-1])
                text_rect = text.get_rect()
                text_rect.center = (text_hpos, text_vpos)
                self.image.blit(text, text_rect)

        self.active = False
        self.destroyed = False
        self.alpha = 0
        self.target_alpha = ALPHA_INACTIVE
        self.image.set_alpha(int(round(255 * self.alpha)))

        self.to_scroll = 0
        self.scroll_frames = int(round(60 / ANIMATION_SPEED))
        self.scroll_dx = 0

    def update(self, *args, **kwargs):
        # scrolling
        if self.to_scroll > 0:
            if self.to_scroll < abs(self.scroll_dx):
                scroll_dx = copysign(self.to_scroll, self.scroll_dx)
                self.to_scroll = 0
            else:
                scroll_dx = self.scroll_dx
                self.to_scroll -= abs(self.scroll_dx)

            self.rect.move_ip(-scroll_dx, 0)

        # activate/deactivate animation
        self.update_alpha()

        if self.destroyed and (self.alpha == 0):
            self.kill()

    def update_alpha(self):
        delta = self.target_alpha - self.alpha
        self.alpha += np.sign(delta) * min(abs(delta), self.decay)
        self.alpha = np.clip(self.alpha, 0.0, 1.0)
        self.image.set_alpha(int(round(255 * self.alpha)))

    def activate(self):
        self.target_alpha = ALPHA_ACTIVE
        self.active = True

    def deactivate(self):
        self.target_alpha = ALPHA_INACTIVE
        self.active = False

    def destroy(self):
        self.target_alpha = 0
        self.destroyed = True


class EventHandler:
    def __init__(
            self,
            canvas_size: tuple[int, int],
            step_size: tuple[int, int],
            sprites: tuple
    ):
        self.sprites = sprites
        self.horizontal_step, self.vertical_step = step_size
        self.canvas = Surface(canvas_size)
        self.canvas.fill((255, 255, 255))
        self.bgd = Surface(canvas_size)
        self.bgd.fill((255, 255, 255))

        self.center = (canvas_size[0]//2, canvas_size[1]//2)

        self.current_step = 0
        self.current_step_view = 0
        self.step_groups = list()
        self.main_group = pg.sprite.Group()

        self.scroll_frames = int(round(60 / ANIMATION_SPEED))

    def handle(self, event):
        print(event)
        event_type = event[0]
        if event_type == 'new_obs':
            pos, obs_state, action = event[1:]

            self.set_position(pos)

            self.step_groups.append(
                {
                    'predicted': [],
                    'clones': []
                }
            )

            state_obj = State(
                (str(obs_state), 0.4, -0.1, 0, COLORS['text']),
                self.sprites[0],
                self.center,
                min(self.horizontal_step, self.vertical_step),
                self.main_group,
                additional_info=[(ACTIONS[action], 0.2, 0.3, 0, COLORS['text'])]
            )
            state_obj.activate()
        elif event_type == 'set_state':
            self.set_position(self.current_step)
            state = event[1]
            self.set_state(state, self.current_step)
        elif event_type == 'move':
            pos = event[1]
            self.set_position(pos)
        elif event_type == 'set_prev_state':
            self.set_position(self.current_step)
            state = event[1]
            self.set_state(state, self.current_step - 1)
            self.clear_prediction(self.current_step)
        elif event_type == 'predict_forward':
            self.set_position(self.current_step)
            correct, wrong_perm, wrong_temp = event[1:]
            self.clear_prediction(self.current_step)

            for clone, obs_state, weight in correct:
                self.set_prediction(
                    clone, obs_state, weight, self.current_step, True, False
                )

            for clone, obs_state, weight in wrong_perm:
                self.set_prediction(
                    clone, obs_state, weight, self.current_step, False, False
                )

            for clone, obs_state, weight in wrong_temp:
                self.set_prediction(
                    clone, obs_state, weight, self.current_step, False, True
                )
        elif event_type == 'predict_backward':
            self.set_position(self.current_step)
            correct, wrong = event[1:]
            self.clear_prediction(self.current_step-1)

            for clone, obs_state, weight in correct:
                self.set_prediction(
                    clone, obs_state, weight, self.current_step-1,
                    True, False, False
                )

            for clone, obs_state, weight in wrong:
                self.set_prediction(
                    clone, obs_state, weight, self.current_step-1,
                    False, False, False
                )
        elif event_type == 'reset':
            for sp in self.main_group:
                sp.kill()
            self.step_groups.clear()
            self.current_step = 0

    def clear_prediction(self, step):
        group = self.step_groups[step]['predicted']
        if len(group) > 0:
            for x in group:
                x.destroy()
            group.clear()

    def set_state(self, state, pos):
        group = self.step_groups[pos]['clones']
        if len(group) > 0:
            group[-1].deactivate()
        shift_y = int(0.8*self.vertical_step) + len(group) * int(0.5 * self.vertical_step)
        shift_x = (self.current_step - pos) * self.horizontal_step
        state_obj = State(
            (str(state), 0.5, 0, 0, COLORS['text']),
            self.sprites[1],
            (self.center[0] - shift_x, self.center[1] - shift_y),
            min(self.horizontal_step, int(0.5 * self.vertical_step)),
            self.main_group
        )
        group.append(state_obj)
        state_obj.activate()

    def set_prediction(self, clone, obs_state, weight, pos, correct, temp, forward=True):
        sprite = self.sprites[2 + int(temp)]
        h_shifts = (0.05, -0.4)
        if not forward:
            sprite = pygame.transform.flip(sprite, True, False)
            h_shifts = (-h_shifts[0], -h_shifts[1])
        group = self.step_groups[pos]['predicted']
        shift_y = int(0.8*self.vertical_step) + len(group) * int(0.5 * self.vertical_step)
        shift_x = (self.current_step - pos) * self.horizontal_step
        state_obj = State(
            (str(clone), 0.4, -0.1, h_shifts[0], COLORS['text']),
            sprite,
            (self.center[0] - shift_x, self.center[1] + shift_y),
            min(self.horizontal_step, int(0.5 * self.vertical_step)),
            self.main_group,
            additional_info=[
                (str(obs_state), 0.2, 0.26, h_shifts[0], COLORS['text']),
                (str(weight), 0.2, -0.25, h_shifts[1], (0, 0, 0))
            ]
        )
        group.append(state_obj)
        if correct:
            state_obj.activate()

    def set_position(self, pos):
        to_scroll = abs(pos - self.current_step_view) * self.horizontal_step
        scroll_dx = (
                np.sign(pos - self.current_step_view) *
                max(1, int(round(to_scroll / self.scroll_frames)))
        )
        for sprite in self.main_group:
            sprite.to_scroll = to_scroll
            sprite.scroll_dx = scroll_dx
        self.current_step = pos
        self.current_step_view = pos

    def scroll(self, delta_pos):
        to_scroll = abs(delta_pos) * self.horizontal_step
        scroll_dx = (
                np.sign(delta_pos) * to_scroll
        )
        for sprite in self.main_group:
            sprite.to_scroll = to_scroll
            sprite.scroll_dx = scroll_dx
        self.current_step_view += delta_pos

    def update(self):
        self.main_group.clear(self.canvas, self.bgd)
        self.main_group.draw(self.canvas)
        self.main_group.update()
