#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import os

import pandas as pd
import panel as pn
import imageio
import holoviews
import hvplot.pandas
import numpy as np


class SequenceViewer:
    ordered_files: list

    def __init__(self, directory='./'):
        self.directory = directory
        self.frames = []

        self.files = pn.widgets.FileSelector(directory)
        self.time_step = pn.widgets.Player(start=0)
        self.window_size = pn.widgets.IntInput(name='window', value=0, start=0)
        self.aspect = pn.widgets.FloatInput(name='aspect', value=1, start=0)
        self.res = pn.widgets.IntInput(name='resolution', value=1000, start=100)
        self.ordered_files = []
        self.frames = []

        self.canvas = pn.bind(
            self.show_frames,
            files=self.files,
            time_step=self.time_step,
            window_size=self.window_size,
            res=self.res,
            aspect=self.aspect
        )
        self.app = pn.Column(
            self.files,
            pn.Row(self.time_step, self.window_size, self.res, self.aspect),
            self.canvas
        )

    def show(self):
        return self.app

    def show_frames(self, files: list, time_step, window_size, res, aspect):
        self.update_ordered_files(files)
        if len(self.ordered_files) == 0:
            self.frames = []
            return

        max_frames = 0
        self.frames = []
        for file in self.ordered_files:
            ext = file.split('.')[-1]
            if ext == 'gif':
                gif = imageio.get_reader(file)
                frames = list()
                for frame in gif:
                    frames.append(self.preprocess(frame))
            elif ext == 'npy':
                frames = np.load(file)
            else:
                raise NotImplementedError

            self.frames.append(frames)

            if len(frames) > max_frames:
                max_frames = len(frames)

        self.time_step.end = max_frames - 1

        imgs = []
        for i, frames in enumerate(self.frames):
            for j in range(time_step - window_size, time_step + window_size + 1):
                if 0 <= j < len(frames):
                    frame = frames[j]
                else:
                    frame = np.zeros_like(frames[0])

                img = self.plot_heatmap(
                    frame,
                    res,
                    max(int(aspect * (frame.shape[0] / frame.shape[1]) * res), 100),
                    self.ordered_files[i].split('/')[-1]
                )
                imgs.append(img)

        if len(imgs) > 0:
            return holoviews.Layout(imgs).cols(window_size*2 + 1)
        else:
            return

    @staticmethod
    def plot_heatmap(data, width, height, title):
        x, y = np.indices(data.shape)
        df = pd.DataFrame({'y': x.flatten()[::-1], 'x': y.flatten(), 'p': data.flatten()})
        return df.hvplot.heatmap(
            x='x',
            y='y',
            C='p',
            cmap='Blues',
            width=width,
            height=height,
            title=title
        ).opts(axiswise=True)

    @staticmethod
    def preprocess(im, channel=0):
        im = im.astype('float')
        if len(im.shape) > 2:
            im = im[:, :, channel]
        im /= 255
        return im

    def update_ordered_files(self, files):
        set_ordered_files = set(self.ordered_files)
        set_files = set(files)

        if set_ordered_files == set_files:
            return

        if len(set_files) == 0:
            self.ordered_files.clear()
            return

        for f in files:
            if not (f in set_ordered_files):
                self.ordered_files.append(f)

        for f in self.ordered_files:
            if not (f in set_files):
                self.ordered_files.remove(f)


if __name__ == '__main__':
    root_dir = os.environ.get('GIF_VIS_ROOT_DIR', '~/')
    gv = SequenceViewer(directory=root_dir)
    gv.app.show()
