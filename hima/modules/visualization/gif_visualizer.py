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


class GifViewer:
    def __init__(self, directory='./'):
        self.directory = directory
        self.frames = []

        self.files = pn.widgets.FileSelector(directory, file_pattern='*.gif')
        self.time_step = pn.widgets.Player(start=0)
        self.window_size = pn.widgets.IntInput(name='window', value=0, start=0)
        self.res = pn.widgets.IntInput(name='resolution', value=1000, start=50)
        self.frames = []

        self.canvas = pn.bind(
            self.show_frames,
            files=self.files,
            time_step=self.time_step,
            window_size=self.window_size,
            res=self.res
        )
        self.app = pn.Column(
            self.files,
            pn.Row(self.time_step, self.window_size, self.res),
            self.canvas
        )

    def show(self):
        return self.app

    def show_frames(self, files, time_step, window_size, res):
        if len(files) == 0:
            self.frames = []
            return

        max_frames = 0
        self.frames = []
        for file in files:
            gif = imageio.get_reader(file)
            frames = list(gif)

            self.frames.append(frames)

            if len(frames) > max_frames:
                max_frames = len(frames)

        self.time_step.end = max_frames - 1

        imgs = []
        for i, frames in enumerate(self.frames):
            for j in range(time_step - window_size, time_step + window_size + 1):
                if 0 < j < len(frames):
                    frame = self.preprocess(frames[j])
                else:
                    frame = np.zeros_like(self.preprocess(frames[0]))

                aspect = frame.shape[0] / frame.shape[1]
                img = self.plot_heatmap(
                    frame,
                    res,
                    int(aspect * res),
                    files[i].split('/')[-1]
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
        )

    @staticmethod
    def preprocess(im, channel=0):
        im = im.astype('float')
        if len(im.shape) > 2:
            im = im[:, :, channel]
        im /= im.max()
        return im


if __name__ == '__main__':
    root_dir = os.environ.get('GIF_VIS_ROOT_DIR', '~/')
    gv = GifViewer(directory=root_dir)
    gv.app.show()
