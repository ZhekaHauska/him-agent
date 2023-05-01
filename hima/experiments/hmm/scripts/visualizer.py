#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pandas as pd
import panel as pn
import imageio
import holoviews
import hvplot.pandas
import numpy as np


class GifViewer:
    def __init__(self, directory='./', res=1000):
        self.directory = directory
        self.res = res
        self.frames = []

        self.files = pn.widgets.FileSelector(directory)
        self.time_step = pn.widgets.Player(start=0)
        self.frames = []

        self.canvas = pn.bind(self.show_frames, files=self.files, time_step=self.time_step)
        self.app = pn.Column(self.files, self.time_step, self.canvas)

    def show(self):
        return self.app

    def show_frames(self, files, time_step):
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

        img = []
        for frames in self.frames:
            if time_step < len(frames):
                frame = self.preprocess(frames[time_step])
                aspect = frame.shape[0] / frame.shape[1]
                x, y = np.indices(frame.shape)
                df = pd.DataFrame({'y': x.flatten()[::-1], 'x': y.flatten(), 'p': frame.flatten()})
                img.append(
                    df.hvplot.heatmap(
                        x='x',
                        y='y',
                        C='p',
                        cmap='Blues',
                        width=self.res,
                        height=int(aspect * self.res)
                    )
                )

        if len(img) > 0:
            return holoviews.Layout(img).cols(1)
        else:
            return

    @staticmethod
    def preprocess(im):
        im = im.astype('float')
        if len(im.shape) > 2:
            im = im.mean(axis=-1)
        im /= im.max()
        return np.power(im, 2)


if __name__ == '__main__':
    gv = GifViewer(directory='~/Загрузки', res=1000)
    gv.app.show()
