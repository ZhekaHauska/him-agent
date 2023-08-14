#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os


class ScalarMetrics:
    def __init__(self, metrics, logger):
        self.logger = logger
        self.metrics = {metric: [] for metric in metrics.keys()}
        self.agg_func = {metric: func for metric, func in metrics.items()}

    def update(self, metric_values):
        for key, value in metric_values.items():
            self.metrics[key].append(value)

    def log(self, step):
        average_metrics = {
            key: self.agg_func[key](values)
            for key, values in self.metrics.items()
            if len(values) > 0
        }

        self.logger.log(
            average_metrics,
            step=step
        )

        self.metrics = {metric: [] for metric in self.metrics.keys()}


class HeatmapMetrics:
    def __init__(self, metrics, logger):
        self.logger = logger
        self.metrics = {metric: [] for metric in metrics.keys()}
        self.agg_func = {metric: func for metric, func in metrics.items()}

    def update(self, metric_values):
        for key, value in metric_values.items():
            self.metrics[key].append(value)

    def log(self, step):
        average_metrics = {
            key: self.agg_func[key](values, axis=0)
            for key, values in self.metrics.items()
            if len(values) > 0
        }

        log_dict = {}
        for key, value in average_metrics.items():
            plt.figure()
            log_dict[key] = wandb.Image(
                sns.heatmap(value)
            )
        self.logger.log(
            log_dict,
            step=step
        )

        plt.close('all')

        self.metrics = {metric: [] for metric in self.metrics.keys()}


class ImageMetrics:
    def __init__(self, metrics, logger, log_fps, log_dir='/tmp'):
        self.metrics = {metric: [] for metric in metrics}
        self.logger = logger
        self.log_fps = log_fps
        self.log_dir = log_dir

    def update(self, metric_values):
        for key, value in metric_values.items():
            self.metrics[key].append(value)

    def log(self, step):
        for metric, values in self.metrics.items():
            if len(values) > 1:
                gif_path = os.path.join(
                    self.log_dir,
                    f'{self.logger.name}_{metric.split("/")[-1]}_{step}.gif'
                )
                with imageio.get_writer(gif_path, mode='I', duration=1000/self.log_fps, loop=0) as writer:
                    for image in values:
                        writer.append_data(image)

                self.logger.log(
                    {
                        metric: wandb.Video(gif_path)
                    },
                    step=step
                )
            elif len(values) > 0:
                self.logger.log(
                    {
                        metric: wandb.Image(values[0])
                    },
                    step=step
                )

        self.metrics = {metric: [] for metric in self.metrics.keys()}
