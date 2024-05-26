import io

import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py
import os
import itertools

from math import ceil
from matplotlib.ticker import AutoMinorLocator
import heartpy as hp

import ECGMetaData
import scipy.signal as sgn


class ECGGenerator(object):
    def __init__(self, ecg_data, ecg_meta_data: ECGMetaData.ECGMetaData = None,
                 to_preprocess=False): #TODO chnage to_preprocess to True
        self.ecg_data = ecg_data
        # if to_normalize:
        #    self.__normalize_ecg_data__()
        self.ecg_meta_data = ecg_meta_data
        if self.ecg_meta_data is None:
            self.ecg_meta_data = ECGMetaData.ECGMetaData()
        if to_preprocess:
            self.__preprocess_ecg_data__()

    def __init_plot__(self):
        self.fig, self.ax = plt.subplots(
            figsize=self.ecg_meta_data.get_fig_size())
        self.fig.subplots_adjust(
            hspace=0,
            wspace=0,
            left=0,  # the left side
            right=1,  # the right side
            bottom=0,  # the bottom of the subplots of the figure
            top=1
        )
        self.fig.suptitle(self.ecg_meta_data.get_title())

    def __init_grid__(self):
        if self.ecg_meta_data.get_show_grid():
            self.ax.set_xticks(np.arange(self.ecg_meta_data.x_min, self.ecg_meta_data.x_max + 0.36, 0.2))
            self.ax.set_yticks(np.arange(self.ecg_meta_data.y_min, self.ecg_meta_data.y_max, 0.5))
            self.ax.minorticks_on()
            self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            self.ax.grid(which='major', linestyle='-', linewidth=0.5 * self.ecg_meta_data.updated_display_factor,
                         color=self.ecg_meta_data.color_major)
            self.ax.grid(which='minor', linestyle='-', linewidth=0.5 * self.ecg_meta_data.updated_display_factor,
                         color=self.ecg_meta_data.color_minor)
            self.ax.spines['left'].set_color(self.ecg_meta_data.color_major)
            self.ax.spines['right'].set_color(self.ecg_meta_data.color_major)
            self.ax.spines['bottom'].set_color(self.ecg_meta_data.color_major)
            self.ax.spines['top'].set_color(self.ecg_meta_data.color_major)
            self.ax.spines['left'].set_linewidth(0.5 * self.ecg_meta_data.updated_display_factor)
            self.ax.xaxis.set_ticks_position('none')
            self.ax.yaxis.set_ticks_position('none')
        self.ax.set_ylim(self.ecg_meta_data.y_min, self.ecg_meta_data.y_max)
        self.ax.set_xlim(self.ecg_meta_data.x_min, self.ecg_meta_data.x_max + 0.36)

    def __plot_short_leads__(self):
        reordered_leads = [self.ecg_meta_data.get_lead_order()[i] for i in range(self.ecg_meta_data.get_num_leads())]
        for index, t_lead in enumerate(reordered_leads):
            c = index // self.ecg_meta_data.short_lead_rows
            i = index % self.ecg_meta_data.short_lead_rows
            if c * self.ecg_meta_data.short_lead_rows + i < self.ecg_meta_data.get_num_leads():
                y_offset = -(self.ecg_meta_data.get_row_height() / 2) * ceil(i)
                # self.ecg_data[t_lead][0]
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25
                x_offset = 0
                if c > 0:
                    x_offset = self.ecg_meta_data.secs_to_display_per_column * c
                    if self.ecg_meta_data.to_show_separate_line():
                        self.ax.plot([x_offset, x_offset],
                                     [self.ecg_data[t_lead][0] + y_offset - 0.3,
                                      self.ecg_data[t_lead][0] + y_offset + 0.3],
                                     linewidth=self.ecg_meta_data.line_width * self.ecg_meta_data.updated_display_factor,
                                     color=self.ecg_meta_data.color_line)
                if self.ecg_meta_data.to_show_lead_name:
                    self.ax.text(
                        x_offset + self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length
                        , y_offset - 0.5,
                        self.ecg_meta_data.get_lead_index()[t_lead],
                        fontsize=9 * self.ecg_meta_data.updated_display_factor)
                x_start, x_end = x_offset, min(x_offset + self.ecg_meta_data.secs_to_display_per_column,
                                               self.ecg_meta_data.x_max)
                if c + 1 == self.ecg_meta_data.get_columns() and self.ecg_meta_data.get_columns() == 1:
                    x_end = x_end + self.ecg_meta_data.step

                y_start, y_end = int(x_start * self.ecg_meta_data.get_sample_rate()), max(
                    int(x_end * self.ecg_meta_data.get_sample_rate()),
                    int((x_offset + self.ecg_meta_data.secs_to_display_per_column) * \
                        self.ecg_meta_data.get_sample_rate()))
                to_debug = False #TODO i Chenge it to True
                color = self.ecg_meta_data.color_line
                if to_debug:
                    print(
                        f'lead_name: {self.ecg_meta_data.get_lead_index()[t_lead]}, y_offset: {y_offset}, x_offset: {x_offset}')
                    print(f'x_start: {x_start}, x_end: {x_end}, y_start: {y_start}, y_end: {y_end}')
                    print(f'first y value in y_start and t_lead: {self.ecg_data[t_lead][y_start]}')
                    if y_end == 5000: #TODO: we changed it
                        print(f'last y value in y_end and t_lead: {self.ecg_data[t_lead][y_end - 1]}')
                    else:
                        print(f'last y value in y_end and t_lead: {self.ecg_data[t_lead][y_end]}')
                    print(f'plotted first y value: {self.ecg_data[t_lead][y_start] + y_offset}')
                    print()
                # calibration box:
                if c == 0:  # only plot the calibration box if the lead is the first lead in the column
                    self.ax.plot([self.ecg_meta_data.line_length_before_calibration_signal,
                                  self.ecg_meta_data.line_length_before_calibration_signal,
                                  self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length,
                                  self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length],
                                 [self.ecg_data[t_lead][0] + y_offset, self.ecg_data[t_lead][0] + y_offset + 1,
                                  self.ecg_data[t_lead][0] + y_offset + 1, self.ecg_data[t_lead][0] + y_offset],
                                 linewidth=self.ecg_meta_data.line_width * self.ecg_meta_data.updated_display_factor,
                                 color=self.ecg_meta_data.color_line)
                    self.ax.plot([0, self.ecg_meta_data.line_length_before_calibration_signal],
                                 [self.ecg_data[t_lead][0] + y_offset, self.ecg_data[t_lead][0] + y_offset],
                                 linewidth=self.ecg_meta_data.line_width,
                                 color=self.ecg_meta_data.color_line)
                    self.ax.plot([
                        self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length
                        ,
                        self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length],
                        [self.ecg_data[t_lead][0] + y_offset, self.ecg_data[t_lead][0] + y_offset],
                        linewidth=self.ecg_meta_data.line_width,
                        color=self.ecg_meta_data.color_line)

                self.ax.plot(np.arange(x_start, x_end, self.ecg_meta_data.step) +
                             self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length,
                             self.ecg_data[t_lead][y_start:y_end] + y_offset,
                             linewidth=self.ecg_meta_data.line_width * self.ecg_meta_data.updated_display_factor,
                             color=color)

                self.ax.tick_params(axis='x', bottom=False, labelbottom=False)
                self.ax.tick_params(axis='y', left=False, labelleft=False)

    def __plot_long_leads__(self):
        if self.ecg_meta_data.to_plot_long_leads():
            for i, idx in enumerate(self.ecg_meta_data.get_long_lead_indexes()):
                y_offset = -(self.ecg_meta_data.get_row_height() / 2) * ceil(
                    (i + self.ecg_meta_data.short_lead_rows) % self.ecg_meta_data.total_rows) - self.ecg_data[idx][0]
                y_start, y_end = int(self.ecg_meta_data.x_min * self.ecg_meta_data.get_sample_rate()), int(
                    self.ecg_meta_data.x_max * self.ecg_meta_data.get_sample_rate())

                if self.ecg_meta_data.get_show_lead_name():
                    self.ax.text(
                        self.ecg_meta_data.x_min + self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length
                        , y_offset - 0.5,
                        self.ecg_meta_data.get_lead_index()[idx],
                        fontsize=9 * self.ecg_meta_data.updated_display_factor)
                # plot calibration box for long leads
                self.ax.plot([self.ecg_meta_data.line_length_before_calibration_signal,
                              self.ecg_meta_data.line_length_before_calibration_signal,
                              self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length,
                              self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length],
                             [self.ecg_data[idx][0] + y_offset, self.ecg_data[idx][0] + y_offset + 1,
                              self.ecg_data[idx][0] + y_offset + 1, self.ecg_data[idx][0] + y_offset],
                             linewidth=self.ecg_meta_data.line_width,
                             color=self.ecg_meta_data.color_line)
                self.ax.plot([0, self.ecg_meta_data.line_length_before_calibration_signal],
                             [self.ecg_data[idx][0] + y_offset, self.ecg_data[idx][0] + y_offset],
                             linewidth=self.ecg_meta_data.line_width,
                             color=self.ecg_meta_data.color_line)
                self.ax.plot([
                    self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length
                    ,
                    self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length],
                    [self.ecg_data[idx][0] + y_offset, self.ecg_data[idx][0] + y_offset],
                    linewidth=self.ecg_meta_data.line_width,
                    color=self.ecg_meta_data.color_line)
                self.ax.plot(np.arange(self.ecg_meta_data.x_min, self.ecg_meta_data.x_max, self.ecg_meta_data.step) +
                             self.ecg_meta_data.line_length_before_calibration_signal + self.ecg_meta_data.calibration_signal_total_length,
                             self.ecg_data[idx][y_start:y_end] + y_offset,
                             linewidth=self.ecg_meta_data.line_width * self.ecg_meta_data.updated_display_factor,
                             color=self.ecg_meta_data.color_line)
                self.ax.tick_params(axis='x', bottom=False, labelbottom=False, which='both')
                self.ax.tick_params(axis='y', left=False, labelleft=False, which='both')

    def plot(self, to_save=False, file_name=None, to_plot=False):
        self.__init_plot__()
        self.__init_grid__()
        self.__plot_short_leads__()
        self.__plot_long_leads__()
        if to_save and file_name is not None:
            plt.savefig(file_name, dpi=300)
        plt.ioff()
        if to_plot:
            plt.show()
        return

    def get_numpy_array(self, dpi=300):
        self.plot()
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.close()
        return img

    def __load_ecg_format_template__(self):
        self.__init_plot__()
        format_type = self.ecg_meta_data.get_format_id()
        template_path = os.path.join(os.getcwd(), 'ecg_formats_templates', f'grid_template_{format_type}.png')
        self.template = cv2.imread(template_path)

    def __copy_template_to_fig__(self):
        # copy self.template to self.fig and self.ax
        self.fig.figimage(self.template, 0, 0, zorder=0, alpha=1)
        self.ax = self.fig.add_subplot(111)

    def __normalize_ecg_data__(self):
        from scipy.signal import sosfiltfilt, butter

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            return butter(order, [low, high], analog=False, btype="band", output="sos")

        def butter_bandpass_forward_backward_filter(data, lowcut, highcut, fs, order=5):
            sos = butter_bandpass(lowcut, highcut, fs, order=order)
            return sosfiltfilt(sos,
                               data)  # Apply a digital filter forward and backward to a signal.This function applies a linear digital filter twice, once forward and once backwards. The combined filter has zero phase and a filter order twice that of the original.

        lowcut = 0.05 * 3.3 * 0.5
        highcut = 5
        fs = 500
        filtered_signal = butter_bandpass_forward_backward_filter(self.ecg_data, lowcut, highcut, fs, order=4)
        self.ecg_data = hp.filter_signal(filtered_signal, 2, sample_rate=500, order=2, filtertype='highpass')
        # self.ecg_data = hp.filter_signal(self.ecg_data, cutoff=5, sample_rate=400, order=5, filtertype='highpass')

    def __remove_baseline_filter__(self, sample_rate):
        fc = 0.8  # [Hz], cutoff frequency
        fst = 0.2  # [Hz], rejection band
        rp = 0.5  # [dB], ripple in passband
        rs = 40# Was 40  # [dB], attenuation in rejection band
        wn = fc / (sample_rate / 2)
        wst = fst / (sample_rate / 2)

        filterorder, aux = sgn.ellipord(wn, wst, rp, rs)
        return sgn.iirfilter(filterorder, wn, rp, rs, btype='high', ftype='ellip', output='sos')

    def __preprocess_ecg_data__(self):
        # for idx in range(len(self.ecg_data)):
            # self.ecg_data[idx] = hp.scale_data(self.ecg_data[idx], 0, 1.5)
            # self.ecg_data[idx] = hp.filter_signal(self.ecg_data[idx], cutoff=2,
            #                                       sample_rate=self.ecg_meta_data.get_sample_rate(), order=5,
            #                                       filtertype='highpass')

        self.ecg_data = sgn.sosfiltfilt(self.__remove_baseline_filter__(self.ecg_meta_data.get_sample_rate()),
                                        self.ecg_data, padtype='constant', axis=-1) #supose to put the signal in the middle

        # self.ecg_data = hp.filter_signal(self.ecg_data, cutoff=40,
        #                                     sample_rate=self.ecg_meta_data.get_sample_rate(), order=5,
        #                                     filtertype='lowpass')



def load_ecg_template_by_format_type(format_type: int):
    template_path = os.path.join(os.getcwd(), 'ecg_formats_templates', f'grid_template_{format_type}.png')
    return cv2.imread(template_path)


def draw_leads_on_grid(ecg_data, ecg_meta_data):
    img_generator = ECGGenerator(ecg_data, ecg_meta_data)
    img_generator.__load_ecg_format_template__()
    img_generator.__copy_template_to_fig__()
    img_generator.__plot_short_leads__()
    img_generator.__plot_long_leads__()
    experiments_folder = os.path.join(os.getcwd(), 'experiments')
    img_generator.ax.figure.savefig(os.path.join(os.getcwd(), 'experiments', 'tmp.png'), dpi=300)


if __name__ == '__main__':
    pass