from math import ceil


class ECGFormat:
    def __init__(self, num_leads_horizontal=2, num_leads_vertical=6, num_rhythm_leads=1, num_pages=1, paper_speed=25,
                 voltage_gain=10, format_id=0):
        self.num_leads_horizontal = num_leads_horizontal
        self.num_leads_vertical = num_leads_vertical
        self.num_rhythm_leads = num_rhythm_leads
        self.num_pages = num_pages
        self.paper_speed = paper_speed  # mm/s
        self.voltage_gain = voltage_gain  # mm/mV
        self.num_leads = num_leads_horizontal * num_leads_vertical
        self.format_id = format_id

    def get_num_leads(self):
        return self.num_leads

    def get_num_leads_horizontal(self):
        return self.num_leads_horizontal

    def get_num_leads_vertical(self):
        return self.num_leads_vertical

    def get_num_rhythm_leads(self):
        return self.num_rhythm_leads

    def get_num_pages(self):
        return self.num_pages

    def get_paper_speed(self):
        return self.paper_speed

    def get_voltage_gain(self):
        return self.voltage_gain

    def get_format_id(self):
        return self.format_id


class ECGParameters:
    def __init__(self, sample_rate=500, title='ECG', lead_index=None, lead_order=None, style='default', columns=4,
                 row_height=6, show_lead_name=True, show_grid=True, show_separate_line=False, long_lead_indexes=None):
        self.sample_rate = sample_rate
        self.title = title
        self.lead_index = lead_index
        self.lead_order = lead_order
        self.style = style
        self.columns = columns
        self.row_height = row_height
        self.show_lead_name = show_lead_name
        self.show_grid = show_grid
        self.show_separate_line = show_separate_line
        self.long_lead_indexes = long_lead_indexes

    def get_sample_rate(self):
        return self.sample_rate

    def get_title(self):
        return self.title

    def get_lead_index(self):
        return self.lead_index

    def get_lead_order(self):
        return self.lead_order

    def get_style(self):
        return self.style

    def get_columns(self):
        return self.columns

    def get_row_height(self):
        return self.row_height

    def get_show_lead_name(self):
        return self.show_lead_name

    def get_show_grid(self):
        return self.show_grid

    def get_show_separate_line(self):
        return self.show_separate_line

    def get_long_lead_indexes(self):
        return self.long_lead_indexes

    def get_num_leads(self):
        return len(self.lead_order)

    def get_num_long_leads(self):
        return len(self.long_lead_indexes)


class ECGMetaData:
    def __init__(self, ecg_len=5000, num_leads_horizontal=4, num_leads_vertical=3, num_rhythm_leads=0,
                 num_pages=1,
                 paper_speed=25,
                 voltage_gain=10, sample_rate=500, title='', lead_index=None, lead_order=None, style='default',
                 columns=4,
                 row_height=6, show_lead_name=True, show_grid=True, show_separate_line=False, long_lead_indexes=None,
                 format_id=0):
        self.ecg_format = ECGFormat(num_leads_horizontal, num_leads_vertical, num_rhythm_leads, num_pages, paper_speed,
                                    voltage_gain, format_id=format_id)
        self.ecg_parameters = ECGParameters(sample_rate, title, lead_index, lead_order, style, columns, row_height,
                                            show_lead_name, show_grid, show_separate_line, long_lead_indexes)
        self.ecg_len = ecg_len
        self.calibration_signal_total_length = 0.20
        self.line_length_before_calibration_signal = 0.04
        self.__init_lead_lists__()
        self.__init_grid_properties__()

    def __init_lead_lists__(self):
        if self.ecg_parameters.lead_index is None:
            self.ecg_parameters.lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if self.ecg_parameters.lead_order is None:
            # The order is as such:
            # I, II, III, aVL, aVF, aVR, V1, V2, V3, V4, V5, V6
            self.ecg_parameters.lead_order = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]

    def __init_grid_properties__(self):
        self.secs_to_display_per_column = (self.ecg_len / self.ecg_parameters.sample_rate) / \
                                          self.ecg_parameters.columns
        self.short_lead_rows = int(ceil(len(self.ecg_parameters.lead_index) / self.ecg_parameters.columns))
        self.total_rows = self.short_lead_rows + len(self.ecg_parameters.long_lead_indexes) if \
            self.ecg_parameters.long_lead_indexes is not None else self.short_lead_rows
        self.original_display_factor = 1
        self.updated_display_factor = self.original_display_factor ** 0.5
        self.line_width = 0.7
        self.step = 1.0 / self.ecg_parameters.sample_rate
        self.x_min = 0
        self.x_max = (self.ecg_parameters.columns * self.secs_to_display_per_column) #- self.step # self.step is only used with the br dataset.
        self.y_min = (self.ecg_parameters.row_height / 4) - ((self.total_rows / 2) * self.ecg_parameters.row_height)
        self.y_max = (self.ecg_parameters.row_height / 4)
        self.__init_colors__()

    def __init_colors__(self):
        if self.ecg_parameters.style == 'default':
            self.color_major = (1, 0, 0)
            self.color_minor = (1, 0.7, 0.7)
        else:
            self.color_major = (0.4, 0.4, 0.4)
            self.color_minor = (0.75, 0.75, 0.75)
        self.color_line = (0, 0, 0)

    def get_ecg_format(self):
        return self.ecg_format

    def get_format_id(self):
        return self.ecg_format.get_format_id()

    def get_ecg_parameters(self):
        return self.ecg_parameters

    def get_fig_size(self):
        return self.secs_to_display_per_column * self.ecg_parameters.columns * self.original_display_factor, \
               self.total_rows * self.ecg_parameters.row_height / 5 * self.original_display_factor

    def get_title(self):
        return self.ecg_parameters.title

    def get_show_grid(self):
        return self.ecg_parameters.show_grid

    def to_plot_long_leads(self):
        return self.ecg_parameters.long_lead_indexes is not None

    def get_long_lead_indexes(self):
        return self.ecg_parameters.long_lead_indexes

    def get_row_height(self):
        return self.ecg_parameters.row_height

    def get_sample_rate(self):
        return self.ecg_parameters.sample_rate

    def get_show_lead_name(self):
        return self.ecg_parameters.show_lead_name

    def get_lead_index(self):
        return self.ecg_parameters.lead_index

    def get_columns(self):
        return self.ecg_parameters.columns

    def get_num_leads(self):
        return len(self.ecg_parameters.lead_order)

    def get_lead_order(self):
        return self.ecg_parameters.lead_order

    def to_show_separate_line(self):
        return self.ecg_parameters.show_separate_line

    def to_show_lead_name(self):
        return self.ecg_parameters.show_lead_name


# ECGMetaDataOptions = [
#     ECGMetaData(long_lead_indexes=[6, 1, 10], format_id=0),
#     ECGMetaData(long_lead_indexes=[1], format_id=1),
#     ECGMetaData(long_lead_indexes=[1, 6], format_id=2),
#     ECGMetaData(columns=2, format_id=3),
#     ECGMetaData(columns=2, long_lead_indexes=[1], format_id=4),
#     # ECGMetaData(columns=1, format_id=5)
# ]
