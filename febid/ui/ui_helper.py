from febid.ui.main_window import Ui_MainWindow as UI_MainPanel


class UI_Group(list):
    """
    A collection of UI elements.
    """

    def __init__(self, *args):
        super().__init__()
        if type(args[0]) in [set, list, tuple]:
            self.extend(args[0])
        else:
            for arg in args:
                self.append(arg)

    def set(self):
        """
        Convert to set
        """
        return set(self)

    def disable(self):
        """
        Disable UI element.

        :return:
        """
        for element in self:
            element.setEnabled(False)

    def enable(self):
        """
        Enable UI element.
        :return:
        """
        for element in self:
            element.setEnabled(True)


class RadioButtonGroup(UI_Group):
    """
    A collection of bound radio buttons.
    """
    def __init__(self, *args, names=None):
        super().__init__(*args)
        self.names = names

    def setChecked(self, param):
        """
        Check one of the radio buttons based on its name.

        :param param: name of the button to check
        :return:
        """
        index = self.names.index(param)
        self[index].setChecked(True)

    def getChecked(self):
        """
        Get the name of the checked button.

        :return: name of checked button
        """
        for i, button in enumerate(self):
            if button.isChecked():
                return self.names[i]


class UIHelper:
    """
    Helper class for UI elements grouping and enabling/disabling.
    """

    def __init__(self, ui: UI_MainPanel):
        self.ui = ui
        self._organize_ui_elements()

    def _organize_ui_elements(self):
        """
        Group all UI element helper initialization.
        """
        self.__group_interface_elements()
        self.__aggregate_radio_buttons()

    def __group_interface_elements(self):
        """
        Group interface elements for easier enabling/disabling.

        :return:
        """
        # Groups of controls on the panel for easier Enabling/Disabling

        # Inputs and their labels
        self.ui_dimensions = UI_Group(self.ui.input_width, self.ui.input_length, self.ui.input_height,
                                      self.ui.l_width, self.ui.l_height, self.ui.l_length,
                                      self.ui.l_dimensions_units)
        self.ui_dimensions_mc = UI_Group(self.ui.input_width_mc, self.ui.input_length_mc, self.ui.input_height_mc,
                                         self.ui.l_width_mc, self.ui.l_height_mc, self.ui.l_length_mc,
                                         self.ui.l_dimensions_units_mc)
        self.ui_cell_size = UI_Group(self.ui.l_cell_size, self.ui.input_cell_size, self.ui.l_cell_size_units)
        self.ui_cell_size_mc = UI_Group(self.ui.l_cell_size_mc, self.ui.input_cell_size_mc,
                                        self.ui.l_cell_size_units_mc)
        self.ui_substrate_height = UI_Group(self.ui.l_substrate_height, self.ui.input_substrate_height,
                                            self.ui.l_substrate_height_units)
        self.ui_substrate_height_mc = UI_Group(self.ui.l_substrate_height_mc, self.ui.input_substrate_height_mc,
                                               self.ui.l_substrate_height_units_mc)

        self.ui_pattern_param1 = UI_Group(self.ui.l_param1, self.ui.input_param1, self.ui.l_param1_units)
        self.ui_pattern_param2 = UI_Group(self.ui.l_param2, self.ui.input_param2, self.ui.l_param2_units)
        self.ui_dwell_time = UI_Group(self.ui.l_dwell_time, self.ui.input_dwell_time, self.ui.l_dwell_time_units)
        self.ui_pitch = UI_Group(self.ui.l_pitch, self.ui.input_pitch, self.ui.l_pitc_units)
        self.ui_repeats = UI_Group(self.ui.l_repeats, self.ui.input_repeats)

        self.ui_hfw = UI_Group(self.ui.l_hfw, self.ui.input_hfw, self.ui.l_hfw_units)

        self.ui_sim_data_interval = UI_Group(self.ui.l_sim_data_interval, self.ui.input_simulation_data_interval,
                                             self.ui.l_sim_data_interval_units)
        self.ui_snapshot = UI_Group(self.ui.l_snapshot_interval, self.ui.input_structure_snapshot_interval,
                                    self.ui.l_snapshot_interval_units)
        self.ui_unique_name = UI_Group(self.ui.l_unique_name, self.ui.input_unique_name)
        self.ui_save_folder = UI_Group(self.ui.open_save_folder_button, self.ui.save_folder_display)

        # Grouping elements by their designation
        self.ui_vtk_choice = UI_Group(self.ui.open_vtk_file_button, self.ui.vtk_filename_display)
        self.ui_vtk_choice_mc = UI_Group(self.ui.open_vtk_file_button_mc, self.ui.vtk_filename_display_mc)

        self.ui_geom_choice = UI_Group(
            {self.ui.open_geom_parameters_file_button} | self.ui_dimensions.set() | self.ui_cell_size.set() | \
            self.ui_substrate_height.set())
        self.ui_geom_choice_mc = UI_Group({self.ui.open_geom_parameters_file_button_mc} | self.ui_dimensions_mc.set() | \
                                          self.ui_cell_size_mc.set() | self.ui_substrate_height_mc.set())

        self.ui_auto_choice = UI_Group(self.ui_cell_size.set() | self.ui_substrate_height.set())

        self.ui_simple_patterns = UI_Group(
            {self.ui.pattern_selection} | self.ui_pattern_param1.set() | self.ui_pattern_param2.set() | \
            self.ui_dwell_time.set() | self.ui_pitch.set() | self.ui_repeats.set())

        self.ui_stream_file = UI_Group({self.ui.open_stream_file_button} | self.ui_hfw.set())

        # Grouping by the groupBoxes
        self.ui_sim_volume = UI_Group(self.ui_vtk_choice.set() | self.ui_geom_choice.set() | self.ui_auto_choice.set())
        self.ui_sim_volume_mc = UI_Group(self.ui_vtk_choice_mc.set() | self.ui_geom_choice_mc.set())
        self.ui_pattern = UI_Group(self.ui_simple_patterns.set() | self.ui_stream_file.set())

    def __aggregate_radio_buttons(self):
        """
        Aggregate radio buttons into a group.

        :return:
        """
        self.radio_buttons_structure_source = RadioButtonGroup(self.ui.choice_vtk_file,
                                                               self.ui.choice_geom_parameters_file,
                                                               self.ui.choice_auto, names=['vtk', 'geom', 'auto'])
        self.radio_buttons_pattern_source = RadioButtonGroup(self.ui.choice_simple_pattern, self.ui.choice_stream_file,
                                                             names=['simple', 'stream_file'])
        self.radio_buttons_viz_data = None

    def set_vtk_chosen(self):
        # Changing FEBID tab interface
        self.ui.choice_vtk_file.setChecked(True)
        self.ui_sim_volume.disable()
        self.ui_vtk_choice.enable()

        # Changing MC tab interface
        self.ui.choice_vtk_file_mc.setChecked(True)
        self.ui_sim_volume_mc.disable()
        self.ui_vtk_choice_mc.enable()

    def set_geom_chosen(self):
        # Changing FEBID tab interface
        self.ui.choice_geom_parameters_file.setChecked(True)
        self.ui_sim_volume.disable()
        self.ui_geom_choice.enable()

        # Changing MC tab interface
        self.ui.choice_geom_parameters_file_mc.setChecked(True)
        self.ui_sim_volume_mc.disable()
        self.ui_geom_choice_mc.enable()

    def set_auto_chosen(self):
        # Changing FEBID tab interface
        self.ui.choice_auto.setChecked(True)
        self.ui_sim_volume.disable()
        self.ui_auto_choice.enable()

        # Changing MC tab interface
        self.ui.choice_geom_parameters_file_mc.setAutoExclusive(False)
        self.ui.choice_geom_parameters_file_mc.setChecked(False)
        self.ui.choice_geom_parameters_file_mc.setAutoExclusive(True)
        self.ui.choice_vtk_file_mc.setAutoExclusive(False)
        self.ui.choice_vtk_file_mc.setChecked(False)
        self.ui.choice_vtk_file_mc.setAutoExclusive(True)

    def set_simple_pattern_chosen(self):
        # Changing FEBID tab interface
        self.ui.choice_simple_pattern.setChecked(True)
        self.ui_pattern.disable()
        self.ui_simple_patterns.enable()

    def set_stream_file_chosen(self):
        # Changing FEBID tab interface
        self.ui.choice_stream_file.setChecked(True)
        self.ui_pattern.disable()
        self.ui_stream_file.enable()

    def set_simple_pattern_change(self, current):
        if current == 'Point':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.enable()
        if current == 'Line':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.disable()
        if current == 'Rectangle':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.enable()
        if current == 'Square':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.disable()
        if current == 'Triangle':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.disable()
        if current == 'Circle':
            self.ui_pattern_param1.enable()
            self.ui_pattern_param2.disable()

    def set_state_save_sim_data(self, param):
        switch = bool(param)
        self.ui.checkbox_save_simulation_data.setChecked(switch)
        if switch:
            self.ui_sim_data_interval.enable()
        else:
            self.ui_sim_data_interval.disable()
        if switch or self.ui.checkbox_save_snapshots.isChecked():
            self.ui_unique_name.enable()
            self.ui_save_folder.enable()
        else:
            self.ui_unique_name.disable()
            self.ui_save_folder.disable()

    def set_state_save_snapshots(self, param):
        switch = bool(param)
        self.ui.checkbox_save_snapshots.setChecked(switch)
        if switch:
            self.ui_snapshot.enable()
        else:
            self.ui_snapshot.disable()
        if switch or self.ui.checkbox_save_simulation_data.isChecked():
            self.ui_unique_name.enable()
            self.ui_save_folder.enable()
        else:
            self.ui_unique_name.disable()
            self.ui_save_folder.disable()

    def set_pattern_source(self, source_name):
        """
        Select the radio button of the specified pattern source

        :param source_name:
        :return:
        """
        self.radio_buttons_pattern_source.setChecked(source_name)

    def set_structure_source(self, source_name):
        """
        Select the radio button of the specified structure source

        :param source_name:
        :return:
        """
        self.radio_buttons_structure_source.setChecked(source_name)