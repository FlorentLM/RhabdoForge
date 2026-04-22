import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import glBindBuffer, glGetBufferSubData, GL_SHADER_STORAGE_BUFFER
import dearpygui.dearpygui as dpg
import numpy as np
import collections

from insectvision.compound_eyes.datatypes import LENS_DYNAMIC_DTYPE
from insectvision.renderers.commons import EyeOutput


class Dashboard:
    MODE_COLORS = {
        "Raw": (80, 80, 80, 80),
        "Ommatidium": (40, 100, 40, 80),
        "Cartridge": (40, 40, 100, 80)
    }

    REC_PALETTE = [
        (255, 0, 0), (0, 255, 0), (0, 150, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (255, 165, 0), (150, 100, 255), (200, 200, 200)
    ]

    def __init__(self, context):
        self.ctx = context
        self.plot_history_len = 200

        self.frame_data = collections.deque(maxlen=self.plot_history_len)
        self.response_mean = collections.deque(maxlen=self.plot_history_len)
        self.response_instant = collections.deque(maxlen=self.plot_history_len)
        self.r_data = collections.deque(maxlen=self.plot_history_len)
        self.g_data = collections.deque(maxlen=self.plot_history_len)
        self.b_data = collections.deque(maxlen=self.plot_history_len)

        self.rec_data_buffers = []

        self.lat_um_data = collections.deque(maxlen=self.plot_history_len)
        self.ax_um_data = collections.deque(maxlen=self.plot_history_len)

        self.current_frame = 0
        self.history_intervals = [[0, None, "Ommatidium"]]
        self._initialised = False

    def _setup_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title='InsectVision Dashboard', width=600, height=950)
        dpg.setup_dearpygui()

        with dpg.window(label="Inspector", width=600, height=950, no_close=True, no_move=True, tag="main_window"):
            with dpg.menu_bar():
                with dpg.menu(label="Settings"):
                    dpg.add_menu_item(label="Reset Position", callback=lambda: self.ctx.reset_position())
                    dpg.add_menu_item(label="Toggle Sun Control", callback=lambda: self.ctx.toggle_sun_control())

            dpg.add_text("Rendering Options", color=[100, 200, 255])
            dpg.add_radio_button(["Raw", "Ommatidium", "Cartridge"],
                                 default_value=self.ctx.renderer.output_mode.name,
                                 callback=self._change_output_mode, horizontal=True)

            dpg.add_separator()

            ra = self.ctx.renderer._ra

            self.rec_data_buffers = [collections.deque(maxlen=self.plot_history_len) for _ in range(ra.receptor_count)]

            dpg.add_text("Ommatidium / Receptor Selection", color=[100, 255, 100])
            self.omm_slider = dpg.add_slider_int(label="Lens ID", default_value=0, max_value=ra.lens_count - 1)

            self.show_all_rec_toggle = dpg.add_checkbox(label=f"Show individual receptors (R1-R{ra.receptor_count})",
                                                        default_value=False)

            self.rec_slider = dpg.add_slider_int(label="Probed Receptor ID", default_value=0,
                                                 max_value=max(0, ra.receptor_count - 1),
                                                 show=(ra.receptor_count > 1))

            self.rgb_toggle = dpg.add_checkbox(label="Show RGB channels", default_value=False)
            self.instant_toggle = dpg.add_checkbox(label="Show instantaneous (Alpha)", default_value=False)

            # Optical plot
            with dpg.plot(label="Optical Signal", height=300, width=-1, tag="response_plot"):
                dpg.add_plot_legend()
                self.bg_layer_1 = dpg.add_draw_node(tag="plot_bg_layer_1")
                dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_1")
                dpg.add_plot_axis(dpg.mvYAxis, label="Intensity", tag="y_axis_1")

                self.series_mean = dpg.add_line_series([], [], label="Mean (EMA)", parent="y_axis_1")
                self.series_instant = dpg.add_line_series([], [], label="Instant", parent="y_axis_1")
                self.series_r = dpg.add_line_series([], [], label="Red", parent="y_axis_1")
                self.series_g = dpg.add_line_series([], [], label="Green", parent="y_axis_1")
                self.series_b = dpg.add_line_series([], [], label="Blue", parent="y_axis_1")

                self.series_receptors = []
                for r in range(ra.receptor_count):
                    tag = dpg.add_line_series([], [], label=f"R{r + 1}", parent="y_axis_1")
                    self.series_receptors.append(tag)

                dpg.set_axis_limits("y_axis_1", 0.0, 1.1)

            # Actuation plot
            with dpg.plot(label="Actuation (um)", height=240, width=-1, tag="actuation_plot"):
                dpg.add_plot_legend()
                self.bg_layer_2 = dpg.add_draw_node(tag="plot_bg_layer_2")
                dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_2")
                dpg.add_plot_axis(dpg.mvYAxis, label="um", tag="y_axis_2")
                self.series_lat = dpg.add_line_series([], [], label="Lateral", parent="y_axis_2")
                self.series_ax = dpg.add_line_series([], [], label="Axial", parent="y_axis_2")
                dpg.set_axis_limits("y_axis_2", -2.0, 5.0)

        self._apply_styles()
        dpg.show_viewport()
        self._initialised = True

    def _apply_styles(self):
        styles = {
            self.series_r: (255, 50, 50),
            self.series_g: (50, 255, 50),
            self.series_b: (50, 50, 255),
            self.series_instant: (200, 200, 200, 150),
            self.series_lat: (0, 200, 255),
            self.series_ax: (255, 150, 0),
        }

        for item, color in styles.items():
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)

            dpg.bind_item_theme(item, theme)

        for i, item in enumerate(self.series_receptors):
            color = self.REC_PALETTE[i % len(self.REC_PALETTE)]

            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)

            dpg.bind_item_theme(item, theme)

    def _change_output_mode(self, sender, app_data):
        mode_map = {"Raw": EyeOutput.Raw, "Ommatidium": EyeOutput.Ommatidium, "Cartridge": EyeOutput.Cartridge}

        self.ctx.renderer.output_mode = mode_map.get(app_data, EyeOutput.Ommatidium)
        self.history_intervals[-1][1] = self.current_frame
        self.history_intervals.append([self.current_frame, None, app_data])

    def render(self, visual_output):

        if not self._initialised:
            self._setup_dpg()

        if not dpg.is_dearpygui_running():
            return False

        ra = self.ctx.renderer._ra
        lens_id = dpg.get_value(self.omm_slider)
        mode = self.ctx.renderer.output_mode

        if mode == EyeOutput.Raw:
            # If we are looking at all receptors, rec_id picks which one to highlight.
            # If we are not, rec_id still picks which specific tile glows.
            rec_id = dpg.get_value(self.rec_slider)
            self.ctx.renderer.selected_id = (lens_id * ra.receptor_count) + rec_id
        else:
            self.ctx.renderer.selected_id = lens_id

        if visual_output is not None and lens_id < ra.lens_count:
            if mode == EyeOutput.Raw:
                start, end = lens_id * ra.receptor_count, (lens_id + 1) * ra.receptor_count
                group = visual_output.receptors[start:end]

            elif mode == EyeOutput.Ommatidium:
                group = visual_output.lenses[lens_id]

            else:  # Cartridge
                group = visual_output.cartridges[lens_id]

            self.frame_data.append(self.current_frame)

            avg_pixel = np.mean(group, axis=0)
            self.response_mean.append(np.mean(avg_pixel[:3]))
            self.r_data.append(avg_pixel[0])
            self.g_data.append(avg_pixel[1])
            self.b_data.append(avg_pixel[2])
            self.response_instant.append(avg_pixel[3])

            for r_idx in range(ra.receptor_count):
                self.rec_data_buffers[r_idx].append(np.mean(group[r_idx, :3]))

            # Download Actuation State from GPU to CPU to graph it
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ctx.renderer._lens_dynamic_ssbo)
            state_bytes = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ra.lens_count * 16)
            state_data = np.frombuffer(state_bytes, dtype=LENS_DYNAMIC_DTYPE)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            # Now you can use it:
            self.lat_um_data.append(state_data['lateral_um'][lens_id])
            self.ax_um_data.append(state_data['axial_um'][lens_id])

            self.lat_um_data.append(getattr(ra, '_lateral_state', np.zeros(ra.lens_count))[lens_id])
            self.ax_um_data.append(getattr(ra, '_axial_state', np.zeros(ra.lens_count))[lens_id])

            self.current_frame += 1

            x = list(self.frame_data)
            show_all_rec = dpg.get_value(self.show_all_rec_toggle)
            show_rgb = dpg.get_value(self.rgb_toggle)

            dpg.configure_item(self.series_mean, show=not (show_all_rec or show_rgb))

            for s in [self.series_r, self.series_g, self.series_b]:
                dpg.configure_item(s, show=show_rgb and not show_all_rec)

            for s in self.series_receptors:
                dpg.configure_item(s, show=show_all_rec)

            if show_all_rec:
                for i, s_tag in enumerate(self.series_receptors):
                    dpg.set_value(s_tag, [x, list(self.rec_data_buffers[i])])

            elif show_rgb:
                dpg.set_value(self.series_r, [x, list(self.r_data)])
                dpg.set_value(self.series_g, [x, list(self.g_data)])
                dpg.set_value(self.series_b, [x, list(self.b_data)])
            else:
                dpg.set_value(self.series_mean, [x, list(self.response_mean)])

            dpg.set_value(self.series_instant, [x, list(self.response_instant)])
            dpg.configure_item(self.series_instant, show=dpg.get_value(self.instant_toggle))

            dpg.set_value(self.series_lat, [x, list(self.lat_um_data)])
            dpg.set_value(self.series_ax, [x, list(self.ax_um_data)])

            # Background
            window_start = self.frame_data[0] if self.frame_data else 0

            for layer, y_max in [(self.bg_layer_1, 1.1), (self.bg_layer_2, 5.0)]:
                dpg.delete_item(layer, children_only=True)

                for start, end, m_name in self.history_intervals:
                    ae = end if end is not None else self.current_frame
                    if ae >= window_start:
                        dpg.draw_rectangle(pmin=[start, -2.0], pmax=[ae, y_max],
                                           fill=self.MODE_COLORS[m_name], color=(0, 0, 0, 0), parent=layer)

            if len(x) > 1:
                dpg.set_axis_limits("x_axis_1", x[0], x[-1])
                dpg.set_axis_limits("x_axis_2", x[0], x[-1])

        dpg.render_dearpygui_frame()

        return True

    def free(self):
        if self._initialised:
            dpg.destroy_context()
            self._initialised = False