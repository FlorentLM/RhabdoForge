import dearpygui.dearpygui as dpg
import numpy as np
import collections

from pyglm import glm
from insectvision.renderers.commons import EyeOutput, OmmatidiaProjection, Colormap
from insectvision.interactive.utils import DisplayMode


class Dashboard:
    MODE_COLORS = {
        'Raw': (80, 80, 80, 80),
        'Ommatidium': (40, 100, 40, 80),
        'Cartridge': (40, 40, 100, 80)
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
        self.history_intervals = [[0, None, 'Ommatidium']]
        self._initialised = False

        self._main_thread_queue = []

        self.ui_tags = {}  # dpg item tags for syncing

    def _setup_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title='InsectVision Dashboard', width=650, height=950, vsync=self.ctx.vsync)
        dpg.setup_dearpygui()

        ra = self.ctx.renderer._ra
        self.rec_data_buffers = [collections.deque(maxlen=self.plot_history_len) for _ in range(ra.receptor_count)]

        with dpg.window(label='Inspector', width=650, height=950, no_close=True, no_move=True, tag='main_window'):

            # Info panel

            with dpg.group(tag='info_panel'):
                with dpg.group(horizontal=True):
                    dpg.add_text("FPS: 00.0", tag="ui_fps_text", color=[150, 255, 150])
                    dpg.add_text("| Renderer: Unknown", tag="ui_renderer_text")

                dpg.add_text("Pos: [0.00, 0.00, 0.00]", tag="ui_pos_text", color=[200, 200, 255])
                dpg.add_text("Samples: 0/om | 0/px", tag="ui_samples_text")
                dpg.add_separator()

            # Tabs

            with dpg.tab_bar(tag='main_tabs'):

                # Tab 1: Plots
                with dpg.tab(label='Plots', tag='tab_plots'):

                    dpg.add_spacer(height=5)
                    dpg.add_text('Ommatidium / Receptor Selection', color=[100, 255, 100])
                    self.omm_slider = dpg.add_slider_int(label="Lens ID", default_value=0, max_value=ra.lens_count - 1)

                    self.show_all_rec_toggle = dpg.add_checkbox(
                        label=f'Show individual receptors (R1-R{ra.receptor_count})', default_value=False)
                    self.rec_slider = dpg.add_slider_int(label='Probed Receptor ID', default_value=0,
                                                         max_value=max(0, ra.receptor_count - 1),
                                                         show=(ra.receptor_count > 1))

                    self.rgb_toggle = dpg.add_checkbox(label='Show RGB channels', default_value=False)
                    self.instant_toggle = dpg.add_checkbox(label='Show instantaneous (Alpha)', default_value=False)

                    # Optical plot
                    with dpg.plot(label="Optical Signal", height=300, width=-1, tag='response_plot'):
                        dpg.add_plot_legend()
                        self.bg_layer_1 = dpg.add_draw_node(tag='plot_bg_layer_1')
                        dpg.add_plot_axis(dpg.mvXAxis, label='Frame', tag='x_axis_1')
                        dpg.add_plot_axis(dpg.mvYAxis, label='Intensity', tag='y_axis_1')

                        self.series_mean = dpg.add_line_series([], [], label='Mean (EMA)', parent='y_axis_1')
                        self.series_instant = dpg.add_line_series([], [], label="Instant", parent='y_axis_1')
                        self.series_r = dpg.add_line_series([], [], label='Red', parent='y_axis_1')
                        self.series_g = dpg.add_line_series([], [], label='Green', parent='y_axis_1')
                        self.series_b = dpg.add_line_series([], [], label='Blue', parent='y_axis_1')

                        self.series_receptors = []
                        for r in range(ra.receptor_count):
                            tag = dpg.add_line_series([], [], label=f'R{r + 1}', parent='y_axis_1')
                            self.series_receptors.append(tag)

                        dpg.set_axis_limits('y_axis_1', 0.0, 1.1)

                    # Actuation plot
                    with dpg.plot(label="Actuation (um)", height=240, width=-1, tag='actuation_plot'):
                        dpg.add_plot_legend()
                        self.bg_layer_2 = dpg.add_draw_node(tag='plot_bg_layer_2')
                        dpg.add_plot_axis(dpg.mvXAxis, label='Frame', tag='x_axis_2')
                        dpg.add_plot_axis(dpg.mvYAxis, label='um', tag='y_axis_2')
                        self.series_lat = dpg.add_line_series([], [], label="Lateral", parent='y_axis_2')
                        self.series_ax = dpg.add_line_series([], [], label="Axial", parent='y_axis_2')
                        dpg.set_axis_limits('y_axis_2', -2.0, 5.0)

                # Tab 2: Dynamics
                with dpg.tab(label='Dynamics'):
                    dpg.add_spacer(height=5)
                    dpg.add_text('Photomechanical Response', color=[255, 150, 100])

                    dpg.add_checkbox(label='Enable Rhabdomeres Actuation',
                                     default_value = self.ctx.renderer.actuation,
                                     callback = lambda s, a: setattr(self.ctx.renderer, 'actuation', a))

                    dpg.add_separator()
                    dpg.add_text('Shader Parameters (EyeDynamics.comp)', color=[100, 200, 255])

                    dpg.add_slider_float(
                        label='Lateral Gain (um)',
                        default_value=self.ctx.renderer.gain_lat,
                        min_value=0.0, max_value=10.0,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'gain_lat', a)
                    )
                    dpg.add_slider_float(
                        label='Axial Gain (um)',
                        default_value=self.ctx.renderer.gain_ax,
                        min_value=0.0, max_value=20.0,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'gain_ax', a)
                    )
                    dpg.add_slider_float(
                        label='Tau Fast (s)',
                        default_value=self.ctx.renderer.tau_fast,
                        min_value=0.001, max_value=0.1,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'tau_fast', a)
                    )
                    dpg.add_slider_float(
                        label='Tau Relaxation (s)',
                        default_value=self.ctx.renderer.tau_relax,
                        min_value=0.01, max_value=0.5,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'tau_relax', a)
                    )

                    dpg.add_separator()

                    if dpg.add_button(label='Reset GPU States', width=-1):
                        self._main_thread_queue.append(
                        self.ctx.renderer.buffers['ema_state'].reset)
                        self._main_thread_queue.append(self.ctx.renderer.buffers['lens_dynamic'].reset)

                # Tab 3: Rendering
                with dpg.tab(label='Rendering'):
                    dpg.add_spacer(height=5)
                    dpg.add_text('Display Modes', color=[100, 200, 255])

                    self.ui_tags['view_mode'] = dpg.add_combo(
                        list(DisplayMode.__members__.keys()),
                        label='View Mode',
                        default_value=self.ctx.view_mode.name,
                        callback=self._set_view_mode
                    )

                    self.ui_tags['output_mode'] = dpg.add_radio_button(
                        ['Raw', 'Ommatidium', 'Cartridge'],
                        default_value=self.ctx.renderer.output_mode.name,
                        callback=self._change_output_mode,
                        horizontal=True
                    )

                    self.ui_tags['proj_mode'] = dpg.add_combo(
                        list(OmmatidiaProjection.__members__.keys()),
                        label='Projection Mode',
                        default_value=self.ctx.renderer.projection_mode.name,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'projection_mode', OmmatidiaProjection[a])
                    )

                    dpg.add_separator()
                    dpg.add_text('Render Settings', color=[100, 200, 255])

                    self.ui_tags['samples'] = dpg.add_slider_int(
                        label='Samples per Receptor',
                        default_value=self.ctx.renderer.nb_samples,
                        min_value=1, max_value=1024,
                        callback=self._update_nb_samples
                    )

                    self.ui_tags['tiled_mode'] = dpg.add_checkbox(
                        label='Tiled/Voronoi Mode (V)',
                        default_value=self.ctx.renderer.tiled_mode,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'tiled_mode', a)
                    )

                    self.ui_tags['time_dither'] = dpg.add_checkbox(
                        label='Time Dithering (T)',
                        default_value=self.ctx.renderer.time_dithering,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'time_dithering', a)
                    )

                    with dpg.group(horizontal=True):
                        self.ui_tags['heatmap'] = dpg.add_checkbox(
                            label='Heatmap Overlay (H)',
                            default_value=self.ctx.renderer.overlay_enabled,
                            callback=lambda s, a: setattr(self.ctx.renderer, 'overlay_enabled', a)
                        )

                    with dpg.group(show=self.ctx.renderer.overlay_enabled):
                        dpg.add_slider_float(
                            label='Heatmap Compression',
                            default_value=self.ctx.renderer._overlay_compression,
                            min_value=0.1, max_value=2.0,
                            callback=lambda s, a: setattr(self.ctx.renderer, '_overlay_compression', a)
                            )

                        dpg.add_combo(
                            list(Colormap.__members__.keys()),
                            label='Colormap',
                            default_value=self.ctx.renderer._overlay_colormap.name,
                            callback=lambda s, a: setattr(self.ctx.renderer, '_overlay_colormap', Colormap[a])
                        )

                        dpg.add_separator()
                        dpg.add_checkbox(
                            label='Mouse Locked (TAB)',
                            default_value=self.ctx.mouse_captured,
                            tag='ui_mouse_lock',
                            callback=self._toggle_mouse_lock
                        )

                # Tab 4: Environment
                with dpg.tab(label='Environment'):
                    dpg.add_spacer(height=5)
                    dpg.add_text('Lighting', color=[255, 200, 100])

                    self.ui_tags['ambient_int'] = dpg.add_slider_float(
                        label='Ambient Intensity', default_value=self.ctx.renderer.ambient_intensity,
                        min_value=0.0, max_value=2.0,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'ambient_intensity', a)
                    )

                    self.ui_tags['sky_int'] = dpg.add_slider_float(
                        label='Skybox Intensity', default_value=self.ctx.renderer.sky_intensity,
                        min_value=0.0, max_value=5.0,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'sky_intensity', a)
                    )

                    dpg.add_separator()
                    if self.ctx.scene.sun:
                        dpg.add_text('Sun Controls', color=[255, 200, 100])
                        self.ui_tags['sun_control'] = dpg.add_checkbox(
                            label='Mouse Controls Sun (L)', default_value=self.ctx.sun_control_mode,
                            callback=lambda s, a: setattr(self.ctx, 'sun_control_mode', a)
                        )
                        self.ui_tags['sun_azimuth'] = dpg.add_slider_float(
                            label='Azimuth', default_value=self.ctx.scene.sun.azimuth,
                            min_value=-180.0, max_value=180.0,
                            callback=self._update_sun
                        )
                        self.ui_tags['sun_elevation'] = dpg.add_slider_float(
                            label='Elevation', default_value=self.ctx.scene.sun.elevation,
                            min_value=1.0, max_value=89.0,
                            callback=self._update_sun
                        )
                        self.ui_tags['sun_intensity'] = dpg.add_slider_float(
                            label='Sun Intensity', default_value=self.ctx.scene.sun.intensity,
                            min_value=0.0, max_value=10.0,
                            callback=lambda s, a: setattr(self.ctx.scene.sun, 'intensity', a)
                        )

                # Tab 5: Agent / cam
                with dpg.tab(label='Agent / Camera'):
                    dpg.add_spacer(height=5)
                    dpg.add_text('Movement', color=[200, 150, 255])

                    self.ui_tags['move_speed'] = dpg.add_slider_float(
                        label='Movement Speed', default_value=self.ctx.move_speed,
                        min_value=0.1, max_value=20.0,
                        callback=lambda s, a: setattr(self.ctx, 'move_speed', a)
                    )

                    dpg.add_separator()
                    dpg.add_text('Transform', color=[200, 150, 255])

                    # Position
                    with dpg.group(horizontal=True):
                        self.ui_tags['pos_x'] = dpg.add_drag_float(label='X', width=100, speed=0.05,
                                                                   callback=self._update_agent_pos)
                        self.ui_tags['pos_y'] = dpg.add_drag_float(label='Y', width=100, speed=0.05,
                                                                   callback=self._update_agent_pos)
                        self.ui_tags['pos_z'] = dpg.add_drag_float(label='Z', width=100, speed=0.05,
                                                                   callback=self._update_agent_pos)

                    # Rotation
                    with dpg.group(horizontal=True):
                        self.ui_tags['rot_y'] = dpg.add_drag_float(label='Yaw', width=100, speed=1.0,
                                                                   callback=self._update_agent_rot)
                        self.ui_tags['rot_p'] = dpg.add_drag_float(label='Pitch', width=100, speed=1.0,
                                                                   callback=self._update_agent_rot)
                        self.ui_tags['rot_r'] = dpg.add_drag_float(label='Roll', width=100, speed=1.0,
                                                                   callback=self._update_agent_rot)

                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='Reset Position (O)', callback=lambda: self.ctx.reset_position())
                        dpg.add_button(label='Reset Rotation (R)', callback=lambda: self.ctx.reset_rotation())

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

    # Callbacks

    def _toggle_mouse_lock(self, sender, app_data):
        import glfw
        self.ctx.mouse_captured = app_data
        mode = glfw.CURSOR_DISABLED if app_data else glfw.CURSOR_NORMAL
        glfw.set_input_mode(self.ctx.window, glfw.CURSOR, mode)

    def _change_output_mode(self, sender, app_data):
        self.ctx.renderer.output_mode = EyeOutput[app_data]
        self.history_intervals[-1][1] = self.current_frame
        self.history_intervals.append([self.current_frame, None, app_data])

    def _set_view_mode(self, sender, app_data):
        self.ctx.view_mode = DisplayMode[app_data]

    def _update_nb_samples(self, sender, app_data):
        self._main_thread_queue.append(lambda: setattr(self.ctx.renderer, 'nb_samples', app_data))

    def _update_sun(self, sender, app_data):
        if self.ctx.scene.sun:
            az = dpg.get_value(self.ui_tags['sun_azimuth'])
            el = dpg.get_value(self.ui_tags['sun_elevation'])
            self.ctx.scene.sun.from_angles(az, el, self.ctx.scene.sun.distance)

    def _update_agent_pos(self, sender, app_data):
        x = dpg.get_value(self.ui_tags['pos_x'])
        y = dpg.get_value(self.ui_tags['pos_y'])
        z = dpg.get_value(self.ui_tags['pos_z'])
        self.ctx.agent.position = glm.vec3(x, y, z)

    def _update_agent_rot(self, sender, app_data):
        y = dpg.get_value(self.ui_tags['rot_y'])
        p = dpg.get_value(self.ui_tags['rot_p'])
        r = dpg.get_value(self.ui_tags['rot_r'])
        self.ctx.agent.yaw, self.ctx.agent.pitch, self.ctx.agent.roll = np.radians(y), np.radians(p), np.radians(r)

    def _sync_ui_state(self):
        """Sync DPG widgets with Python states in case they were modified via keyboard."""

        dpg.set_value('ui_fps_text', f'FPS: {self.ctx.fps:.1f}')

        from insectvision.renderers import Raytracer, Pathtracer
        r = self.ctx.renderer
        r_type = 'Pathtracer' if isinstance(r, Pathtracer) else ('Raytracer' if isinstance(r, Raytracer) else 'Rasterizer')
        dpg.set_value('ui_renderer_text', f'| Renderer: {r_type}')

        # Position
        agent_pos = self.ctx.agent.position
        dpg.set_value('ui_pos_text', f'Pos: [{agent_pos.x:.2f}, {agent_pos.y:.2f}, {agent_pos.z:.2f}]')

        # Samples
        om_s = r.nb_samples
        px_s = getattr(r, 'samples_per_pixel', 1)
        dpg.set_value("ui_samples_text", f"Samples: {om_s}/om | {px_s}/px")

        # Rendering sync
        dpg.set_value(self.ui_tags['view_mode'], self.ctx.view_mode.name)
        dpg.set_value(self.ui_tags['proj_mode'], self.ctx.renderer.projection_mode.name)
        dpg.set_value(self.ui_tags['tiled_mode'], self.ctx.renderer.tiled_mode)
        dpg.set_value(self.ui_tags['heatmap'], self.ctx.renderer.overlay_enabled)

        if hasattr(self.ctx.renderer, 'time_dithering'):
            dpg.set_value(self.ui_tags['time_dither'], self.ctx.renderer.time_dithering)
        if hasattr(self.ctx.renderer, 'nb_samples'):
            dpg.set_value(self.ui_tags['samples'], self.ctx.renderer.nb_samples)

        # Env sync
        if self.ctx.scene.sun:
            dpg.set_value(self.ui_tags['sun_control'], self.ctx.sun_control_mode)
            dpg.set_value(self.ui_tags['sun_azimuth'], self.ctx.scene.sun.azimuth)
            dpg.set_value(self.ui_tags['sun_elevation'], self.ctx.scene.sun.elevation)
            dpg.set_value(self.ui_tags['sun_intensity'], self.ctx.scene.sun.intensity)

        # Agent sync
        dpg.set_value(self.ui_tags['pos_x'], agent_pos.x)
        dpg.set_value(self.ui_tags['pos_y'], agent_pos.y)
        dpg.set_value(self.ui_tags['pos_z'], agent_pos.z)

        dpg.set_value(self.ui_tags['rot_y'], self.ctx.agent.yaw)
        dpg.set_value(self.ui_tags['rot_p'], self.ctx.agent.pitch)
        dpg.set_value(self.ui_tags['rot_r'], self.ctx.agent.roll)

        dpg.set_value('ui_mouse_lock', self.ctx.mouse_captured)

    # Main render
    def render(self, visual_output):

        if not self._initialised:
            self._setup_dpg()

        if not dpg.is_dearpygui_running():
            return False

        # Process queued GL tasks on the main thread
        while self._main_thread_queue:
            task = self._main_thread_queue.pop(0)
            task()

        # Sync with stuff driven by keyboard
        self._sync_ui_state()

        is_plotting = dpg.is_item_visible('tab_plots')

        ra = self.ctx.renderer._ra
        lens_id = dpg.get_value(self.omm_slider)
        mode = self.ctx.renderer.output_mode

        if mode == EyeOutput.Raw:
            rec_id = dpg.get_value(self.rec_slider)
            self.ctx.renderer.selected_id = (lens_id * ra.receptor_count) + rec_id
        else:
            self.ctx.renderer.selected_id = lens_id

        if is_plotting and visual_output is not None and lens_id < ra.lens_count:
            if mode == EyeOutput.Raw:
                start, end = lens_id * ra.receptor_count, (lens_id + 1) * ra.receptor_count
                group = visual_output.receptors[start:end]
            elif mode == EyeOutput.Ommatidium:
                group = visual_output.lenses[lens_id]
            else:
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

            # Actuation buffer readback
            state_data = self.ctx.renderer.buffers['lens_dynamic'].read(start=lens_id, count=1)
            self.lat_um_data.append(float(state_data['lateral_um'][0]))
            self.ax_um_data.append(float(state_data['axial_um'][0]))

            self.current_frame += 1

            # Update plot data
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

            # Background overlays
            window_start = self.frame_data[0] if self.frame_data else 0

            for layer, y_max in [(self.bg_layer_1, 1.1), (self.bg_layer_2, 5.0)]:
                dpg.delete_item(layer, children_only=True)
                for start, end, m_name in self.history_intervals:
                    ae = end if end is not None else self.current_frame
                    if ae >= window_start:
                        dpg.draw_rectangle(pmin=[start, -2.0], pmax=[ae, y_max],
                                           fill=self.MODE_COLORS[m_name], color=(0, 0, 0, 0), parent=layer)

            if len(x) > 1:
                dpg.set_axis_limits('x_axis_1', x[0], x[-1])
                dpg.set_axis_limits('x_axis_2', x[0], x[-1])

        dpg.render_dearpygui_frame()
        return True

    def free(self):
        if self._initialised:
            dpg.destroy_context()
            self._initialised = False