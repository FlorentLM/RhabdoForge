import dearpygui.dearpygui as dpg
import numpy as np
import collections
from pyglm import glm

from insectvision.compound_eyes.rhabdomeres import RHAB_COLOURS
from insectvision.utils.shared import EyeOutput, OmmatidiaProjection, Colormap, DisplayMode, RandomnessMode, SamplingMode


class Dashboard:
    MODE_COLORS = {
        'Raw': (80, 80, 80, 80),
        'Ommatidium': (40, 100, 40, 80),
        'Cartridge': (40, 40, 100, 80)
    }

    def __init__(self, context):
        
        self.ctx = context
        self.plot_history_len = 200

        self.frame_data = collections.deque(maxlen=self.plot_history_len)

        self.selected_ommatidia = []
        self.max_selected = 10
        self.omm_histories = {}

        self.current_frame = 0
        self.history_intervals = [[0, None, 'Ommatidium']]
        self._initialised = False

        self._main_thread_queue = []

        self.ui_tags = {}  # dpg item tags for syncing

    def _setup_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title='InsectVision Dashboard', width=650, height=950, vsync=self.ctx.vsync)
        dpg.setup_dearpygui()

        model = self.ctx.renderer._model
        self.rec_data_buffers = [collections.deque(maxlen=self.plot_history_len) for _ in range(model.R)]

        with dpg.window(label='Inspector', width=650, height=950, no_close=True, no_move=True, tag='main_window'):

            # Info panel

            with dpg.group(tag='info_panel'):
                with dpg.group(horizontal=True):
                    dpg.add_text("FPS: 00.0", tag="ui_fps_text", color=[150, 255, 150])
                    dpg.add_text("| Renderer: Unknown", tag="ui_renderer_text")

                # Timing: hardware (wall) clock vs biological (sim) clock
                with dpg.group(horizontal=True):
                    dpg.add_text("Wall dt: -- ms", tag="ui_wall_dt_text", color=[180, 220, 255])
                    dpg.add_text("| Sim: --", tag="ui_sim_mode_text", color=[180, 220, 255])
                dpg.add_text("Total sim time: 0.000 s", tag="ui_sim_total_text", color=[180, 220, 255])

                with dpg.group(horizontal=True):
                    dpg.add_text("View: Unknown", tag="ui_view_text", color=[200, 200, 200])
                    dpg.add_text("| Proj: Unknown", tag="ui_proj_text", color=[200, 200, 200])

                dpg.add_text("Pos: [0.00, 0.00, 0.00]", tag="ui_pos_text", color=[200, 200, 255])

                with dpg.group(horizontal=True):
                    dpg.add_text("Ommatidia: 0", tag="ui_omm_text")
                    dpg.add_text("| Samples: 0/om | 0/px", tag="ui_samples_text")

                with dpg.group(horizontal=True):
                    dpg.add_text("Triangles: 0", tag="ui_tri_text", color=[255, 200, 150])
                    dpg.add_text("| Points: 0", tag="ui_pts_text", color=[255, 200, 150])

                dpg.add_separator()

            # Tabs

            with dpg.tab_bar(tag='main_tabs'):

                # Tab 1: Plots
                with dpg.tab(label='Plots', tag='tab_plots'):

                    dpg.add_spacer(height=5)
                    dpg.add_text('Ommatidium / Rhabdomere Selection', color=[100, 255, 100])
                    dpg.add_text('Click in viewport (when mouse free) to pick. Shift+click for multi.',
                                 color=[200, 200, 200])

                    self.omm_selection_text = dpg.add_text("Selected ommatidia: [0]")
                    self.omm_slider = dpg.add_slider_int(label="Primary Ommatidium ID (or type)", default_value=0,
                                                         max_value=model.N - 1, callback=self._on_slider_change)
                    dpg.add_button(label="Clear Selection", callback=lambda: self.toggle_omm_selection(None))

                    dpg.add_separator()
                    self.show_all_rec_toggle = dpg.add_checkbox(
                        label=f'Show individual rhabdomeres (R1-R{model.R})', default_value=False)
                    self.rec_slider = dpg.add_slider_int(label='Probed Rhabdomere ID', default_value=0,
                                                         max_value=max(0, model.R - 1),
                                                         show=(model.R > 1))

                    self.rgb_toggle = dpg.add_checkbox(label='Show RGB channels', default_value=False)
                    self.instant_toggle = dpg.add_checkbox(label='Show instantaneous (Alpha)', default_value=False)

                    # Optical plot
                    with dpg.plot(label="Optical Signal", height=300, width=-1, tag='response_plot'):
                        dpg.add_plot_legend()
                        self.bg_layer_1 = dpg.add_draw_node(tag='plot_bg_layer_1')
                        dpg.add_plot_axis(dpg.mvXAxis, label='Frame', tag='x_axis_1')
                        dpg.add_plot_axis(dpg.mvYAxis, label='Intensity', tag='y_axis_1')
                        dpg.set_axis_limits('y_axis_1', 0.0, 1.1)

                    # Actuation plot
                    with dpg.plot(label="Actuation (um)", height=240, width=-1, tag='actuation_plot'):
                        dpg.add_plot_legend()
                        self.bg_layer_2 = dpg.add_draw_node(tag='plot_bg_layer_2')
                        dpg.add_plot_axis(dpg.mvXAxis, label='Frame', tag='x_axis_2')
                        dpg.add_plot_axis(dpg.mvYAxis, label='um', tag='y_axis_2')
                        dpg.set_axis_limits('y_axis_2', -2.0, 5.0)

                    self.series_pool = []
                    for i in range(self.max_selected):
                        pool_item = {}
                        pool_item['mean'] = dpg.add_line_series([], [], label=f'Mean L{i}', parent='y_axis_1')
                        pool_item['instant'] = dpg.add_line_series([], [], label=f'Inst L{i}', parent='y_axis_1')
                        pool_item['r'] = dpg.add_line_series([], [], label=f'R L{i}', parent='y_axis_1')
                        pool_item['g'] = dpg.add_line_series([], [], label=f'G L{i}', parent='y_axis_1')
                        pool_item['b'] = dpg.add_line_series([], [], label=f'B L{i}', parent='y_axis_1')

                        pool_item['lat'] = dpg.add_line_series([], [], label=f'Lat L{i}', parent='y_axis_2')
                        pool_item['ax'] = dpg.add_line_series([], [], label=f'Ax L{i}', parent='y_axis_2')

                        pool_item['rhabdomeres'] = []
                        for r_idx in range(model.R):
                            tag = dpg.add_line_series([], [], label=f'R{r_idx + 1} L{i}', parent='y_axis_1')
                            pool_item['rhabdomeres'].append(tag)

                        self.series_pool.append(pool_item)

                # Tab 2: Dynamics
                with dpg.tab(label='Dynamics'):
                    dpg.add_spacer(height=5)
                    dpg.add_text('Photomechanical Response', color=[255, 150, 100])

                    dpg.add_checkbox(label='Enable Rhabdomeres Actuation',
                                     default_value=self.ctx.renderer.microsaccades_enabled,
                                     callback=lambda s, a: setattr(self.ctx.renderer, 'actuation', a))

                    dpg.add_separator()
                    dpg.add_text('Shader Parameters (EyeDynamics.comp)', color=[100, 200, 255])

                    dpg.add_slider_float(
                        label='Lateral Gain (um)',
                        default_value=float(model.ommatidia.lateral_amplitude[0]),
                        min_value=0.0, max_value=5.0,
                        callback=lambda s, a: setattr(self.ctx.renderer._model.ommatidia, 'lateral_amplitude', a)
                    )
                    dpg.add_slider_float(
                        label='Axial Gain (um)',
                        default_value=float(model.ommatidia.axial_amplitude[0]),
                        min_value=0.0, max_value=5.0,
                        callback=lambda s, a: setattr(self.ctx.renderer._model.ommatidia, 'axial_amplitude', a)
                    )
                    dpg.add_slider_float(
                        label='Tau Fast (s)',
                        default_value=float(model.ommatidia.tau_adapt_fast[0]),
                        min_value=0.001, max_value=0.1,
                        callback=lambda s, a: setattr(self.ctx.renderer._model.ommatidia, 'tau_adapt_fast', a)
                    )
                    dpg.add_slider_float(
                        label='Tau Relaxation (s)',
                        default_value=float(model.ommatidia.tau_relax[0]),
                        min_value=0.01, max_value=0.5,
                        callback=lambda s, a: setattr(self.ctx.renderer._model.ommatidia, 'tau_relax', a)
                    )

                    dpg.add_separator()

                    if dpg.add_button(label='Reset GPU States', width=-1):
                        self._main_thread_queue.append(
                        self.ctx.renderer.eye_buffers['ema_state'].reset)
                        self._main_thread_queue.append(self.ctx.renderer.eye_buffers['omm_dynamic'].reset)

                # Tab 3: Rendering
                with dpg.tab(label='Rendering'):
                    dpg.add_spacer(height=5)
                    dpg.add_text('Display Modes', color=[100, 200, 255])

                    self.ui_tags['view_mode'] = dpg.add_combo(
                        list(DisplayMode.__members__.keys()),
                        label='View Mode',
                        default_value=self.ctx.display_mode.name,
                        callback=self._set_view_mode
                    )

                    self.ui_tags['output_mode'] = dpg.add_radio_button(
                        list(EyeOutput.__members__.keys()),
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

                    self.ui_tags['tiled_mode'] = dpg.add_checkbox(
                        label='Tiled/Voronoi Mode (V)',
                        default_value=self.ctx.renderer.tiled_mode,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'tiled_mode', a)
                    )

                    dpg.add_separator()

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

                    dpg.add_separator()
                    dpg.add_text('Sampling & Noise', color=[100, 200, 255])

                    self.ui_tags['samples'] = dpg.add_slider_int(
                        label='Samples per Rhabdomere',
                        default_value=self.ctx.renderer.nb_samples,
                        min_value=1, max_value=1024,
                        callback=self._update_nb_samples
                    )

                    self.ui_tags['randomness_mode'] = dpg.add_combo(
                        list(RandomnessMode.__members__.keys()),
                        label='Randomness Mode',
                        default_value=self.ctx.renderer.randomness_mode.name,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'randomness_mode', a)
                    )

                    self.ui_tags['sampling_mode'] = dpg.add_combo(
                        list(SamplingMode.__members__.keys()),
                        label='Sampling Mode',
                        default_value=self.ctx.renderer.sampling_mode.name,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'sampling_mode', a)
                    )

                    self.ui_tags['time_dither'] = dpg.add_checkbox(
                        label='Time Dithering (T)',
                        default_value=self.ctx.renderer.time_dithering,
                        callback=lambda s, a: setattr(self.ctx.renderer, 'time_dithering', a)
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

    def toggle_omm_selection(self, omm_idx, multi=False):
        """Toggle selection. If omm_idx is None, clears all."""
        if omm_idx is None:
            self.selected_ommatidia = []
            # reset the slider to 0 if cleared
            if dpg.does_item_exist(self.omm_slider):
                dpg.set_value(self.omm_slider, 0)
        elif not multi:
            self.selected_ommatidia = [omm_idx]
        else:
            if omm_idx in self.selected_ommatidia:
                self.selected_ommatidia.remove(omm_idx)
            else:
                if len(self.selected_ommatidia) < self.max_selected:
                    self.selected_ommatidia.append(omm_idx)

        model = self.ctx.renderer._model
        pad_len = len(self.frame_data)
        for lid in self.selected_ommatidia:
            if lid not in self.omm_histories:
                self.omm_histories[lid] = {
                    'mean': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'instant': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'r': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'g': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'b': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'lat': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'ax': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'rhabdomeres': [collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len) for _ in
                                  range(model.R)]
                }

        if dpg.does_item_exist(self.omm_selection_text):
            val = self.selected_ommatidia if self.selected_ommatidia else "None"
            dpg.set_value(self.omm_selection_text, f"Selected ommatidia: {val}")
        if self.selected_ommatidia and dpg.does_item_exist(self.omm_slider):
            dpg.set_value(self.omm_slider, self.selected_ommatidia[-1])

    def _on_slider_change(self, sender, app_data):
        self.toggle_omm_selection(app_data, multi=False)

    def _apply_styles(self):
        LENS_PALETTE = [
            (255, 255, 255), (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255), (255, 150, 50),
            (150, 50, 255), (50, 255, 150)
        ]

        for i, pool_item in enumerate(self.series_pool):
            color = LENS_PALETTE[i % len(LENS_PALETTE)]

            with dpg.theme() as theme_mean:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(pool_item['mean'], theme_mean)

            with dpg.theme() as theme_inst:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (*color[:3], 150), category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(pool_item['instant'], theme_inst)

            for key, c in [('r', (255, 50, 50)), ('g', (50, 255, 50)), ('b', (50, 50, 255))]:
                with dpg.theme() as t:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, c, category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(pool_item[key], t)

            for key, c in [('lat', (0, 200, 255)), ('ax', (255, 150, 0))]:
                with dpg.theme() as t:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, c, category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(pool_item[key], t)

            for r_idx, rec_series in enumerate(pool_item['rhabdomeres']):
                rec_color = RHAB_COLOURS[r_idx % len(RHAB_COLOURS)]
                with dpg.theme() as t:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, rec_color, category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(rec_series, t)

    # Callbacks

    def _toggle_mouse_lock(self, sender, app_data):
        self.ctx.mouse_captured = app_data

    def _change_output_mode(self, sender, app_data):
        self.ctx.renderer.output_mode = EyeOutput[app_data]
        self.history_intervals[-1][1] = self.current_frame
        self.history_intervals.append([self.current_frame, None, app_data])

    def _set_view_mode(self, sender, app_data):
        self.ctx.display_mode = DisplayMode[app_data]

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

        # Info panel sync
        from insectvision.renderers import Raytracer, Pathtracer

        # Renderer detection
        is_ray_based = isinstance(self.ctx.renderer, Raytracer)
        if isinstance(self.ctx.renderer, Pathtracer):
            renderer_name = "Pathtracer"
        elif is_ray_based:
            renderer_name = "Raytracer"
        else:
            renderer_name = "Rasterizer"

        dpg.set_value('ui_fps_text', f'FPS: {self.ctx.fps:5.1f}')
        dpg.set_value('ui_renderer_text', f'| Renderer: {renderer_name}')

        # Timing: wall vs sim clock
        dpg.set_value('ui_wall_dt_text', f'Wall dt: {self.ctx.wall_dt * 1000.0:6.2f} ms')

        if self.ctx.time_step is not None:
            sim_hz = 1.0 / self.ctx.time_step
            sim_mode = f'Sim: {sim_hz:5.1f} Hz (fixed {self.ctx.time_step * 1000:.1f} ms)'
        else:
            sim_mode = 'Sim: real-time (variable)'
        dpg.set_value('ui_sim_mode_text', f'| {sim_mode}')

        dpg.set_value('ui_sim_total_text', f'Total sim time: {self.ctx.total_time:7.3f} s')

        model = self.ctx.renderer._model
        dpg.configure_item(self.rec_slider, max_value=max(0, model.R - 1))

        # Modes
        view_str = self.ctx.display_mode.name.replace('_', ' ')
        proj_str = self.ctx.renderer.projection_mode.name
        dpg.set_value('ui_view_text', f'View: {view_str}')
        dpg.set_value('ui_proj_text', f'| Proj: {proj_str}')

        # Position
        pos = self.ctx.agent.position
        dpg.set_value('ui_pos_text', f'Pos: [{pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}]')

        # Samples and ommatidia
        nb_om = self.ctx.renderer._model.N
        nb_om_samples = getattr(self.ctx.renderer, 'nb_samples', 1)
        nb_px_samples = getattr(self.ctx.renderer, 'samples_per_pixel', 1)

        dpg.set_value('ui_omm_text', f'Ommatidia: {nb_om:,}')
        dpg.set_value('ui_samples_text', f'| Samples: {nb_om_samples}/om | {nb_px_samples}/px')

        # Scene Stats
        dpg.set_value('ui_tri_text', f'Triangles: {self.ctx.scene.total_triangles:,}')
        dpg.set_value('ui_pts_text', f'| Points: {self.ctx.scene.total_points:,}')

        # Interactive inputs sync

        # Rendering and display modes sync
        dpg.set_value(self.ui_tags['view_mode'], self.ctx.display_mode.name)
        dpg.set_value(self.ui_tags['proj_mode'], self.ctx.renderer.projection_mode.name)
        dpg.set_value(self.ui_tags['tiled_mode'], self.ctx.renderer.tiled_mode)
        dpg.set_value(self.ui_tags['heatmap'], self.ctx.renderer.overlay_enabled)
        dpg.set_value(self.ui_tags['randomness_mode'], self.ctx.renderer.randomness_mode.name)
        dpg.set_value(self.ui_tags['sampling_mode'], self.ctx.renderer.sampling_mode.name)

        if hasattr(self.ctx.renderer, 'time_dithering'):
            dpg.set_value(self.ui_tags['time_dither'], self.ctx.renderer.time_dithering)
        if hasattr(self.ctx.renderer, 'nb_samples'):
            dpg.set_value(self.ui_tags['samples'], self.ctx.renderer.nb_samples)

        # Environment sync
        if self.ctx.scene.sun:
            dpg.set_value(self.ui_tags['sun_control'], self.ctx.sun_control_mode)
            dpg.set_value(self.ui_tags['sun_azimuth'], self.ctx.scene.sun.azimuth)
            dpg.set_value(self.ui_tags['sun_elevation'], self.ctx.scene.sun.elevation)
            dpg.set_value(self.ui_tags['sun_intensity'], self.ctx.scene.sun.intensity)

        # Agent sync
        dpg.set_value(self.ui_tags['pos_x'], pos.x)
        dpg.set_value(self.ui_tags['pos_y'], pos.y)
        dpg.set_value(self.ui_tags['pos_z'], pos.z)

        dpg.set_value(self.ui_tags['rot_y'], self.ctx.agent.yaw)
        dpg.set_value(self.ui_tags['rot_p'], self.ctx.agent.pitch)
        dpg.set_value(self.ui_tags['rot_r'], self.ctx.agent.roll)

        dpg.set_value('ui_mouse_lock', self.ctx.mouse_captured)

    def _update_plot_data(self, visual_output, model, mode):
        """Processes histories and updates all plot slots."""  # TODO: More than 10?

        self.frame_data.append(self.current_frame)
        x = list(self.frame_data)

        dynamic_states = self.ctx.renderer.eye_buffers['omm_dynamic'].read()

        for i, pool in enumerate(self.series_pool):

            # Check if an ommatidium is selected for this slot
            if i < len(self.selected_ommatidia):
                om_id = self.selected_ommatidia[i]

                # Update history data
                if mode == EyeOutput.Raw:
                    start = om_id * model.R
                    group = visual_output.data[start: start + model.R]
                elif mode == EyeOutput.Ommatidium:
                    group = visual_output.per_ommatidium[om_id]
                else:
                    group = visual_output.per_cartridge[om_id]

                hist = self.omm_histories[om_id]
                avg_pixel = np.mean(group, axis=0)

                hist['mean'].append(np.mean(avg_pixel[:3]))
                hist['r'].append(avg_pixel[0])
                hist['g'].append(avg_pixel[1])
                hist['b'].append(avg_pixel[2])
                hist['instant'].append(avg_pixel[3])
                hist['lat'].append(float(dynamic_states[om_id]['curr_lateral_disp']))
                hist['ax'].append(float(dynamic_states[om_id]['curr_axial_disp']))

                for r_idx in range(model.R):
                    hist['rhabdomeres'][r_idx].append(np.mean(group[r_idx, :3]))

                # Refresh lines for this active slot
                self._refresh_series(pool, om_id, hist, x, visible=True)
            else:
                # This slot is unused: hide all lines
                self._refresh_series(pool, None, None, None, visible=False)

        self._update_plot_backgrounds()
        self.current_frame += 1

    def _refresh_series(self, pool, lid, hist, x, visible):

        if not visible:
            # Hide everything and return
            for key, tag in pool.items():
                if key == 'rhabdomeres':
                    for r_tag in tag: dpg.configure_item(r_tag, show=False)
                else:
                    dpg.configure_item(tag, show=False)
            return

        # Fetch UI Toggle States
        show_all_rec = dpg.get_value(self.show_all_rec_toggle)
        show_rgb = dpg.get_value(self.rgb_toggle)
        show_instant = dpg.get_value(self.instant_toggle)

        # Mean is only shown if individual Rhabdomeres and RGB are both off
        show_mean = not show_all_rec and not show_rgb
        dpg.configure_item(pool['mean'], label=f'Mean L{lid}', show=show_mean)
        dpg.set_value(pool['mean'], [x, list(hist['mean'])])

        # Instantaneous is an independent overlay
        dpg.configure_item(pool['instant'], label=f'Inst L{lid}', show=show_instant)
        dpg.set_value(pool['instant'], [x, list(hist['instant'])])

        # RGB lines shown if RGB is on and individual Rhabdomeres is off
        for key in ['r', 'g', 'b']:
            dpg.configure_item(pool[key], label=f'{key.upper()} L{lid}',
                               show=show_rgb and not show_all_rec)
            dpg.set_value(pool[key], [x, list(hist[key])])

        # Individual Rhabdomere lines
        for r_idx, r_tag in enumerate(pool['rhabdomeres']):
            dpg.configure_item(r_tag, label=f'R{r_idx + 1} L{lid}', show=show_all_rec)
            if show_all_rec:
                dpg.set_value(r_tag, [x, list(hist['rhabdomeres'][r_idx])])

        # Actuation plot (always show if ommatidium selected)
        dpg.configure_item(pool['lat'], label=f'Lat L{lid}', show=True)
        dpg.set_value(pool['lat'], [x, list(hist['lat'])])

        dpg.configure_item(pool['ax'], label=f'Ax L{lid}', show=True)
        dpg.set_value(pool['ax'], [x, list(hist['ax'])])

    def _update_plot_backgrounds(self):

        window_start = self.frame_data[0] if self.frame_data else 0

        for layer, y_max in [(self.bg_layer_1, 1.1), (self.bg_layer_2, 5.0)]:
            dpg.delete_item(layer, children_only=True)

            for start_frame, end_frame, mode_name in self.history_intervals:
                effective_end = end_frame if end_frame is not None else self.current_frame

                # Only draw if interval is within the current visible history window
                if effective_end >= window_start:
                    dpg.draw_rectangle(
                        pmin=[start_frame, -2.0],
                        pmax=[effective_end, y_max],
                        fill=self.MODE_COLORS[mode_name],
                        color=(0, 0, 0, 0),
                        parent=layer
                    )

        # Auto-scroll X axis
        if len(self.frame_data) > 1:
            x_min, x_max = self.frame_data[0], self.frame_data[-1]
            dpg.set_axis_limits('x_axis_1', x_min, x_max)
            dpg.set_axis_limits('x_axis_2', x_min, x_max)

    # Main render
    def render(self, visual_output: 'VisualOutput'):
        if not self._initialised:
            self._setup_dpg()
            # Initial sync of selection from renderer to dashboard
            initial = self.ctx.renderer.selected_ommatidia
            if initial:
                for lid in initial: self.toggle_omm_selection(lid, multi=True)

        if not dpg.is_dearpygui_running():
            return False

        # Main thread tasks (GL calls)
        while self._main_thread_queue:
            self._main_thread_queue.pop(0)()

        # Sync UI state (renderer -> dashboard)
        self._sync_ui_state()

        model = self.ctx.renderer._model
        mode = self.ctx.renderer.output_mode

        # maintain a mapping of what the shader needs to highlight
        shader_selection = np.full(10, -1, dtype=np.int32)
        if self.selected_ommatidia:
            for idx, l_id in enumerate(self.selected_ommatidia[:10]):
                if l_id >= model.N: continue

                if mode == EyeOutput.Raw:
                    rec_id = dpg.get_value(self.rec_slider)
                    shader_selection[idx] = (l_id * model.R) + rec_id
                else:
                    shader_selection[idx] = l_id

        # Push updated selection to renderer
        self.ctx.renderer.selected_ommatidia = shader_selection

        # Update plots (if tab is visible)
        if dpg.is_item_visible('tab_plots') and visual_output is not None:
            self._update_plot_data(visual_output, model, mode)

        dpg.render_dearpygui_frame()
        return True

    def free(self):
        if self._initialised:
            dpg.destroy_context()
            self._initialised = False