import dearpygui.dearpygui as dpg
import numpy as np
import collections

from pyglm import glm
from insectvision.utils import EyeOutput, OmmatidiaProjection, Colormap, DisplayMode


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

        self.selected_lenses = [0]
        self.max_selected = 10
        self.lens_histories = {}

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
        self.rec_data_buffers = [collections.deque(maxlen=self.plot_history_len) for _ in range(ra.receptors_per_lens)]

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
                    dpg.add_text('Ommatidium / Receptor Selection', color=[100, 255, 100])
                    dpg.add_text('Click in viewport (when mouse free) to pick. Shift+click for multi.',
                                 color=[200, 200, 200])

                    self.omm_selection_text = dpg.add_text("Selected Lenses: [0]")
                    self.omm_slider = dpg.add_slider_int(label="Primary Lens ID (or type)", default_value=0,
                                                         max_value=ra.lens_count - 1, callback=self._on_slider_change)
                    dpg.add_button(label="Clear Selection",
                                   callback=lambda: self.toggle_lens_selection(dpg.get_value(self.omm_slider),
                                                                               multi=False))

                    dpg.add_separator()
                    self.show_all_rec_toggle = dpg.add_checkbox(
                        label=f'Show individual receptors (R1-R{ra.receptors_per_lens})', default_value=False)
                    self.rec_slider = dpg.add_slider_int(label='Probed Receptor ID', default_value=0,
                                                         max_value=max(0, ra.receptors_per_lens - 1),
                                                         show=(ra.receptors_per_lens > 1))

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

                        pool_item['receptors'] = []
                        for r_idx in range(ra.receptors_per_lens):
                            tag = dpg.add_line_series([], [], label=f'R{r_idx + 1} L{i}', parent='y_axis_1')
                            pool_item['receptors'].append(tag)

                        self.series_pool.append(pool_item)

                    self.toggle_lens_selection(0, multi=False)

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
                        self.ctx.renderer.eye_buffers['ema_state'].reset)
                        self._main_thread_queue.append(self.ctx.renderer.eye_buffers['lens_dynamic'].reset)

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

    def toggle_lens_selection(self, lens_id, multi=False):
        if not multi:
            self.selected_lenses = [lens_id]
        else:
            if lens_id in self.selected_lenses:
                self.selected_lenses.remove(lens_id)
                if not self.selected_lenses:
                    self.selected_lenses = [0]
            else:
                if len(self.selected_lenses) < self.max_selected:
                    self.selected_lenses.append(lens_id)

        ra = self.ctx.renderer._ra
        pad_len = len(self.frame_data)
        for lid in self.selected_lenses:
            if lid not in self.lens_histories:
                self.lens_histories[lid] = {
                    'mean': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'instant': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'r': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'g': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'b': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'lat': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'ax': collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len),
                    'receptors': [collections.deque([np.nan] * pad_len, maxlen=self.plot_history_len) for _ in
                                  range(ra.receptors_per_lens)]
                }

        if dpg.does_item_exist(self.omm_selection_text):
            dpg.set_value(self.omm_selection_text, f"Selected Lenses: {self.selected_lenses}")
        if self.selected_lenses and dpg.does_item_exist(self.omm_slider):
            dpg.set_value(self.omm_slider, self.selected_lenses[-1])

    def _on_slider_change(self, sender, app_data):
        self.toggle_lens_selection(app_data, multi=False)

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

            for r_idx, rec_series in enumerate(pool_item['receptors']):
                rec_color = self.REC_PALETTE[r_idx % len(self.REC_PALETTE)]
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

        if self.ctx.fixed_sim_dt is not None:
            sim_hz = 1.0 / self.ctx.fixed_sim_dt
            sim_mode = f'Sim: {sim_hz:5.1f} Hz (fixed {self.ctx.fixed_sim_dt * 1000:.1f} ms)'
        else:
            sim_mode = 'Sim: real-time (variable)'
        dpg.set_value('ui_sim_mode_text', f'| {sim_mode}')

        dpg.set_value('ui_sim_total_text', f'Total sim time: {self.ctx.total_sim_time:7.3f} s')

        # Modes
        view_str = self.ctx.display_mode.name.replace('_', ' ')
        proj_str = self.ctx.renderer.projection_mode.name
        dpg.set_value('ui_view_text', f'View: {view_str}')
        dpg.set_value('ui_proj_text', f'| Proj: {proj_str}')

        # Position
        pos = self.ctx.agent.position
        dpg.set_value('ui_pos_text', f'Pos: [{pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}]')

        # Samples and ommatidia
        nb_om = self.ctx.renderer._ra.lens_count
        nb_om_samples = getattr(self.ctx.renderer, 'nb_samples', 1)
        nb_px_samples = getattr(self.ctx.renderer, 'samples_per_pixel', 1)

        dpg.set_value('ui_omm_text', f'Ommatidia: {nb_om:,}')
        dpg.set_value('ui_samples_text', f'| Samples: {nb_om_samples}/om | {nb_px_samples}/px')

        # Scene Stats
        dpg.set_value('ui_tri_text', f'Triangles: {self.ctx.scene.total_triangles:,}')
        dpg.set_value('ui_pts_text', f'| Points: {self.ctx.scene.total_points:,}')

        # Interactive inputs sync

        # Rendering sync
        dpg.set_value(self.ui_tags['view_mode'], self.ctx.display_mode.name)
        dpg.set_value(self.ui_tags['proj_mode'], self.ctx.renderer.projection_mode.name)
        dpg.set_value(self.ui_tags['tiled_mode'], self.ctx.renderer.tiled_mode)
        dpg.set_value(self.ui_tags['heatmap'], self.ctx.renderer.overlay_enabled)

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
        mode = self.ctx.renderer.output_mode


        shader_selection = np.full(10, -1, dtype=np.int32)
        for idx, l_id in enumerate(self.selected_lenses):
            if mode == EyeOutput.Raw:
                rec_id = dpg.get_value(self.rec_slider)
                shader_selection[idx] = (l_id * ra.receptors_per_lens) + rec_id
            else:
                shader_selection[idx] = l_id

        self.ctx.renderer.selected_lenses = shader_selection

        if is_plotting and visual_output is not None:
            self.frame_data.append(self.current_frame)

            for lid in self.selected_lenses:
                if lid >= ra.lens_count:
                    continue

                if mode == EyeOutput.Raw:
                    start, end = lid * ra.receptors_per_lens, (lid + 1) * ra.receptors_per_lens
                    group = visual_output.receptors[start:end]
                elif mode == EyeOutput.Ommatidium:
                    group = visual_output.lenses[lid]
                else:
                    group = visual_output.cartridges[lid]

                hist = self.lens_histories[lid]

                avg_pixel = np.mean(group, axis=0)
                hist['mean'].append(np.mean(avg_pixel[:3]))
                hist['r'].append(avg_pixel[0])
                hist['g'].append(avg_pixel[1])
                hist['b'].append(avg_pixel[2])
                hist['instant'].append(avg_pixel[3])

                for r_idx in range(ra.receptors_per_lens):
                    hist['receptors'][r_idx].append(np.mean(group[r_idx, :3]))

                # Actuation buffer readback
                state_data = self.ctx.renderer.eye_buffers['lens_dynamic'].read(start=lid, count=1)
                hist['lat'].append(float(state_data['lateral_um'][0]))
                hist['ax'].append(float(state_data['axial_um'][0]))

            # Prune out data for previously de-selected lenses
            for lid in list(self.lens_histories.keys()):
                if lid not in self.selected_lenses:
                    del self.lens_histories[lid]

            self.current_frame += 1

            # Update plot data
            x = list(self.frame_data)
            show_all_rec = dpg.get_value(self.show_all_rec_toggle)
            show_rgb = dpg.get_value(self.rgb_toggle)
            show_instant = dpg.get_value(self.instant_toggle)

            for i, pool_item in enumerate(self.series_pool):
                if i < len(self.selected_lenses):
                    lid = self.selected_lenses[i]
                    hist = self.lens_histories[lid]

                    dpg.configure_item(pool_item['mean'], label=f'Mean L{lid}', show=not (show_all_rec or show_rgb))
                    dpg.configure_item(pool_item['instant'], label=f'Inst L{lid}', show=show_instant)

                    for key in ['r', 'g', 'b']:
                        dpg.configure_item(pool_item[key], label=f'{key.upper()} L{lid}',
                                           show=show_rgb and not show_all_rec)

                    dpg.configure_item(pool_item['lat'], label=f'Lat L{lid}', show=True)
                    dpg.configure_item(pool_item['ax'], label=f'Ax L{lid}', show=True)

                    dpg.set_value(pool_item['mean'], [x, list(hist['mean'])])
                    dpg.set_value(pool_item['instant'], [x, list(hist['instant'])])
                    dpg.set_value(pool_item['r'], [x, list(hist['r'])])
                    dpg.set_value(pool_item['g'], [x, list(hist['g'])])
                    dpg.set_value(pool_item['b'], [x, list(hist['b'])])
                    dpg.set_value(pool_item['lat'], [x, list(hist['lat'])])
                    dpg.set_value(pool_item['ax'], [x, list(hist['ax'])])

                    for r_idx in range(ra.receptors_per_lens):
                        dpg.configure_item(pool_item['receptors'][r_idx], label=f'R{r_idx + 1} L{lid}',
                                           show=show_all_rec)
                        dpg.set_value(pool_item['receptors'][r_idx], [x, list(hist['receptors'][r_idx])])
                else:
                    for key in pool_item:
                        if key == 'receptors':
                            for rec in pool_item[key]:
                                dpg.configure_item(rec, show=False)
                        else:
                            dpg.configure_item(pool_item[key], show=False)

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