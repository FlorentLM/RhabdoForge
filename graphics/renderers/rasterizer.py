import OpenGL

OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
import numpy as np
from pyglm import glm

from geometry.compound_eyes import CompoundEye
from graphics.agent import Agent
from graphics.renderers.panoramic import PanoramicEye
from graphics.renderers.base import EyeRendererBase
from graphics.scene import Scene, MeshAsset, PointsAsset
from graphics.utils import load_shaders, load_texture, ShaderProgram, ViewMode


class CubemapFBO:
    def __init__(self, resolution=256):
        self.resolution = resolution

        # Create FBO and Color cubemap texture
        self.fbo_id = glGenFramebuffers(1)
        self.texture_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_id)
        for i in range(6):
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA8,
                         self.resolution, self.resolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        # Generate mipmap storage
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

        # Create the Depth Renderbuffer
        self.depth_buffer_id = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer_id)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.resolution, self.resolution)

        # Attach textures/buffers to the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)

        # Attach just one face initially to make the FBO 'complete'
        # The render loop will correctly attach the other faces as needed
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X, self.texture_id, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_buffer_id)

        # Check if the FBO is complete
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer is not complete: {status}")

        # Unbind everything
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self):
        glViewport(0, 0, self.resolution, self.resolution)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def free(self):
        glDeleteFramebuffers(1, [self.fbo_id])
        glDeleteTextures(1, [self.texture_id])
        glDeleteRenderbuffers(1, [self.depth_buffer_id])


class RasterMesh:
    """
    A renderable (rasterization) representation of a mesh asset
    Holds OpenGL resources
    """

    def __init__(self, asset: MeshAsset, vert_shader_path, frag_shader_path):
        self.source_asset_id = asset.id

        self.vertices = asset.vertices
        self.indices = asset.indices
        self.draw_count = self.indices.size

        self.shaders = ShaderProgram(vert_shader_path, frag_shader_path)
        self.texture = load_texture(asset.texture_path)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Vertex Buffer Object (VBO) for vertex data
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Element Buffer Object (EBO) for index data
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # Vertex attribute pointers (stride and offsets are correct for the new 'vertices' array)
        vertex_size_bytes = self.vertices.itemsize * 5

        pos_loc = glGetAttribLocation(self.shaders.program_id, "position")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, False, vertex_size_bytes, ctypes.c_void_p(0))

        tex_loc = glGetAttribLocation(self.shaders.program_id, "vertTexCoord")
        glEnableVertexAttribArray(tex_loc)
        glVertexAttribPointer(tex_loc, 2, GL_FLOAT, False, vertex_size_bytes,
                              ctypes.c_void_p(self.vertices.itemsize * 3))

        glBindVertexArray(0)

    def free(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        self.shaders.free()
        glDeleteTextures(1, [self.texture])


class RasterPoints:
    """
    A renderable (rasterization) representation of a point cloud asset
    Holds OpenGL resources
    """

    def __init__(self, asset: PointsAsset, vert_shader_path: str, frag_shader_path: str):
        self.source_asset_id = asset.id
        self.draw_count = asset.num_points

        self.shaders = ShaderProgram(vert_shader_path, frag_shader_path)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        # Interleave positions, colors, and radii (x, y, z, r, g, b, radius)
        packed_data = np.hstack([asset.points, asset.colors, asset.radii.reshape(-1, 1)]).astype(np.float32)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, packed_data.nbytes, packed_data, GL_STATIC_DRAW)

        stride = packed_data.itemsize * 7  # 3 for pos, 3 for color, 1 for radius

        # Vertex attribute for position
        pos_loc = glGetAttribLocation(self.shaders.program_id, "position")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))

        # Vertex attribute for color
        color_loc = glGetAttribLocation(self.shaders.program_id, "color")
        glEnableVertexAttribArray(color_loc)
        glVertexAttribPointer(color_loc, 3, GL_FLOAT, False, stride, ctypes.c_void_p(packed_data.itemsize * 3))

        # Vertex attribute for radius
        radius_loc = glGetAttribLocation(self.shaders.program_id, "radius")
        glEnableVertexAttribArray(radius_loc)
        glVertexAttribPointer(radius_loc, 1, GL_FLOAT, False, stride, ctypes.c_void_p(packed_data.itemsize * 6))

        glBindVertexArray(0)

    def free(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        self.shaders.free()


class RasterInstance:
    """ An instance wrapper for the rasterizer, holding a baked asset and transform. """
    def __init__(self, asset: RasterMesh | RasterPoints, transform: glm.mat4, properties: dict):
        self.asset = asset
        self.transform = transform
        self.properties = properties


class RasterSceneBaker:
    """
    Creates and caches OpenGL vertex arrays (VAOs) for each unique asset
    """

    def __init__(self, scene: Scene):
        self.scene = scene
        self._raster_asset_cache = {}

    def get_renderables(self):
        """
        Provides a list of all instances in a format ready for the rasterizer.
        Creates and caches OpenGL resources on the fly.
        """
        renderables = []

        for instance in self.scene.instances:
            asset = instance.asset

            # Skip any unknown assets
            if not isinstance(asset, (MeshAsset, PointsAsset)):
                continue

            # Create the raster wrapper if it doesn't exist yet
            if asset.id not in self._raster_asset_cache:

                if isinstance(asset, MeshAsset):
                    self._raster_asset_cache[asset.id] = RasterMesh(
                        asset, 'shaders/mesh.vert', 'shaders/mesh.frag'
                    )

                elif isinstance(asset, PointsAsset):
                    self._raster_asset_cache[asset.id] = RasterPoints(
                        asset, 'shaders/pointclouds.vert', 'shaders/pointclouds.frag'
                    )

            # Create a renderable instance with the cached raster asset and the instance's transform
            raster_asset = self._raster_asset_cache[asset.id]
            renderables.append(RasterInstance(raster_asset, instance.transform, instance.properties))

        return renderables

    def free(self):
        """ Frees all cached OpenGL resources """

        for raster_asset in self._raster_asset_cache.values():
            raster_asset.free()

        self._raster_asset_cache.clear()


class EyeRendererRaster(EyeRendererBase):

    def __init__(self, eye_model: CompoundEye, scene: Scene,
                 time_dithering: bool = False,
                 nb_samples: int = 256,
                 cubemap_res: int = 512,
                 batch_size: int = 1):

        self.scene = scene  # just for convenience
        self._scene_baked = RasterSceneBaker(scene)

        self._cubemap_fbo = CubemapFBO(resolution=cubemap_res)

        # call super after creating the scene for correct VRAM usage computation
        super().__init__(eye_model, time_dithering=time_dithering, nb_samples=nb_samples, batch_size=batch_size)

        self._rasterizer_shader = ShaderProgram(comp_path='shaders/ommatidia_rasterizer.comp')

        # simple 90 degrees view projection matrix for each cube face
        self._proj_mat = Agent(fov=90.0, ratio=1.0).projection

        self._raster_panoramic = PanoramicEye()

    @property
    def samples_per_ommatidium(self):
        return self._samples_per_ommatidium

    @samples_per_ommatidium.setter
    def samples_per_ommatidium(self, value):
        self._samples_per_ommatidium = int(min(32768, max(1, value)))
        # no buffers to reallocate

    def estimate_vram_usage(self) -> float:
        """ Override base method to provide a VRAM estimate for the rasterizer """

        # This is kinda hard to calculate precisely without inspecting every texture and mesh
        # so we assume a baseline and add the cubemap FBO size

        # Cubemap is RGBA8 (4 bytes) * 6 faces
        cubemap_mb = (self._cubemap_fbo.resolution ** 2 * 4 * 6) / (1024 * 1024)

        # A rough conservative guess for shaders, VAOs, VBOs etc
        scene_assets_mb = 50.0
        return cubemap_mb + scene_assets_mb

    def _render_instance(self, instance: RasterInstance, view_matrix, projection_matrix):
        """ Renders a single RasterInstance """

        asset = instance.asset

        asset.shaders.use()
        glBindVertexArray(asset.vao)

        cam_mat = projection_matrix * view_matrix
        glUniformMatrix4fv(asset.shaders.get_loc("camera"), 1, False, glm.value_ptr(cam_mat))
        glUniformMatrix4fv(asset.shaders.get_loc("model"), 1, False, glm.value_ptr(instance.transform))

        # TODO: unbinding may be skipped when rendering several instances of the same thing

        if isinstance(asset, RasterMesh):

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, asset.texture)
            glDrawElements(GL_TRIANGLES, asset.draw_count, GL_UNSIGNED_INT, None)

            glBindTexture(GL_TEXTURE_2D, 0)

        elif isinstance(asset, RasterPoints):
            glEnable(GL_PROGRAM_POINT_SIZE)

            # Global multiplier to convert the world-unit radius to an visible pixel size
            pixel_mult = 25.0
            radius_scale = instance.properties.get('radius_scale', 1.0) * pixel_mult

            glUniform1f(asset.shaders.get_loc("radius_scale"), radius_scale)

            glDrawArrays(GL_POINTS, 0, asset.draw_count)

            glDisable(GL_PROGRAM_POINT_SIZE)

        glBindVertexArray(0)

        asset.shaders.stop()

    def _render_to_cubemap(self, agent):
        """ Pass 1: renders to the cubemap """

        main_viewport = glGetIntegerv(GL_VIEWPORT)

        self._cubemap_fbo.bind()

        # look-at directions and 'up' vectors for each face must correspond to the OpenGL cubemap coordinate system:
        #  - GL_TEXTURE_CUBE_MAP_POSITIVE_X  ->  Right
        #  - GL_TEXTURE_CUBE_MAP_NEGATIVE_X  ->  Left
        #  - GL_TEXTURE_CUBE_MAP_POSITIVE_Y  ->  Up
        #  - GL_TEXTURE_CUBE_MAP_NEGATIVE_Y  ->  Down
        #  - GL_TEXTURE_CUBE_MAP_POSITIVE_Z  ->  Back
        #  - GL_TEXTURE_CUBE_MAP_NEGATIVE_Z  ->  Front
        #
        # Note: the agent's local vectors are used to maintain its orientation (roll)

        look_dirs = [
            agent.right,  # For +X face (index 0), we look to the agent's right
            agent.left,  # For -X face (index 1), we look to the agent's left
            agent.up,  # For +Y face (index 2), we look up
            agent.down,  # For -Y face (index 3), we look down
            agent.backward,  # For +Z face (index 4), we look backward
            agent.forward,  # For -Z face (index 5), we look forward
        ]

        # The 'up' vectors for each look-at direction
        ups = [
            agent.down,  # Up for looking right/left is agent's down
            agent.down,
            agent.backward,  # Up for looking up is agent's backward
            agent.forward,  # Up for looking down is agent's forward
            agent.down,  # Up for looking backward/forward is agent's down
            agent.down,
        ]

        renderables = self._scene_baked.get_renderables()

        bg = self.scene.background_color
        glClearColor(bg[0], bg[1], bg[2], 1.0)

        for i in range(6):
            # Attach the correct face of the cubemap texture for rendering
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   self._cubemap_fbo.texture_id, 0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Generate the specific view matrix for this face
            view = glm.lookAt(
                glm.vec3(agent.position),
                glm.vec3(agent.position + look_dirs[i]),  # target point is eye + direction
                glm.vec3(ups[i])
            )

            # Draw skybox first into the cubemap face
            if self._scene_baked.scene.skybox is not None:
                self._scene_baked.scene.skybox.draw(self._proj_mat, view)

            for instance in renderables:
                self._render_instance(instance, view, self._proj_mat)

        self._cubemap_fbo.unbind()

        # Regenerate mipmaps after rendering to the cubemap
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._cubemap_fbo.texture_id)
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

        # Restore the viewport to the main window's dimensions
        glViewport(main_viewport[0], main_viewport[1], main_viewport[2], main_viewport[3])

    def _sample_cubemap(self):
        """ Pass 2: samples the cubemap """

        self._rasterizer_shader.use()

        # Set uniforms for the data pass
        glUniform1i(self._rasterizer_shader.get_loc('nb_ommatidia'), self.num_ommatidia)
        glUniform1i(self._rasterizer_shader.get_loc('nb_samples'), self.samples_per_ommatidium)
        glUniform1f(self._rasterizer_shader.get_loc('time'), float(self._time_counter))

        # Write into the history buffer circularly
        frame_offset = self._current_frame_index % self._batch_size
        glUniform1i(self._rasterizer_shader.get_loc('frame_index'), frame_offset)

        # Bind input cubemap (texture unit 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._cubemap_fbo.texture_id)
        glUniform1i(self._rasterizer_shader.get_loc('scene_cubemap'), 0)

        # Bind directions SSBO to binding point 0 (for reading)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        # Bind colors SSBO to binding point 1 (for writing)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        # Dispatch the compute shader
        # Divide the total number of ommatidia by the workgroup size (64)
        work_groups_x = (self.num_ommatidia + 63) // 64
        glDispatchCompute(work_groups_x, 1, 1)

        # Wait for compute shader to finish writing to the SSBO
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self._rasterizer_shader.stop()

    def _compute_colors(self, agent):
        """ The core ommatidia rendering logic """

        # Pass 1: Render to cubemap
        self._render_to_cubemap(agent)

        # Pass 2: Sample from cubemap
        self._sample_cubemap()

        # Unbind resources
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    def draw(self, view_mode: ViewMode, point_of_view: Agent, agent: Agent):
        """ Renders one of the rasterizer's supported views to the screen """

        if view_mode == ViewMode.compound_eye:
            self._draw_voronoi()

        elif view_mode == ViewMode.panoramic:
            self._raster_panoramic.draw(self._cubemap_fbo.texture_id)

        elif view_mode == ViewMode.perspective or view_mode == ViewMode.third_person:

            if self._scene_baked.scene.skybox is not None:
                self._scene_baked.scene.skybox.draw(point_of_view.projection, point_of_view.view)

            renderables = self._scene_baked.get_renderables()

            for instance in renderables:
                self._render_instance(instance, point_of_view.view, point_of_view.projection)

    def free(self):
        self._scene_baked.free()

        self._rasterizer_shader.free()
        self._cubemap_fbo.free()
        self._raster_panoramic.free()

        super().free()