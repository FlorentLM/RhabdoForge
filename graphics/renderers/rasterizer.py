from typing import Tuple
import OpenGL
OpenGL.ERROR_CHECKING = False

from OpenGL.GL import *
from pyglm import glm

from geometry.compound_eyes import CompoundEye
from graphics.camera import Camera
from graphics.renderers.panoramic import PanoramicEye
from graphics.renderers.base import EyeRendererBase
from graphics.scene import Scene, MeshAsset
from graphics.utils import load_shaders, load_texture, ShaderProgram


class CubemapFBO:
    def __init__(self, resolution=256):
        self.resolution = resolution

        # Create Framebuffer object
        self.fbo_id = glGenFramebuffers(1)

        # Create the Color cubemap texture
        self.color_texture_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_CUBE_MAP, self.color_texture_id)
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
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X, self.color_texture_id, 0)
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
        glDeleteTextures(1, [self.color_texture_id])
        glDeleteRenderbuffers(1, [self.depth_buffer_id])


class RasterMesh:
    """ A renderable (rasterization) representation of a MeshAsset. Holds OpenGL resources. """

    def __init__(self, asset: MeshAsset, vert_shader_path, frag_shader_path):
        self.source_asset_id = asset.id
        self.data = asset.vertex_data
        self.draw_count = len(self.data) // 5

        self.shaders = load_shaders(vert_shader_path, frag_shader_path)
        self.texture = load_texture(asset.texture_path)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.data.nbytes, self.data, GL_STATIC_DRAW)

        pos_loc = glGetAttribLocation(self.shaders, "pos")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 5 * self.data.itemsize, ctypes.c_void_p(0))

        tex_loc = glGetAttribLocation(self.shaders, "vertTexCoord")
        glEnableVertexAttribArray(tex_loc)
        glVertexAttribPointer(tex_loc, 2, GL_FLOAT, GL_FALSE, 5 * self.data.itemsize,
                              ctypes.c_void_p(3 * self.data.itemsize))

        glBindVertexArray(0)

    def free(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteProgram(self.shaders)
        glDeleteTextures(1, [self.texture])


class RasterInstance:
    """ An instance wrapper for the rasterizer, holding a RasterMesh """

    def __init__(self, raster_mesh: RasterMesh, transform: glm.mat4):
        self.asset = raster_mesh
        self.transform = transform


class RasterSceneBaker:
    """
    Manages the conversion of a logical Scene into GPU-ready data for the rasterizer.
    This involves creating and caching OpenGL vertex arrays (VAOs) for each unique mesh asset.
    """
    def __init__(self, scene: Scene, vert_shader_path: str, frag_shader_path: str):
        self.scene = scene
        self.vert_shader_path = vert_shader_path
        self.frag_shader_path = frag_shader_path
        self._raster_mesh_cache = {}

    def get_renderables(self):
        """
        Provides a list of all instances in a format ready for the rasterizer.
        Creates and caches OpenGL resources on the fly.
        """

        # TODO: This should not recreate the list every time, only on scene change

        renderables = []
        for logical_inst in self.scene.instances:
            asset = logical_inst.asset
            if not isinstance(asset, MeshAsset):
                continue

            if asset.id not in self._raster_mesh_cache:
                print(f"Creating raster representation for asset '{asset.name}'...")
                self._raster_mesh_cache[asset.id] = RasterMesh(asset, self.vert_shader_path, self.frag_shader_path)

            raster_mesh = self._raster_mesh_cache[asset.id]
            renderables.append(RasterInstance(raster_mesh, logical_inst.transform))
        return renderables

    def free(self):
        """ Frees all cached OpenGL resources """

        for raster_mesh in self._raster_mesh_cache.values():
            raster_mesh.free()
        self._raster_mesh_cache.clear()

        print("RasterSceneData resources freed.")


class EyeRendererRaster(EyeRendererBase):

    def __init__(self, eye_model: CompoundEye, scene: Scene, window_size: Tuple[int, int] = (1280, 720), time_dithering: bool = False, nb_samples: int = 256, cubemap_res: int = 512):
        super().__init__(eye_model, window_size=window_size, time_dithering=time_dithering, nb_samples=nb_samples)

        self.scene = scene  # just for convenience
        self._scene_baked = RasterSceneBaker(scene, 'shaders/base.vert', 'shaders/base.frag')

        self._rasterizer_shader = ShaderProgram(comp_path='shaders/ommatidia_rasterizer.comp')

        self._cubemap_fbo = CubemapFBO(resolution=cubemap_res)
        self._cubemap_id = self._cubemap_fbo.color_texture_id

        # simple 90 degrees view projection matrix for each cube face
        self._proj_mat = Camera(fov=90.0, ratio=1.0).projection

        self._pano_view = PanoramicEye()

        # Get renderables once
        self.renderables = self._scene_baked.get_renderables()

    @property
    def samples_per_ommatidium(self):
        return self._samples_per_ommatidium

    @samples_per_ommatidium.setter
    def samples_per_ommatidium(self, value):
        self._samples_per_ommatidium = int(min(32768, max(1, value)))
        # no buffers to reallocate

    def _render_instance(self, instance: RasterInstance, view_matrix, projection_matrix):
        """ Renders a single RasterInstance """

        mesh = instance.asset

        glUseProgram(mesh.shaders)
        glBindVertexArray(mesh.vao)

        camera_matrix = projection_matrix * view_matrix
        glUniformMatrix4fv(glGetUniformLocation(mesh.shaders, "camera"), 1, False, glm.value_ptr(camera_matrix))
        glUniformMatrix4fv(glGetUniformLocation(mesh.shaders, "model"), 1, False, glm.value_ptr(instance.transform))

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, mesh.texture)
        glDrawArrays(GL_TRIANGLES, 0, mesh.draw_count)

        # TODO: these may be skipped when rendering several instances of the same thing
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindVertexArray(0)
        glUseProgram(0)

    def _render_to_cubemap(self, camera):

        self._cubemap_fbo.bind()

        # look-at directions and 'up' vectors for each face must correspond to the OpenGL cubemap coordinate system:
        #  - GL_TEXTURE_CUBE_MAP_POSITIVE_X  ->  Right
        #  - GL_TEXTURE_CUBE_MAP_NEGATIVE_X  ->  Left
        #  - GL_TEXTURE_CUBE_MAP_POSITIVE_Y  ->  Up
        #  - GL_TEXTURE_CUBE_MAP_NEGATIVE_Y  ->  Down
        #  - GL_TEXTURE_CUBE_MAP_POSITIVE_Z  ->  Back
        #  - GL_TEXTURE_CUBE_MAP_NEGATIVE_Z  ->  Front
        #
        # Note: the camera's local vectors are used to maintain its orientation (roll)

        look_dirs = [
            camera.right,  # For +X face (index 0), we look to the camera's right
            camera.left,  # For -X face (index 1), we look to the camera's left
            camera.up,  # For +Y face (index 2), we look up
            camera.down,  # For -Y face (index 3), we look down
            camera.backward,  # For +Z face (index 4), we look backward
            camera.forward,  # For -Z face (index 5), we look forward
        ]

        # The 'up' vectors for each look-at direction
        ups = [
            camera.down,  # Up for looking right/left is camera's down
            camera.down,
            camera.backward,  # Up for looking up is camera's backward
            camera.forward,  # Up for looking down is camera's forward
            camera.down,  # Up for looking backward/forward is camera's down
            camera.down,
        ]

        for i in range(6):
            # Attach the correct face of the cubemap texture for rendering
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   self._cubemap_fbo.color_texture_id, 0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Generate the specific view matrix for this face
            view = glm.lookAt(
                glm.vec3(camera.position),
                glm.vec3(camera.position + look_dirs[i]),  # target point is eye + direction
                glm.vec3(ups[i])
            )

            # Draw skybox first into the cubemap face
            if self._scene_baked.scene.skybox and self._scene_baked.scene.skybox_texture_id is not None:
                self._scene_baked.scene.skybox.draw(self._proj_mat, view, self._scene_baked.scene.skybox_texture_id)

            for instance in self.renderables:
                self._render_instance(instance, view, self._proj_mat)

        self._cubemap_fbo.unbind()

        # Regenerate mipmaps after rendering to the cubemap
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._cubemap_fbo.color_texture_id)
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

        # Restore the viewport to the main window's dimensions
        glViewport(0, 0, self.w, self.h)

    def _sample_cubemap(self):

        self._rasterizer_shader.use()

        # Set uniforms for the data pass
        glUniform1i(self._rasterizer_shader.get_loc('u_num_ommatidia'), self.num_ommatidia)
        glUniform1i(self._rasterizer_shader.get_loc('u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1f(self._rasterizer_shader.get_loc('u_time'), float(self._time_counter))

        # Bind input cubemap (texture unit 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._cubemap_id)
        glUniform1i(self._rasterizer_shader.get_loc('u_scene_cubemap'), 0)

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

    def _compute_colors(self, camera):
        """ The core ommatidia rendering logic """

        # Pass 1: Render to cubemap
        self._render_to_cubemap(camera)

        # Pass 2: Sample from cubemap
        self._sample_cubemap()

        # Unbind resources
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    def get_ommatidia_data(self, camera, to_cpu=False):
        """ Generates a cubemap and then computes ommatidia data from it """

        self._compute_colors(camera)

        if self._time_dithering:
            self._time_counter += 1

        if to_cpu:
            self._fetch_to_cpu()

        return self.cpu_read_buffer

    def draw(self, view_mode: str, camera, tiled_mode: bool = False):
        """ Renders one of the rasterizer's supported views to the screen """

        if view_mode == 'compound_eye':
            # This calls the draw() method in EyeRendererBase for Voronoi rendering
            super().draw(tiled_mode=tiled_mode)

        elif view_mode == 'panoramic':
            self._pano_view.draw(self._cubemap_id)

        elif view_mode == 'standard_3d':
            if self._scene_baked.scene.skybox and self._scene_baked.scene.skybox_texture_id is not None:
                self._scene_baked.scene.skybox.draw(camera.projection, camera.view,
                                                    self._scene_baked.scene.skybox_texture_id)
            renderables = self._scene_baked.get_renderables()
            for instance in renderables:
                self._render_instance(instance, camera.view, camera.projection)

    def free(self):
        self._scene_baked.free()

        self._rasterizer_shader.free()
        self._cubemap_fbo.free()
        self._pano_view.free()

        super().free()
