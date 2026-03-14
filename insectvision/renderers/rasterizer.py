import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

import numpy as np
from pyglm import glm

from insectvision.engine.agent import Agent
from insectvision.engine.scene import Scene, Asset, AssetType
from insectvision.engine.lights import DirectionalLight
from insectvision.engine.shader_utils import ShaderProgram
from insectvision.engine.utils import WORLD_UP, WORLD_RIGHT
from insectvision.geometry.compound_eyes import ReceptorArray
from insectvision.interactive.utils import DisplayMode

from .commons import BaseRenderer, BINDING_RECEPTORS, BINDING_COLORS, BINDING_STATE


##

def light_space_matrix(light: DirectionalLight, scene_center=(0.0, 0.0, 0.0), scene_radius: float = 50.0) -> glm.mat4:

    center = glm.vec3(scene_center)
    light_pos = center + light.direction * scene_radius

    dots = np.abs(glm.dot(glm.normalize(light.direction), WORLD_UP))
    up = np.where(dots > 0.999, WORLD_RIGHT, WORLD_UP)

    light_view = glm.lookAt(light_pos, center, up)
    light_proj = glm.ortho(
        -scene_radius, scene_radius,
        -scene_radius, scene_radius,
        0.01, scene_radius * 2.0)

    return light_proj * light_view



class ShadowMapFBO:
    """Depth-only FBO for directional light shadow mapping."""

    def __init__(self, resolution: int = 2048):
        self.resolution = resolution
        self.fbo_id = 0
        self.depth_texture = 0
        self._create()

    def _create(self):

        # Depth texture with hardware PCF comparison
        self.depth_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, self.resolution, self.resolution, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)

        # Outside the shadow frustum = fully lit
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR,
                         np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))

        # FBO (depth-only, no colour attachment)
        self.fbo_id = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, self.depth_texture, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Shadow map FBO incomplete: {status:#x}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def bind(self):
        glViewport(0, 0, self.resolution, self.resolution)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glClear(GL_DEPTH_BUFFER_BIT)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind_texture(self, unit: int = 1):
        """Bind the depth texture for sampling in colour shaders."""
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.depth_texture)

    def free(self):

        if self.fbo_id:
            glDeleteFramebuffers(1, [self.fbo_id])

        if self.depth_texture:
            glDeleteTextures(1, [self.depth_texture])


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


class EquirectangularCubemap:
    """A simple asset to render a cubemap to the screen as a panoramic (equirectangular) view."""

    def __init__(self):
        self._shader = None
        self._vao = None

    @property
    def shader(self):
        if self._shader is None:
            print("Compiling panoramic debug shaders...")
            self._shader = ShaderProgram(vert_path='visualisation/fullscreen.vert', frag_path='visualisation/cubemapSampler.frag')
        return self._shader

    @property
    def vao(self):
        if self._vao is None:
            # A dummy VAO is sufficient as vertices are generated in the vertex shader
            self._vao = glGenVertexArrays(1)
        return self._vao

    def draw(self, cubemap_texture_id):
        """Draws the panoramic view of the given cubemap."""
        self.shader.use()

        # Bind the cubemap texture we want to inspect
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id)
        glUniform1i(self.shader.get_loc("cubemap"), 0)

        # Draw a full-screen triangle
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # Unbind
        glBindVertexArray(0)
        self.shader.stop()

    def free(self):
        """Frees the GPU resources (shader and VAO)."""

        if self._shader:
            self._shader.free()
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
        self._shader = None
        self._vao = None


class RasterMesh:
    """
    A renderable (rasterization) representation of a mesh asset.
    Holds OpenGL resources.
    """

    def __init__(self, asset: Asset, vert_shader_path, frag_shader_path):

        self.source_asset_id = asset.id

        self.vertices = asset.vertices
        self.indices = asset.indices
        self.draw_count = self.indices.size

        self.shaders = ShaderProgram(vert_path=vert_shader_path, frag_path=frag_shader_path)

        self.texture = self._load_texture_from_asset(asset)
        self.has_texture = self.texture != 0
        self.base_color = asset.material.base_color.copy()  # Store base colour for fallback

        positions = self.vertices[:, :3]
        self.normals = self._compute_vertex_normals(positions, self.indices)

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

        # Vertex attribute pointers
        vertex_size_bytes = self.vertices.itemsize * 5
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, vertex_size_bytes, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, False, vertex_size_bytes,
                              ctypes.c_void_p(self.vertices.itemsize * 3))

        self.nbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.nbo)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

        glBindVertexArray(0)

    @staticmethod
    def _compute_vertex_normals(positions, faces):

        normals = np.zeros_like(positions, dtype=np.float32)
        flat_faces = faces.ravel().reshape(-1, 3) if faces.ndim == 1 else faces

        v0 = positions[flat_faces[:, 0]]
        v1 = positions[flat_faces[:, 1]]
        v2 = positions[flat_faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32)
        for i in range(3):
            np.add.at(normals, flat_faces[:, i], face_normals)

        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(lengths, 1e-8)
        return normals.astype(np.float32)

    def _load_texture_from_asset(self, asset):

        img = asset.texture_image
        if img is None:
            return 0

        # Convert to RGBA just to be sure
        img = img.convert("RGBA")
        img_data = img.tobytes()
        width, height = img.size

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB_ALPHA, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        return tex_id

    def free(self):

        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.nbo])
        glDeleteBuffers(1, [self.ebo])

        self.shaders.free()

        if self.texture != 0:
            glDeleteTextures(1, [self.texture])


class RasterPoints:
    """
    A renderable (rasterization) representation of a point cloud asset.
    Holds OpenGL resources.
    """

    def __init__(self, asset: Asset, vert_shader_path: str, frag_shader_path: str):

        self.source_asset_id = asset.id
        self.draw_count = asset._nb_points

        self.shaders = ShaderProgram(vert_path=vert_shader_path, frag_path=frag_shader_path)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        # Interleave positions, colours, and radii (x, y, z, r, g, b, radius)
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
    """Instance wrapper for the rasterizer, holding a baked asset and transform."""
    def __init__(self, asset: RasterMesh | RasterPoints, transform: glm.mat4, properties: dict):
        self.asset = asset
        self.transform = transform
        self.properties = properties


class RasterSceneBaker:
    """
    Creates and caches OpenGL vertex arrays (VAOs) for each unique asset in the scene.
    """

    def __init__(self, scene: Scene, enable_shadows: bool = False):

        self.scene = scene

        self._raster_asset_cache = {}

        self._shadow_mesh_shader = None
        self._shadow_points_shader = None

        self._bake_all_assets()

        if enable_shadows:
            self._compile_shadow_shaders()

    def _bake_all_assets(self):
        print("Baking rasterizer assets...")

        for asset in self.scene.assets.values():

            if asset.id in self._raster_asset_cache:
                continue

            if asset.asset_type == AssetType.Mesh:
                self._raster_asset_cache[asset.id] = RasterMesh(
                    asset, 'meshRaster.vert', 'meshRaster.frag'
                )

            elif asset.asset_type == AssetType.Points:
                self._raster_asset_cache[asset.id] = RasterPoints(
                    asset, 'pointcloudRaster.vert', 'pointcloudRaster.frag'
                )

        print("Rasterizer asset baking complete.")

    def _compile_shadow_shaders(self):
        print("Compiling shadow depth shaders...")

        self._shadow_mesh_shader = ShaderProgram(vert_path='shadowDepthMesh.vert', frag_path='shadowDepth.frag')

        self._shadow_points_shader = ShaderProgram(vert_path='shadowDepthPointcloud.vert', frag_path='shadowDepth.frag')

    def get_renderables(self):
        renderables = []

        for instance in self.scene.instances:
            baked_asset = self._raster_asset_cache.get(instance.asset.id)

            if baked_asset:
                renderables.append(RasterInstance(baked_asset, instance.transform, instance.properties))

        return renderables

    def free(self):
        for raster_asset in self._raster_asset_cache.values():
            raster_asset.free()

        self._raster_asset_cache.clear()

        if self._shadow_mesh_shader:
            self._shadow_mesh_shader.free()

        if self._shadow_points_shader:
            self._shadow_points_shader.free()


class Rasterizer(BaseRenderer):

    SHADOW_TEX_UNIT = 1

    def __init__(self, receptor_array: ReceptorArray, scene: Scene,
                 time_dithering: bool = False,
                 nb_samples: int = 256,
                 quasi_random: bool = False,
                 cubemap_res: int = 512,
                 batch_size: int = 1,
                 enable_direct: bool = False,
                 enable_shadows: bool = False,
                 enable_ambient: bool = False,
                 shadow_resolution: int = 2048,
                 shadow_radius: float = 50.0
                 ):

        self.scene = scene

        # Global lighting controls
        self.enable_direct = enable_direct
        self.enable_shadows = enable_shadows
        self.enable_ambient = enable_ambient
        self.ambient_intensity = 0.3
        self.ambient_color = (0.4, 0.45, 0.5)   # TODO: derive ambient from the cubemap instead

        self._scene_baked = RasterSceneBaker(scene, enable_shadows=self.enable_shadows)
        self._cubemap_fbo = CubemapFBO(resolution=cubemap_res)

        self._shadow_map = None
        self._light_space_matrix = glm.mat4(1.0)
        self._shadow_radius = shadow_radius

        self.shadow_bias = 0.002
        self.shadow_darkness = 0.3
        self.shadow_splat_scale = 1.5

        if self.enable_shadows:
            self._shadow_map = ShadowMapFBO(resolution=shadow_resolution)
            print(f"Shadow mapping enabled ({shadow_resolution}x{shadow_resolution}, radius={shadow_radius})")

        # super *after* creating the scene for correct VRAM usage computation :)
        super().__init__(
            receptor_array,
            time_dithering=time_dithering,
            nb_samples=nb_samples,
            quasi_random=quasi_random,
            batch_size=batch_size,
        )

        self._rasterizer_shader = ShaderProgram(comp_path='ommatidiaRasterizer.comp')

        # simple 90 degrees view projection matrix for each cube face
        self._proj_mat = Agent(fov=90.0, ratio=1.0).projection

        self._raster_panoramic = EquirectangularCubemap()

    @property
    def samples_per_receptor(self):
        return self._samples_per_receptor

    @samples_per_receptor.setter
    def samples_per_receptor(self, value):
        self._samples_per_receptor = int(min(32768, max(1, value)))

    def estimate_vram_usage(self):

        # Cubemap is RGBA8 (4 bytes) * 6 faces
        cubemap_mb = (self._cubemap_fbo.resolution ** 2 * 4 * 6) / (1024 * 1024)

        shadow_mb = 0.0
        if self._shadow_map:
            shadow_mb = (self._shadow_map.resolution ** 2 * 4) / (1024 * 1024)

        # rough conservative guess for shaders, VAOs, VBOs etc
        scene_assets_mb = 50.0
        return cubemap_mb + shadow_mb + scene_assets_mb

    @property
    def primary_directional_light(self):
        for l in self.scene.directional_lights:
            if l.enabled and l.intensity > 0:
                return l
        return None

    def _render_shadow_pass(self, agent):

        light = self.primary_directional_light
        if light is None:
            return

        self._light_space_matrix = light_space_matrix(
            light=light,
            scene_center=(agent.position.x, agent.position.y, agent.position.z),
            scene_radius=self._shadow_radius)

        lsm_ptr = glm.value_ptr(self._light_space_matrix)
        self._shadow_map.bind()

        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(2.0, 4.0)

        renderables = self._scene_baked.get_renderables()

        # Meshes
        mesh_shader = self._scene_baked._shadow_mesh_shader
        mesh_shader.use()

        glUniformMatrix4fv(mesh_shader.get_loc('light_space_matrix'), 1, False, lsm_ptr)
        glUniform1i(mesh_shader.get_loc('is_point_cloud'), 0)

        for inst in renderables:
            if not isinstance(inst.asset, RasterMesh):
                continue
            glUniformMatrix4fv(mesh_shader.get_loc('model'), 1, False,
                               glm.value_ptr(inst.transform))
            glBindVertexArray(inst.asset.vao)
            glDrawElements(GL_TRIANGLES, inst.asset.draw_count, GL_UNSIGNED_INT, None)

        mesh_shader.stop()

        # Point clouds (splats)
        pts_shader = self._scene_baked._shadow_points_shader
        pts_shader.use()

        glUniformMatrix4fv(pts_shader.get_loc('light_space_matrix'), 1, False, lsm_ptr)

        pixel_mult = 25.0
        glEnable(GL_PROGRAM_POINT_SIZE)

        for inst in renderables:
            if not isinstance(inst.asset, RasterPoints):
                continue
            radius_scale = inst.properties.get('radius_scale', 1.0) * pixel_mult * self.shadow_splat_scale
            glUniform1f(pts_shader.get_loc('radius_scale'), radius_scale)
            glUniformMatrix4fv(pts_shader.get_loc('model'), 1, False,
                               glm.value_ptr(inst.transform))
            glBindVertexArray(inst.asset.vao)
            glDrawArrays(GL_POINTS, 0, inst.asset.draw_count)

        glDisable(GL_PROGRAM_POINT_SIZE)
        pts_shader.stop()

        glDisable(GL_POLYGON_OFFSET_FILL)
        glBindVertexArray(0)

        self._shadow_map.unbind()

    def _set_shadow_uniforms(self, shader: ShaderProgram):

        has_shadow = self.enable_shadows and self._shadow_map and self.primary_directional_light is not None
        glUniform1i(shader.get_loc('enable_shadows'), int(has_shadow))

        if has_shadow:
            glUniform1f(shader.get_loc('shadow_bias'), self.shadow_bias)

            glUniformMatrix4fv(
                shader.get_loc('light_space_matrix'),
                1,
                False,
                glm.value_ptr(self._light_space_matrix)
            )

            self._shadow_map.bind_texture(unit=self.SHADOW_TEX_UNIT)
            glUniform1i(shader.get_loc('shadow_map'), self.SHADOW_TEX_UNIT)

    def _render_instance(self, instance: RasterInstance, view_matrix, projection_matrix):
        """Renders a single RasterInstance."""

        asset = instance.asset
        asset.shaders.use()

        glBindVertexArray(asset.vao)

        cam_mat = projection_matrix * view_matrix
        glUniformMatrix4fv(asset.shaders.get_loc("camera"), 1, False, glm.value_ptr(cam_mat))
        glUniformMatrix4fv(asset.shaders.get_loc("model"), 1, False, glm.value_ptr(instance.transform))

        if isinstance(asset, RasterMesh):

            # Normal matrix for lighting
            normal_mat = glm.transpose(glm.inverse(glm.mat3(instance.transform)))
            normal_np = np.array(normal_mat, dtype=np.float32).reshape(3, 3)
            glUniformMatrix3fv(asset.shaders.get_loc("normal_matrix"), 1, True, normal_np)

            # Directional light (independent of shadows)
            light = self.primary_directional_light if self.enable_direct else None
            glUniform1i(asset.shaders.get_loc("enable_ambient"), int(light is not None))

            if light is not None:
                glUniform3f(asset.shaders.get_loc("light_direction"), light.direction.x, light.direction.y, light.direction.z)
                glUniform3f(asset.shaders.get_loc("light_color"), light.color.x, light.color.y, light.color.z)
                glUniform1f(asset.shaders.get_loc("light_intensity"), light.intensity)
                glUniform3f(asset.shaders.get_loc("ambient_color"), *self.ambient_color)
                glUniform1f(asset.shaders.get_loc("ambient_intensity"), self.ambient_intensity)

            self._set_shadow_uniforms(asset.shaders)

            glUniform1i(asset.shaders.get_loc("has_texture"), int(asset.has_texture))
            glUniform4fv(asset.shaders.get_loc("base_color"), 1, asset.base_color)

            glActiveTexture(GL_TEXTURE0)
            if asset.has_texture:
                glBindTexture(GL_TEXTURE_2D, asset.texture)
            else:
                glBindTexture(GL_TEXTURE_2D, 0)

            glDrawElements(GL_TRIANGLES, asset.draw_count, GL_UNSIGNED_INT, None)
            glBindTexture(GL_TEXTURE_2D, 0)

        elif isinstance(asset, RasterPoints):
            glEnable(GL_PROGRAM_POINT_SIZE)

            pixel_mult = 25.0
            radius_scale = instance.properties.get('radius_scale', 1.0) * pixel_mult
            glUniform1f(asset.shaders.get_loc("radius_scale"), radius_scale)

            self._set_shadow_uniforms(asset.shaders)
            glUniform1f(asset.shaders.get_loc("shadow_darkness"), self.shadow_darkness)

            glDrawArrays(GL_POINTS, 0, asset.draw_count)

            glDisable(GL_PROGRAM_POINT_SIZE)

        glBindVertexArray(0)
        asset.shaders.stop()

    def _render_to_cubemap(self, agent):
        """Pass 1: renders to the cubemap."""

        main_viewport = glGetIntegerv(GL_VIEWPORT)

        if self.enable_shadows and self._shadow_map:
            self._render_shadow_pass(agent)

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
        """Pass 2: samples the cubemap."""

        self._rasterizer_shader.use()

        glUniform1i(self._rasterizer_shader.get_loc('nb_receptors'), self.total_receptors)
        glUniform1i(self._rasterizer_shader.get_loc('nb_samples'), self.samples_per_receptor)
        glUniform1f(self._rasterizer_shader.get_loc('time'), float(self._dither_counter))

        # Quasi-random sampling
        glUniform1i(self._rasterizer_shader.get_loc('use_quasi_random'), int(self._quasi_random))

        # Photoreceptor temporal integration
        glUniform1f(self._rasterizer_shader.get_loc('dt'), self._dt)

        # Write into the history buffer circularly
        frame_offset = self._frame_index % self._batch_size
        glUniform1i(self._rasterizer_shader.get_loc('frame_index'), frame_offset)

        # Bind input cubemap (texture unit 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._cubemap_fbo.texture_id)
        glUniform1i(self._rasterizer_shader.get_loc('scene_cubemap'), 0)

        # Bind SSBOs
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RECEPTORS, self.receptors_data_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_COLORS, self.final_colors_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_STATE, self.receptor_state_ssbo)

        # Dispatch the compute shader
        # Divide the total number of receptors by the workgroup size (64)
        work_groups_x = (self.total_receptors + 63) // 64
        glDispatchCompute(work_groups_x, 1, 1)

        # Wait for compute shader to finish writing to the SSBO
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self._rasterizer_shader.stop()

    def _compute_colors(self, agent):
        """The core receptors rendering logic."""

        self._tick()

        # Pass 1: Render to cubemap
        self._render_to_cubemap(agent)

        # Pass 2: Sample from cubemap
        self._sample_cubemap()

        # Unbind resources
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RECEPTORS, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_COLORS, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_STATE, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    def draw(self, view_mode: DisplayMode, point_of_view: Agent, agent: Agent):

        if view_mode in (DisplayMode.Perspective, DisplayMode.Third_person):
            if self.enable_shadows and self._shadow_map:
                self._render_shadow_pass(agent)

        if view_mode == DisplayMode.Compound:
            self._draw_voronoi()

        elif view_mode == DisplayMode.Panoramic:
            self._raster_panoramic.draw(self._cubemap_fbo.texture_id)

        elif view_mode == DisplayMode.Perspective or view_mode == DisplayMode.Third_person:

            if self._scene_baked.scene.skybox is not None:
                self._scene_baked.scene.skybox.draw(point_of_view.projection, point_of_view.view)

            renderables = self._scene_baked.get_renderables()

            for instance in renderables:
                self._render_instance(instance, point_of_view.view, point_of_view.projection)

        if view_mode == DisplayMode.Third_person:
            self._draw_eye_model(point_of_view, agent)

    def free(self):
        self._scene_baked.free()

        self._rasterizer_shader.free()
        self._cubemap_fbo.free()
        self._raster_panoramic.free()

        if self._shadow_map:
            self._shadow_map.free()

        super().free()