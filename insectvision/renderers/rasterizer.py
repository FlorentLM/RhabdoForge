import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from typing import TYPE_CHECKING, List, Union, Optional
import numpy as np
from PIL import Image
from pyglm import glm

from insectvision.geometry.linalg import tangent_frames
from insectvision.utils.shared import DisplayMode, RandomnessMode, SamplingMode
from insectvision.engine.agent import Agent
from insectvision.engine.scene import Scene, Asset, AssetType
from insectvision.engine.lights import DirectionalLight
from insectvision.engine.resources import ShaderProgram, GPUResourceManager
from insectvision.renderers.base import BaseRenderer

if TYPE_CHECKING:
    from insectvision.compound_eyes import Model
    from insectvision.engine.context import Context


def _get_light_space_matrix(light: DirectionalLight, scene_center=(0.0, 0.0, 0.0), scene_radius: float = 50.0) -> glm.mat4:
    """Computes the orthographic projection and view matrix for a directional light."""

    center = glm.vec3(scene_center)
    light_pos = center + light.direction * scene_radius

    dir_np = np.array(light.direction)
    _, up_np = tangent_frames(dir_np)

    up = glm.vec3(*up_np)

    light_view = glm.lookAt(light_pos, center, up)
    light_proj = glm.ortho(
        -scene_radius, scene_radius,
        -scene_radius, scene_radius,
        0.01, scene_radius * 2.0
    )

    return light_proj * light_view


class ShadowMapFBO:
    """
    Depth-only FBO for directional light shadow mapping.
    """

    def __init__(self, resolution: int = 2048):
        self.resolution = resolution
        self.fbo_id = 0
        self.depth_tex = 0
        self._create()

    def _create(self):
        # Depth texture with hardware PCF comparison
        self.depth_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_tex)
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
                               GL_TEXTURE_2D, self.depth_tex, 0)
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
        """
        Bind the depth texture for sampling in colour shaders.
        """
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.depth_tex)

    def free(self):
        if self.fbo_id:
            glDeleteFramebuffers(1, [self.fbo_id])

        if self.depth_tex:
            glDeleteTextures(1, [self.depth_tex])


class CubemapFBO:
    """
    FBO for rendering the scene into an omnidirectional cubemap.
    """

    def __init__(self, resolution=256):
        self.resolution = resolution

        # Create FBO and Color cubemap texture
        self.fbo_id = glGenFramebuffers(1)
        self.tex_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_CUBE_MAP, self.tex_id)
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
        self.depth_id = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_id)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.resolution, self.resolution)

        # Attach textures/buffers to the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)

        # Attach just one face initially to make the FBO 'complete'
        # The render loop will correctly attach the other faces as needed
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X, self.tex_id, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_id)

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
        if self.fbo_id:
            glDeleteFramebuffers(1, [self.fbo_id])
        if self.tex_id:
            glDeleteTextures(1, [self.tex_id])
        if self.depth_id:
            glDeleteRenderbuffers(1, [self.depth_id])


class RasterMesh:
    """
    A renderable (rasterization) representation of a mesh asset.
    Holds OpenGL resources.
    """

    def __init__(self, asset: Asset, vert_path: str, frag_path: str):
        self.asset_id = asset.id
        self.draw_count = asset.indices.size

        self.shader = ShaderProgram(vert_path=vert_path, frag_path=frag_path)

        self.texture = self._load_texture(asset)
        self.has_texture = self.texture != 0
        self.base_color = asset.material.base_color.copy()  # Store base colour for fallback

        positions = asset.vertices[:, :3]
        self.normals = self._compute_normals(positions, asset.indices)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Vertex Buffer Object (VBO) for vertex data
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, asset.vertices.nbytes, asset.vertices, GL_STATIC_DRAW)

        # Element Buffer Object (EBO) for index data
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, asset.indices.nbytes, asset.indices, GL_STATIC_DRAW)

        # Vertex attribute pointers
        v_stride = asset.vertices.itemsize * 5
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, v_stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, False, v_stride, ctypes.c_void_p(asset.vertices.itemsize * 3))

        self.nbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.nbo)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

        glBindVertexArray(0)

    @staticmethod
    def _compute_normals(positions, faces):
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

    def _load_texture(self, asset):
        img = asset.texture_image
        if img is None:
            self.tex_w = 0
            self.tex_h = 0
            return 0

        # Convert to RGBA just to be sure
        img = img.convert("RGBA")
        img_data = img.tobytes()
        self.tex_w, self.tex_h = img.size

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB_ALPHA, self.tex_w, self.tex_h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        return tex_id

    def free(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.nbo])
        glDeleteBuffers(1, [self.ebo])

        self.shader.free()

        if self.texture != 0:
            glDeleteTextures(1, [self.texture])


class RasterPoints:
    """
    A renderable (rasterization) representation of a point cloud asset.
    Holds OpenGL resources.
    """

    def __init__(self, asset: Asset, vert_path: str, frag_path: str):
        self.asset_id = asset.id
        self.draw_count = asset._nb_points

        self.shader = ShaderProgram(vert_path=vert_path, frag_path=frag_path)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        # Interleave positions, colours, and radii (x, y, z, r, g, b, radius)
        packed_data = np.hstack([asset.points, asset.colors, asset.radii.reshape(-1, 1)]).astype(np.float32)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, packed_data.nbytes, packed_data, GL_STATIC_DRAW)

        stride = packed_data.itemsize * 7  # 3 for pos, 3 for color, 1 for radius

        # Vertex attribute for position
        pos_loc = glGetAttribLocation(self.shader.program_id, "position")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, False, stride, ctypes.c_void_p(0))

        # Vertex attribute for color
        color_loc = glGetAttribLocation(self.shader.program_id, "color")
        glEnableVertexAttribArray(color_loc)
        glVertexAttribPointer(color_loc, 3, GL_FLOAT, False, stride, ctypes.c_void_p(packed_data.itemsize * 3))

        # Vertex attribute for radius
        rad_loc = glGetAttribLocation(self.shader.program_id, "radius")
        glEnableVertexAttribArray(rad_loc)
        glVertexAttribPointer(rad_loc, 1, GL_FLOAT, False, stride, ctypes.c_void_p(packed_data.itemsize * 6))

        glBindVertexArray(0)

    def free(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        self.shader.free()


class RasterInstance:
    """
    Instance wrapper for the rasterizer, holding a baked asset and transform.
    """
    def __init__(self, asset: RasterMesh | RasterPoints, transform: glm.mat4, properties: dict):
        self.asset = asset
        self.transform = transform
        self.properties = properties


class RasterBaker:
    """
    Creates and caches OpenGL vertex arrays (VAOs) for each unique asset in the scene.
    """

    def __init__(self, scene: Scene, resource_manager: GPUResourceManager, enable_shadows: bool = False):
        self.scene = scene
        self.resource_manager = resource_manager

        self._cache = {}

        self._shadow_mesh = None
        self._shadow_points = None

        self._bake_assets()

        if enable_shadows:
            self._compile_shadows()

    def _bake_assets(self):
        print("Baking rasterizer assets...")

        for asset in self.scene.assets.values():

            if asset.id in self._cache:
                continue

            if asset.asset_type == AssetType.Mesh:
                self._cache[asset.id] = RasterMesh(
                    asset, 'meshRaster.vert', 'meshRaster.frag'
                )

            elif asset.asset_type == AssetType.Points:
                self._cache[asset.id] = RasterPoints(
                    asset, 'pointcloudRaster.vert', 'pointcloudRaster.frag'
                )

        print("Rasterizer asset baking complete.")

    def _compile_shadows(self):
        print("Compiling shadow depth shaders...")
        self._shadow_mesh = ShaderProgram(vert_path='shadowDepthMesh.vert', frag_path='shadowDepth.frag')
        self._shadow_points = ShaderProgram(vert_path='shadowDepthPointcloud.vert', frag_path='shadowDepth.frag')

    @property
    def renderables(self) -> List[RasterInstance]:
        renderables_list = []

        for instance in self.scene.instances:
            baked_asset = self._cache.get(instance.asset.id)

            if baked_asset:
                renderables_list.append(RasterInstance(baked_asset, instance.transform, instance.properties))

        return renderables_list

    def update_texture(self, asset: Asset):
        """Replace a raster mesh texture without rebuilding the VAO."""
        mesh = self._cache.get(asset.id)

        if not mesh or not mesh.has_texture:
            print(f"Warning: Asset '{asset.name}' has no baked texture to update.")
            return

        img = asset.texture_image
        if img is None:
            return

        if img.size != (mesh.tex_w, mesh.tex_h):
            img = img.resize((mesh.tex_w, mesh.tex_h), Image.Resampling.LANCZOS)

        img_data = img.convert("RGBA").tobytes()

        glBindTexture(GL_TEXTURE_2D, mesh.texture)
        glTexSubImage2D(
            GL_TEXTURE_2D, 0,
            0, 0, mesh.tex_w, mesh.tex_h,
            GL_RGBA, GL_UNSIGNED_BYTE, img_data
        )
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def free(self):
        for raster_asset in self._cache.values():
            raster_asset.free()

        self._cache.clear()

        if self._shadow_mesh:
            self._shadow_mesh.free()

        if self._shadow_points:
            self._shadow_points.free()


class Rasterizer(BaseRenderer):

    SHADOW_TEX_UNIT = 1

    def __init__(self,
                 model: 'Model',
                 scene: 'Scene',
                 agent: Agent,
                 context: Optional['Context'] = None,
                 time_dithering: bool = False,
                 nb_samples: int = 256,
                 randomness_mode: Union[int, str, RandomnessMode] = RandomnessMode.Pseudo,
                 sampling_mode: Union[int, str, SamplingMode] = SamplingMode.Gaussian,
                 cubemap_res: int = 512,
                 batch_size: int = 1,
                 enable_microsaccades: bool = False,
                 enable_shadows: bool = True,
                 enable_ambient: bool = True,
                 enable_direct: bool = True,
                 shadow_res: int = 2048,
                 shadow_radius: float = 50.0
                 ):

        self._model = model
        self.scene = scene

        self.resource_manager = GPUResourceManager()
        self._baker = RasterBaker(scene, enable_shadows=enable_shadows, resource_manager=self.resource_manager)

        self._cubemap_fbo = CubemapFBO(resolution=cubemap_res)
        self._cubemap_sampler = ShaderProgram(comp_path='ommatidiaRasterizer.comp')
        self._cubemap_proj_mat = Agent(fov=90.0, ratio=1.0).projection  # 90 degrees view proj matrix for each cube face

        # Lighting (before super so base can pick them up)
        self.enable_direct = enable_direct
        self.enable_shadows = enable_shadows
        self.enable_ambient = enable_ambient
        self.ambient_intensity = 0.3
        self.ambient_color = (0.4, 0.45, 0.5)  # TODO: derive ambient from the cubemap instead

        # Shadow-specific (rasterizer only)
        self._shadow_map = ShadowMapFBO(resolution=shadow_res) if enable_shadows else None
        self._shadow_radius = shadow_radius
        self.shadow_bias = 0.002
        self.shadow_darkness = 0.3
        self.shadow_splat = 1.5
        self._light_mat = glm.mat4(1.0)

        super().__init__(
            model=model,
            agent=agent,
            context=context,
            time_dithering=time_dithering,
            nb_samples=nb_samples,
            randomness_mode=randomness_mode,
            sampling_mode=sampling_mode,
            batch_size=batch_size,
            enable_microsaccades=enable_microsaccades
        )

    # Various internal helpers

    def _estim_vram_use(self):
        # Cubemap is RGBA8 (4 bytes) * 6 faces
        cubemap_mb = (self._cubemap_fbo.resolution ** 2 * 4 * 6) / (1024 * 1024)

        shadow_mb = 0.0
        if self._shadow_map:
            shadow_mb = (self._shadow_map.resolution ** 2 * 4) / (1024 * 1024)

        # rough conservative guess for shaders, VAOs, VBOs etc
        scene_assets_mb = 50.0
        return cubemap_mb + shadow_mb + scene_assets_mb

    # Internal helpers for GL resource binding

    def _set_shadow_uniforms(self, shader: ShaderProgram):
        has_shadow = self.enable_shadows and self._shadow_map and self._primary_light is not None
        glUniform1i(shader.get_loc('enable_shadows'), int(has_shadow))

        if has_shadow:
            glUniform1f(shader.get_loc('shadow_bias'), self.shadow_bias)

            glUniformMatrix4fv(
                shader.get_loc('light_space_matrix'),
                1,
                False,
                glm.value_ptr(self._light_mat)
            )

            self._shadow_map.bind_texture(unit=self.SHADOW_TEX_UNIT)
            glUniform1i(shader.get_loc('shadow_map'), self.SHADOW_TEX_UNIT)

    # Internal rendering logic and draw calls

    def _render_shadow_pass(self):
        light = self._primary_light
        if light is None:
            return

        self._light_mat = _get_light_space_matrix(
            light=light,
            scene_center=(self.agent.position.x, self.agent.position.y, self.agent.position.z),
            scene_radius=self._shadow_radius
        )

        lsm_ptr = glm.value_ptr(self._light_mat)
        self._shadow_map.bind()

        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(2.0, 4.0)

        renderables_list = self._baker.renderables

        # Meshes
        with self._baker._shadow_mesh as mesh_shader:

            glUniformMatrix4fv(mesh_shader.get_loc('light_space_matrix'), 1, False, lsm_ptr)

            for inst in renderables_list:
                if not isinstance(inst.asset, RasterMesh):
                    continue
                glUniformMatrix4fv(mesh_shader.get_loc('model'), 1, False, glm.value_ptr(inst.transform))
                glBindVertexArray(inst.asset.vao)
                glDrawElements(GL_TRIANGLES, inst.asset.draw_count, GL_UNSIGNED_INT, None)

        # Point clouds (splats)
        with self._baker._shadow_points as pts_shader:

            glUniformMatrix4fv(pts_shader.get_loc('light_space_matrix'), 1, False, lsm_ptr)

            pixel_mult = 25.0
            glEnable(GL_PROGRAM_POINT_SIZE)

            for inst in renderables_list:
                if not isinstance(inst.asset, RasterPoints):
                    continue
                rad_scale = inst.properties.get('radius_scale', 1.0) * pixel_mult * self.shadow_splat
                glUniform1f(pts_shader.get_loc('radius_scale'), rad_scale)
                glUniformMatrix4fv(pts_shader.get_loc('model'), 1, False, glm.value_ptr(inst.transform))
                glBindVertexArray(inst.asset.vao)
                glDrawArrays(GL_POINTS, 0, inst.asset.draw_count)

            glDisable(GL_PROGRAM_POINT_SIZE)

        glDisable(GL_POLYGON_OFFSET_FILL)
        glBindVertexArray(0)

        self._shadow_map.unbind()

    def _render_instance(self, inst: RasterInstance, view_mat: np.ndarray, proj_mat: np.ndarray, to_screen: bool = False):
        """Renders a single RasterInstance."""

        with inst.asset.shader as shader:

            glBindVertexArray(inst.asset.vao)

            cam_mat = proj_mat * view_mat
            glUniformMatrix4fv(shader.get_loc("camera"), 1, False, glm.value_ptr(cam_mat))
            glUniformMatrix4fv(shader.get_loc("model"), 1, False, glm.value_ptr(inst.transform))

            # these toggles are only if drawing to the human screen
            sim_insect = int(self.simulate_insect_colours and not self.uv_encoded_textures) if to_screen else 0
            uv_enc = int(self.uv_encoded_textures) if to_screen else 0

            glUniform1i(shader.get_loc('false_colors'), sim_insect)
            glUniform1i(shader.get_loc('uv_encoding'), uv_enc)

            if isinstance(inst.asset, RasterMesh):

                # Normal matrix for lighting
                norm_mat = glm.transpose(glm.inverse(glm.mat3(inst.transform)))
                norm_np = np.array(norm_mat, dtype=np.float32).reshape(3, 3)
                glUniformMatrix3fv(shader.get_loc("normal_matrix"), 1, True, norm_np)

                # Directional light (independent of shadows)
                light = self._primary_light if self.enable_direct else None
                glUniform1i(shader.get_loc("enable_ambient"), int(light is not None))

                if light is not None:
                    glUniform3f(shader.get_loc("light_direction"), light.direction.x, light.direction.y, light.direction.z)
                    glUniform3f(shader.get_loc("light_color"), light.color.x, light.color.y, light.color.z)
                    glUniform1f(shader.get_loc("light_intensity"), light.intensity)
                    glUniform3f(shader.get_loc("ambient_color"), *self.ambient_color)
                    glUniform1f(shader.get_loc("ambient_intensity"), self.ambient_intensity)

                self._set_shadow_uniforms(shader)

                glUniform1i(shader.get_loc("has_texture"), int(inst.asset.has_texture))
                glUniform4fv(shader.get_loc("base_color"), 1, inst.asset.base_color)

                glActiveTexture(GL_TEXTURE0)
                if inst.asset.has_texture:
                    glBindTexture(GL_TEXTURE_2D, inst.asset.texture)
                else:
                    glBindTexture(GL_TEXTURE_2D, 0)

                glDrawElements(GL_TRIANGLES, inst.asset.draw_count, GL_UNSIGNED_INT, None)
                glBindTexture(GL_TEXTURE_2D, 0)

            elif isinstance(inst.asset, RasterPoints):
                glEnable(GL_PROGRAM_POINT_SIZE)

                pixel_mult = 25.0
                rad_scale = inst.properties.get('radius_scale', 1.0) * pixel_mult
                glUniform1f(shader.get_loc("radius_scale"), rad_scale)

                self._set_shadow_uniforms(shader)
                glUniform1f(shader.get_loc("shadow_darkness"), self.shadow_darkness)

                glDrawArrays(GL_POINTS, 0, inst.asset.draw_count)

                glDisable(GL_PROGRAM_POINT_SIZE)

            glBindVertexArray(0)

    def _render_to_cubemap(self):
        """Pass 1: renders to the cubemap."""

        main_viewport = glGetIntegerv(GL_VIEWPORT)

        if self.enable_shadows and self._shadow_map:
            self._render_shadow_pass()

        self._cubemap_fbo.bind()

        # Look-at directions and 'up' vectors for each face:

        # GL_TEXTURE_CUBE_MAP_POSITIVE_X  ->  Right
        # GL_TEXTURE_CUBE_MAP_NEGATIVE_X  ->  Left
        # GL_TEXTURE_CUBE_MAP_POSITIVE_Y  ->  Up
        # GL_TEXTURE_CUBE_MAP_NEGATIVE_Y  ->  Down
        # GL_TEXTURE_CUBE_MAP_POSITIVE_Z  ->  Back
        # GL_TEXTURE_CUBE_MAP_NEGATIVE_Z  ->  Front

        look_dirs = [
            self.agent.right,     # For +X, camera looks towards the agent's right
            self.agent.left,      # For -X, camera looks towards the agent's left
            self.agent.up,        # For +Y, camera looks up
            self.agent.down,      # For -Y, camera looks down
            self.agent.backward,  # For +Z, camera looks backward
            self.agent.forward,   # For -Z, camera looks forward
        ]

        ups = [
            self.agent.down,      # Up for looking right/left is agent's down
            self.agent.down,
            self.agent.backward,  # Up for looking up is agent's backward
            self.agent.forward,   # Up for looking down is agent's forward
            self.agent.down,      # Up for looking backward/forward is agent's down
            self.agent.down,
        ]

        r, g, b = self.scene.background_color
        glClearColor(r, g, b, 1.0)

        for i in range(6):
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   self._cubemap_fbo.tex_id, 0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            view = glm.lookAt(
                glm.vec3(self.agent.position),
                glm.vec3(self.agent.position + look_dirs[i]),  # target point is eye + direction
                glm.vec3(ups[i])
            )

            # Draw skybox first into the cubemap face
            if self._baker.scene.skybox is not None:
                self._baker.scene.skybox.draw(self._cubemap_proj_mat, view)

            for instance in self._baker.renderables:
                self._render_instance(instance, view, self._cubemap_proj_mat, to_screen=False)

        self._cubemap_fbo.unbind()

        # Regenerate mipmaps after rendering to the cubemap
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._cubemap_fbo.tex_id)
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

        # Restore the viewport to the main window's dimensions
        glViewport(main_viewport[0], main_viewport[1], main_viewport[2], main_viewport[3])

    def _sample_cubemap(self):
        """Pass 2: samples the cubemap into the intermediate ray_results buffer."""

        with self._cubemap_sampler as shader:

            # Dynamic buffer for actuated directions and static (for positions)
            with self.eye_buffers.grouped_bind(['rays_intermediate', 'rcpt_static', 'lens_static', 'rcpt_dynamic']):

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_CUBE_MAP, self._cubemap_fbo.tex_id)
                glUniform1i(shader.get_loc('scene_cubemap'), 0)
                glUniform1i(shader.get_loc('randomness_mode '), self._randomness_mode)

                glUniform1i(shader.get_loc('nb_samples'), self.nb_samples)
                glUniform1f(shader.get_loc('dither_counter'), float(self._dither_counter))

                # Dispatch: one thread per sample
                total_work = self._model.size * self.nb_samples
                work_groups = (total_work + 63) // 64

                glDispatchCompute(work_groups, 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _sample_scene(self):
        self._render_to_cubemap()
        self._sample_cubemap()

    # Main public methods

    def draw(self, view_mode: DisplayMode, point_of_view: Union['Agent', 'OrbitCamera']):

        if view_mode in (DisplayMode.Perspective, DisplayMode.Third_person):
            if self.enable_shadows and self._shadow_map:
                self._render_shadow_pass()

        if view_mode == DisplayMode.Compound:
            self._draw_eye_firstperson()

        elif view_mode == DisplayMode.Panoramic:
            self.screen_surface.draw(
                self._cubemap_fbo.tex_id, is_cubemap=True,
                simulate_insect_vision=self.simulate_insect_colours,
                uv_encoded_textures=self.uv_encoded_textures
            )

        elif view_mode == DisplayMode.Perspective or view_mode == DisplayMode.Third_person:

            if self._baker.scene.skybox is not None:
                self._baker.scene.skybox.draw(point_of_view.projection, point_of_view.view, self.simulate_insect_colours, self.uv_encoded_textures)

            for instance in self._baker.renderables:
                self._render_instance(instance, point_of_view.view, point_of_view.projection, to_screen=True)

        if view_mode == DisplayMode.Third_person:
            self._draw_eye_thirdperson(point_of_view)

    # Dynamic updates

    def update_texture(self, asset: Asset):
        """Update a texture on the GPU for given Asset."""
        self._baker.update_texture(asset)

    # Public properties and methods

    @property
    def _primary_light(self):
        for l in self.scene.directional_lights:
            if l.active and l.intensity > 0:
                return l
        return None

    # Cleanup

    def free(self):
        self._baker.free()
        self._cubemap_sampler.free()
        self._cubemap_fbo.free()

        if self._screen_surface:
            self._screen_surface.free()

        if self._shadow_map:
            self._shadow_map.free()

        super().free()