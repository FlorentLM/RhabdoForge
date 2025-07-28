import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader

class DarwinPlatform( baseplatform.BasePlatform ):
    """
    Patch for PyOpenGL's darwin platform to work with MGL
    Note: the MGL library has to be compiled with `-reexport_framework OpenGL`
    """

    DEFAULT_FUNCTION_TYPE = staticmethod( ctypes.CFUNCTYPE )
    EXTENSIONS_USE_BASE_FUNCTIONS = True

    @baseplatform.lazy_property
    def GL(self):
        MGL_WRAPPER_LIBRARY_PATH = "/Users/florent/Development/MGL/libmgl.dylib"
        try:
            lib = ctypesloader.loadLibrary(
                ctypes.cdll,
                MGL_WRAPPER_LIBRARY_PATH,
                mode=ctypes.RTLD_GLOBAL
            )
        except OSError as err:
            raise ImportError(f"Unable to load MGL wrapper library from: {MGL_WRAPPER_LIBRARY_PATH}", *err.args)

        # ========== CUSTOM MGL CONTEXT SETUP ==========
        GLenum = ctypes.c_uint
        GLMContext = ctypes.c_void_p

        createGLMContext = lib.createGLMContext
        createGLMContext.restype = GLMContext
        createGLMContext.argtypes = [GLenum, GLenum, GLenum, GLenum, GLenum, GLenum]

        MGLsetCurrentContext = lib.MGLsetCurrentContext
        MGLsetCurrentContext.restype = None
        MGLsetCurrentContext.argtypes = [GLMContext]

        GL_RGBA = 0x1908
        GL_UNSIGNED_BYTE = 0x1401
        GL_DEPTH_COMPONENT = 0x1902
        GL_STENCIL_INDEX = 0x1901

        ctx = createGLMContext(GL_RGBA, GL_UNSIGNED_BYTE,
                               GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE,
                               GL_STENCIL_INDEX, GL_UNSIGNED_BYTE)

        if not ctx:
            raise RuntimeError("MGL: Failed to create GL context")

        MGLsetCurrentContext(ctx)

        return lib

    # CGL is an alias for GL because the MGL wrapper provides all necessary symbols
    @baseplatform.lazy_property
    def CGL(self):
        return self.GL

    @baseplatform.lazy_property
    def GLU(self):
        return self.GL

    @baseplatform.lazy_property
    def GLUT( self ):
        try:
            return ctypesloader.loadLibrary(
                ctypes.cdll, 'GLUT', mode=ctypes.RTLD_GLOBAL
            )
        except OSError:
            return None

    @baseplatform.lazy_property
    def GLE(self): return self.GLUT

    @baseplatform.lazy_property
    def GetCurrentContext( self ):
        return self.CGL.CGLGetCurrentContext

    def getGLUTFontPointer( self, constant ):
        name = [ x.title() for x in constant.split( '_' )[1:] ]
        internal = 'glut' + "".join( [x.title() for x in name] )
        pointer = ctypes.c_void_p.in_dll( self.GLUT, internal )
        return ctypes.c_void_p(ctypes.addressof(pointer))

    @baseplatform.lazy_property
    def glGetError( self ):
        return self.GL.glGetError