from ctypes import CDLL, POINTER, Structure, byref, util
from ctypes import (c_bool, c_byte, c_void_p, c_int,
                    c_double, c_uint32, c_char_p, c_ulonglong)


class _PycairoContext(Structure):
    _fields_ = [("PyObject_HEAD", c_byte * object.__basicsize__),
                ("ctx", c_void_p),
                ("base", c_void_p)]


class _RsvgProps(Structure):
    _fields_ = [("width", c_int), ("height", c_int),
                ("em", c_double), ("ex", c_double)]


class _GError(Structure):
    _fields_ = [("domain", c_uint32), ("code", c_int), ("message", c_char_p)]


def _load_rsvg(rsvg_lib_path=None, gobject_lib_path=None, glib_lib_path=None):
    if rsvg_lib_path is None:
        rsvg_lib_path = util.find_library('rsvg-2')
    if gobject_lib_path is None:
        gobject_lib_path = util.find_library('gobject-2.0')
    if glib_lib_path is None:
        glib_lib_path = util.find_library("glib-2.0")
    gl = CDLL(glib_lib_path)
    l = CDLL(rsvg_lib_path)
    g = CDLL(gobject_lib_path)
    g.g_type_init()

    _GErrorP = POINTER(_GError)

    gl.g_error_free.restype = None
    gl.g_error_free.argtypes = [_GErrorP]

    g.g_object_unref.restype = None
    g.g_object_unref.argtypes = [c_void_p]

    l.rsvg_handle_new_from_file.argtypes = [c_char_p,
                                            POINTER(POINTER(_GError))]
    l.rsvg_handle_new_from_file.restype = c_void_p
    l.rsvg_handle_new_from_data.argtypes = [c_char_p,
                                            c_ulonglong,
                                            POINTER(POINTER(_GError))]
    l.rsvg_handle_new_from_data.restype = c_void_p
    l.rsvg_handle_render_cairo.argtypes = [c_void_p, c_void_p]
    l.rsvg_handle_render_cairo.restype = c_bool
    l.rsvg_handle_get_dimensions.argtypes = [c_void_p, POINTER(_RsvgProps)]

    return l, g

_librsvg, _libgo = _load_rsvg()


class Handle(object):
    def __init__(self, xml_string):
        self._libgo = _libgo
        lib = _librsvg
        err = POINTER(_GError)()
        self.handle = lib.rsvg_handle_new_from_data(xml_string,
                                                    len(xml_string),
                                                    byref(err))
        if self.handle is None:
            gerr = err.contents
            raise Exception(gerr.message)
        self.props = _RsvgProps()
        lib.rsvg_handle_get_dimensions(self.handle, byref(self.props))

    def render_cairo(self, ctx):
        """Returns True is drawing succeeded."""
        z = _PycairoContext.from_address(id(ctx))
        return _librsvg.rsvg_handle_render_cairo(self.handle, z.ctx)

    def __del__(self):
        if self.handle:
            self._libgo.g_object_unref(self.handle)
            self.handle = None
