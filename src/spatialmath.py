from ctypes import cdll, Structure, POINTER, c_double
from viam.proto.common import Orientation
from scipy.spatial.transform import Rotation

lib = cdll.LoadLibrary("./libviam_rust_utils-linux_x86_64.so")

class OrientationVector(Structure): 
    pass

class Quaternion(Structure):
    pass

ov_array = c_double * 4

from viam.logging import getLogger
LOGGER = getLogger(__name__)

lib.free_orientation_vector_memory.argtypes = (POINTER(OrientationVector),)
lib.orientation_vector_from_quaternion.argtypes = (POINTER(Quaternion),)
lib.orientation_vector_from_quaternion.restype = POINTER(OrientationVector)
lib.orientation_vector_get_components.argtypes = (POINTER(OrientationVector),)
lib.orientation_vector_get_components.restype = POINTER(ov_array)
lib.free_quaternion_memory.argtypes = (POINTER(Quaternion),)
lib.new_quaternion.argtypes = (c_double, c_double, c_double, c_double)
lib.new_quaternion.restype = POINTER(Quaternion)

def quaternion_to_orientation_vector(r: Rotation) -> Orientation:
    q = r.as_quat()
    
    # the real component is the last one in the numpy representation
    quaternion = lib.new_quaternion(q[3], q[0], q[1], q[2])
    orientation_vector = lib.orientation_vector_from_quaternion(quaternion)
    o_x, o_y, o_z, theta = lib.orientation_vector_get_components(orientation_vector).contents
    lib.free_quaternion_memory(quaternion)
    lib.free_orientation_vector_memory(orientation_vector)
    return Orientation(o_x=o_x, o_y=o_y, o_z=o_z, theta=theta)
