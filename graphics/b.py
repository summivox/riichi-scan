import os
import json

import itertools

from math import *
import random
import numpy as np

import bpy
from mathutils import *


########################################
# parameters
#	NOTE:
#	-	Unit: 1 blender unit == 1 cm
#	-	Currently hardcoded so NOT guaranteed to be in sync with actual tile model
#		BTW this is exactly why we need parametric CSG (e.g. SolidWorks) ...
tile_l = 2.55
tile_w = 1.85
tile_h = 1.60


########################################
# Alias

C = bpy.context
D = bpy.data
O = bpy.ops


########################################
# API Helpers

def duplicateObject(scene, name, copyobj):
	"""
	Duplicate mesh and attach to new named object (defekt)
	source: http://bathatmedia.blogspot.com/2012/08/duplicating-objects-in-blender-26.html
	"""

	# Create new mesh
	mesh = bpy.data.meshes.new(name)

	# Create new object associated with the mesh
	ob_new = bpy.data.objects.new(name, mesh)

	# Copy data block from the old object into the new object
	ob_new.data = copyobj.data.copy()
	ob_new.scale = copyobj.scale
	ob_new.location = copyobj.location

	# Link new object to the given scene and select it
	scene.objects.link(ob_new)
	ob_new.select = True

	return ob_new


def make_dupes(src_name, n):
	"""
	Make `n` duplicates of a source object specified by name.
	Existing objects in the series are overwritten.

	e.g. `make_dupes('Cube', 10)` will make 'Cube.001', 'Cube.002', ..., 'Cube.010'
	"""
	names = ['%s.%03d' % (src_name, i + 1) for i in range(n)]

	# delete existing objects in the series
	O.object.select_all(action='DESELECT')
	for name in names:
		if name in D.objects:
			D.objects[name].select = True
	O.object.delete()
	O.object.select_all(action='DESELECT')

	# clone source object `n` times
	D.objects[src_name].select = True
	for i in range(n):
		O.object.duplicate()
	O.object.select_all(action='DESELECT')

	return names




########################################
# Tile Makers

def make_tiles_naive(x0, y0, n):
	"""
	Make `n` tiles, all next to each other, left-to-right (increasing x)
	starting from the center of the leftmost tile at (x0, y0)
	"""
	pitch = tile_w * 1.05
	names = make_dupes('Tile', n)
	for i, name in enumerate(names):
		o = D.objects[name]
		o.location = Vector((x0 + i*pitch, y0, 0))
		o.rotation_euler.z = 0
		o.hide = False
		o.hide_render = False
	return names


########################################
# Geometry

def get_camera_intrinsic():
	""" 
	Calculate camera intrinsic matrix (3r3c)
	NOTE:
	-	Blender convention: z axis positive direction: towards back of camera
	"""

	render = C.scene.render
	kr = render.resolution_percentage / 100
	image_w = render.resolution_x * kr
	image_h = render.resolution_y * kr

	cd = C.scene.camera.data
	f_w = 1/(2*tan(cd.angle/2)) # focal length over sensor width
	if cd.sensor_fit == 'HORIZONTAL':
		ks = f_w*image_w
	else:
		ks = f_w*image_h

	return np.matrix([
		[-ks, 0, image_w/2],
		[0, +ks, image_h/2],
		[0, 0, 1]
	])

def get_camera_extrinsic():
	"""
	Calculate camera extrinsic matrix (3r4c)
	NOTE:
	-	4r4c homogeneous matrix of camera object is "camera in world" while
		extrinsic matrix is "world in camera" (inverse)
	"""
	return np.matrix(C.scene.camera.matrix_world).I[0:3]

def get_camera():
	""" Calculate camera matrix (3r4c = intrinsic 3r3c * extrinsic 3r4c) """
	return get_camera_intrinsic() * get_camera_extrinsic()


def get_tile_face_3D(name):
	"""
	Returns 4 corners of tile face rectangle in world frame
	Order: [top-left, top-right, bottom-left, bottom-right], each point is [x, y, z]
	"""
	o = D.objects[name]
	m = np.matrix(o.matrix_world)
	dx = tile_w/2
	dy = tile_l/2
	dz = tile_h
	return [m.A.dot(ph)[0:3] for ph in [
		[-dx, +dy, +dz, 1],
		[+dx, +dy, +dz, 1],
		[-dx, -dy, +dz, 1],
		[+dx, -dy, +dz, 1]
	]]

def project_world_to_picture(m_cam, P):
	""" Project 3D point in world to picture frame (uv) """
	ph = m_cam.A.dot(np.append(np.array(P), [1], axis=0)) # 2D homogeneous
	p = ph[0:2]/ph[2]
	return p


def test():
	prefix = 'b3-test3'
	m_cam = get_camera()
	names = make_tiles_naive(-13, 0, 13)

	# render
	C.scene.render.filepath = os.path.realpath('./%s.png' % prefix)
	O.render.render(write_still=True)

	# points
	p = [list(project_world_to_picture(m_cam, p).round()) for name in names for p in get_tile_face_3D(name)]
	with open('%s.json' % prefix, 'w') as fout:
		json.dump(p, fout)

test()
