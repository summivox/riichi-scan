import os
import sys
import json
import argparse

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

DEG = pi/180
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
	Existing objects in the series are **deleted**

	e.g. `make_dupes('Cube', 10)` will make 'Cube.001', 'Cube.002', ..., 'Cube.010' and
	delete 'Cube.011', 'Cube.012', ... if they existed
	"""
	names = ['%s.%03d' % (src_name, i + 1) for i in range(n)]

	# delete all existing objects in the series
	O.object.select_all(action='DESELECT')
	O.object.select_pattern(pattern=(src_name + '.*'))
	O.object.delete()
	O.object.select_all(action='DESELECT')

	# clone source object `n` times
	D.objects[src_name].select = True
	for i in range(n):
		O.object.duplicate_move_linked()
	O.object.select_all(action='DESELECT')

	C.scene.update()
	return names

def render_to_file(filename):
	""" Render active camera to `filename` """
	C.scene.render.filepath = os.path.realpath(filename)
	O.render.render(write_still=True)


########################################
# Material/texture generation

TILE_NAMES = (
	['%d%s' % (n, 'mps'[s]) for s in range(3) for n in range(10)] +
	['%dz' % n for n in range(1, 8)]
)
get_tile_image_path = lambda n: '//assets/tenhou/0-%s.png' % n

def make_textures(mat_base_name):
	"""
	Generate mat/tex pairs for all kinds of tiles based on material specified by `name`
	Return: dict mapping tile name to material name
	"""
	mat_base = D.materials[mat_base_name]
	tex_base = mat_base.texture_slots[0].texture
	tex_base_name = tex_base.name

	tile_to_mat = {}

	for tile_name in TILE_NAMES:
		mat_name = '%s-%s' % (mat_base_name, tile_name)
		tex_name = '%s-%s' % (tex_base_name, tile_name)
		img_name = 'img-tile-%s' % (tile_name)

		img_path = get_tile_image_path(tile_name)
		tile_to_mat[tile_name] = mat_name

		# rebuild mat->tex->img dependency chain

		if mat_name in D.materials:
			mat = D.materials[mat_name]
			D.materials.remove(mat)
		mat = mat_base.copy()
		mat.name = mat_name

		if tex_name in D.textures:
			tex = D.textures[tex_name]
			D.textures.remove(tex)
		tex = tex_base.copy()
		tex.name = tex_name

		if img_name in D.images:
			img = D.images[img_name]
			img.filepath = img_path
			img.update()
		else:
			img = D.images.load(img_path)
			img.name = img_name

		tex.image = img
		mat.texture_slots[0].texture = tex

	return tile_to_mat

# FIXME: hard-coded hacky autorun
make_dupes('Tile', 0)
C.scene.update()
TILE_TO_MAT = make_textures('mat-tile')

def set_ground_hue_shift(hue_shift):
	""" hue_shift: number from [0, 1), added to ground hue """
	D.materials['Ground'].node_tree.nodes['HueShift'].inputs[1].default_value = hue_shift

# FIXME: hard-coded hacky autorun
set_ground_hue_shift(random.random())


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
	C.scene.update()
	return names

def make_tiles_baseline_norot(x0, y0, n, **kwargs):
	"""
	Slightly more sophisticated tile-maker.
	Options:
		sigma_dy: std dev of dy (y = y0 + dy) [cm]
		prob_h: probability of any tile being horizontal instead of vertical [0 <= p <= 1]
		min_g: min size of gap [cm]
		prob_g: probability of deliberate gap insertion [0 <= p <= 1]
		mean_g: mean size of a deliberate gap [cm]
	"""
	sigma_dy = kwargs.get('sigma_dy', tile_l*0.05)
	prob_h = kwargs.get('prob_h', 2/14)
	min_g = kwargs.get('min_g', tile_w*0.01)
	prob_g = kwargs.get('prob_g', 2/14)
	mean_g = kwargs.get('mean_g', tile_w*0.1)

	names = make_dupes('Tile', n)
	x = x0
	for i, name in enumerate(names):
		isHori = random.random() < prob_h
		isGapBig = random.random() < prob_g
		if isGapBig:
			gap = max(min_g, random.expovariate(1/mean_g))
		else:
			gap = min_g
		dy = random.gauss(0, sigma_dy)
		tile_name = random.sample(TILE_NAMES, 1)[0]
		mat_name = TILE_TO_MAT[tile_name]

		if isHori:
			sx = tile_l; sy = tile_w; rot = -pi/2
			dy = dy - (tile_l - tile_w)/2
		else:
			sx = tile_w; sy = tile_l; rot = 0

		o = D.objects[name]
		o.location = Vector((x + sx/2, y0 + dy, 0))
		o.rotation_euler.z = rot
		o.hide = False
		o.hide_render = False
		o.material_slots[0].material = D.materials[mat_name]

		x = x + sx + gap

	C.scene.update()
	return names



########################################
# Camera & Geometry

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

	cam_d = C.scene.camera.data
	f_w = 1/(2*tan(cam_d.angle/2)) # focal length over sensor width
	if cam_d.sensor_fit == 'HORIZONTAL':
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

def set_camera_pan_tilt(pan, tilt):
	"""
	Set orientation of camera to given pan and tilt angles, with roll = 0 .
	Both angles are in radian. x axis of camera remains parallel to world xy plane.
	pan: angle between (camera +x) and (world +x)
	tilt: angle between (camera -y) and (world xy)
		NOTE: positive tilt => camera points downward
	"""
	cam = C.scene.camera
	cam.rotation_mode = 'XYZ'
	cam.rotation_euler = Euler((pi/2 - tilt, 0, pan), 'XYZ')

def fit_camera_to_objects(names):
	O.object.select_all(action='DESELECT')
	for name in names:
		D.objects[name].select = True
	O.view3d.camera_to_view_selected()
	O.object.select_all(action='DESELECT')

def move_camera_local(dp):
	"""
	Translate camera with given vector `dp` in its local frame
	"""
	cam = C.scene.camera
	dph = np.asarray(list(dp) + [1])
	loc = np.asarray(cam.matrix_world).dot(dph)
	cam.location = Vector(tuple(loc[0:3]))
	C.scene.update()

def set_camera_pan_tilt_fit_scale(pan, tilt, names, scale):
	"""
	Set orientation of camera, fit view to objects, then move camera away by a fraction of
	its distance to intersection point of camera z axis with world xy plane

	Return: distance to intersection point before scaling
	"""
	set_camera_pan_tilt(pan, tilt)
	fit_camera_to_objects(names)
	cam = C.scene.camera
	z = cam.location.z
	r = z/sin(tilt)
	move_camera_local([0, 0, r*scale])
	return r

def set_camera_random_naive(names, **kwargs):
	"""
	1.	`set_camera_pan_tilt_fit_scale` with random pan/tilt/scale
	2.	apply random camera plane translation while keeping all objects in picture
	Options:
	-	pan_range: (min, max) of pan angle [rad]
	-	tilt_range: (min, max) of tilt angle [rad]
	-	scale_range: (min, max) of scale factor
	"""

	pan_lo, pan_hi = kwargs.get('pan_range', (-15*DEG, +15*DEG))
	tilt_lo, tilt_hi = kwargs.get('tilt_range', (45*DEG, 90*DEG))
	scale_lo, scale_hi = kwargs.get('scale_range', (0.1, 0.3))
	pan = random.random()*(pan_hi - pan_lo) + pan_lo
	tilt = random.random()*(tilt_hi - tilt_lo) + tilt_lo
	scale = random.random()*(scale_hi - scale_lo) + scale_lo

	r = set_camera_pan_tilt_fit_scale(pan, tilt, names, scale)

	# random translation: ghetto estimate of visible range
	cam_d = C.scene.camera.data
	w_2f = cam_d.sensor_width / cam_d.lens / 2
	h_2f = cam_d.sensor_height / cam_d.lens / 2
	dx_max = scale*w_2f*r
	dy_max = scale*h_2f*r

	dx = (random.random()*2 - 1) * dx_max * 0.9
	dy = (random.random()*2 - 1) * dy_max * 0.9

	move_camera_local([dx, dy, 0])


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

def get_all_tile_face_2D(m_cam, names):
	""" flattened list of all corner points of faces of all tiles """
	return [list(project_world_to_picture(m_cam, p).round())
		for name in names for p in get_tile_face_3D(name)]


########################################
# main

def output(names, prefix):
	""" render to "${prefix}.png" and output point list to "${prefix}.json" """
	prefix = os.path.splitext(os.path.realpath(prefix))[0]
	render_to_file('%s.png' % prefix)
	m_cam = get_camera()
	p = get_all_tile_face_2D(m_cam, names);
	with open('%s.json' % prefix, 'w') as fout:
		json.dump(p, fout)

def run(prefix):
	# names = make_tiles_naive(-13, 0, 13)
	names = make_tiles_baseline_norot(-13, random.random()*15 - 7, 13)
	set_camera_random_naive(names)
	output(names, prefix)

def main():
	parser = argparse.ArgumentParser(prog='riichi-scan generator')
	parser.add_argument('--prefix', required=True)

	argv = sys.argv
	argv = argv[(argv.index('--') + 1):]

	args = parser.parse_args(argv)
	run(args.prefix)

if __name__ == '__main__':
	main()
