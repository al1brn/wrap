#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:14:13 2021

@author: alain
"""

bl_info = {
    "name":     "Wrapanime control",
    "author":   "Alain Bernard",
    "version":  (1, 0),
    "blender":  (2, 80, 0),
    "location": "View3D > Sidebar > Wrap",
    "description": "Wrapanime commands and custom parameters",
    "warning":   "",
    "wiki_url":  "",
    "category":  "3D View"}

import numpy as np
from math import radians, degrees, sin, cos
import wageometry as wgeo

#"""
import bpy
from mathutils import Matrix, Vector, Quaternion, Euler
#"""

wa_error_title = "\n>>>>> WrapAnime error - %s\n"

# =============================================================================================================================
# Arrays of vectors

# ---------------------------------------------------------------------------
# List of points interpolated as a function

class VFunction():
    
    def __init__(self, points, lefts=None, rights=None):
        
        self.points = np.array(points)
        
        der = np.array(self.points)
        der[1:-1] = self.points[2:] - self.points[:-2]
        der[0]    = (self.points[1] - self.points[0])/2
        der[-1]   = (self.points[-1] - self.points[-2])/2

        der *= 2/(len(self.points)-1)/3
        
        if lefts is None:
            self.lefts = self.points - der
        else:
            self.lefts = lefts
            
        if rights is None:
            self.rights = self.points + der
        else:
            self.rights = rights
        
    @property
    def count(self):
        return len(self.points)
        
    @property
    def delta(self):
        return 1/(len(self.points)-1)
        
    def __call__(self, t):
        
        n     = len(self.points)
        delta = self.delta
        
        # Numpy algorithm
        
        if True:
            ts = np.array(t)
            inds = (ts*(n-1)).astype(int)

            inds[np.greater(inds, n-2)] = n-2
            inds[np.less(inds, 0)]      = 0
            
            ps = (ts - inds*delta) / delta
            
            ps2  = ps*ps
            ps3  = ps2*ps
            _ps  = 1 - ps
            _ps2 = _ps*_ps
            _ps3 = _ps2*_ps
            
            return self.points[inds]*_ps3[:,np.newaxis] + 3*self.rights[inds]*(_ps2*ps)[:,np.newaxis] + 3*self.lefts[inds+1]*(_ps*ps2)[:,np.newaxis] + self.points[inds+1]*ps3[:,np.newaxis]
        
        # Unary algorithm
        
        index = min(n-1, max(0, int(t * (n - 1))))
        if index >= n-1:
            return self.points[-1]
        
        p   = (t - delta*index)/delta
        
        p2  = p*p
        p3  = p2*p
        _p  = 1 - p
        _p2 = _p*_p
        _p3 = _p2*_p
        
        return self.points[index]*_p3 + 3*self.rights[index]*_p2*p + 3*self.lefts[index+1]*_p*p2 + self.points[index+1]*p3
    
    
    def bezier(self):
        return self.points, self.lefts, self.rights
    
# ---------------------------------------------------------------------------
# Bezier from a function

def bezier_control_points(f, count, t0=0., t1=1., dt=0.0001):
    
    count  = max(2, count)
    delta  = (t1 - t0) / (count - 1)
    ts     = t0 + np.arange(count) * delta

    try:
        points = f(ts)
        ders   = (f(ts+dt) - f(ts-dt)) /2 /dt
    except:
        points = np.array([f(t)    for t in ts])
        d1     = np.array([f(t+dt) for t in ts])
        d0     = np.array([f(t-dt) for t in ts])
        ders   = (d1 - d0) /2 /dt
        
    ders *= delta / 3
        
    return points, points - ders, points + ders

# ---------------------------------------------------------------------------
# User friendly

def bezier_from_points(count, verts, lefts=None, rights=None):
    vf = VFunction(verts, lefts, rights)
    return bezier_control_points(vf, count)

def bezier_from_function(count, f, t0, t1):
    dt = (t1-t0)/10000
    return bezier_control_points(f, count, t0, t1, dt)


# ---------------------------------------------------------------------------
# Array transformation

def to_shape(a, shape):
    
    size = np.product(shape)
            
    na = np.array(a)
    if na.size == size:
        return np.reshape(na, shape)
    elif size % na.size == 0:
        return np.resize(na, shape)
    else:
        raise RuntimeError(wa_error_title % "to_shape" +
            f"target shape: {shape}\n" +
            f"input size:   {na.size}")
        
# =============================================================================================================================
# Plural getter and setter
# shape: integer for scalars or vectors

def target_shape(count, shape):
    
    size  = np.product(shape)
    
    if size == 1:
        return count
    
    dims =  np.size(shape)
    
    if dims == 1:
        return (count, shape)

    if dims == 2:
        return (count, shape[0], shape[1])
    
    a = [count]
    a.extend(shape)
    
    return list(a)
    

def getattrs(coll, name, shape, nptype=np.float):
    
    count = len(coll)
    size  = np.product(shape)
    
    # ----- Quick method
    
    if hasattr(coll, 'foreach_get'):

        vals = np.empty(count*size, nptype)
        
        coll.foreach_get(name, vals)
        
        # Matrices must be transposed
        if np.size(shape) == 2:
            vals = np.reshape(vals, (count, shape[0], shape[1]))
            return np.transpose(vals, axes=(0, 2, 1))
        
        # Otherwise it is ok
        if size > 1:
            return np.reshape(vals, target_shape(count, shape))
        else:
            return vals
        
    # ----- Loop
        
    vals = np.empty(target_shape(count, shape), nptype)
    for i, item in enumerate(coll):
        vals[i] = getattr(item, name)
        
    return vals
        
def setattrs(coll, name, value, shape):
    
    count  = len(coll)
    size   = np.product(shape)    
        
    val    = np.array(value)
    vals   = to_shape(val, count*size)
    
    if hasattr(coll, 'foreach_set'):

        # Matrices must be transposed
        if np.size(shape) == 2:
            vals = np.reshape(vals, (count, shape[0], shape[1]))
            vals = np.transpose(vals, axes=(0, 2, 1))
            vals = np.reshape(vals, count*size)
        
        coll.foreach_set(name, vals)
        
    else:
        if size > 1:
            vals = np.reshape(vals, target_shape(count, shape))

        for i, item in enumerate(coll):
            setattr(item, name, vals[i])
    
# -----------------------------------------------------------------------------------------------------------------------------
# As methods

class WColl():
    
    def getattrs(self, name, shape, nptype=np.float):
        return getattrs(self.wrapped, name, shape, nptype)
    
    def setattrs(self, name, value, shape):
        setattrs(self.wrapped, name, value)
        
        
        
# =============================================================================================================================
# Collections

# -----------------------------------------------------------------------------------------------------------------------------
# Get a collection
# Collection can be a name or the collection itself
# return None if it doesn't exist

def collection_by_name(collection):
    if type(collection) is str:
        return bpy.data.collections.get(collection)
    else:
        return collection

# -----------------------------------------------------------------------------------------------------------------------------
# Create a collection if it doesn't exist

def create_collection(name, parent=None):

    new_coll = bpy.data.collections.get(name)

    if new_coll is None:

        new_coll = bpy.data.collections.new(name)

        cparent = collection_by_name(parent)
        if cparent is None:
            cparent = bpy.context.scene.collection

        cparent.children.link(new_coll)

    return new_coll

# -----------------------------------------------------------------------------------------------------------------------------
# Get a collection
# Can create it if it doesn't exist

def get_collection(collection, create=True, parent=None):
    
    if type(collection) is str:
        coll = bpy.data.collections.get(collection)
        if (coll is None) and create:
            return create_collection(collection, parent)
        return coll
    else:
        return collection

# -----------------------------------------------------------------------------------------------------------------------------
# Get the collection of an object

def get_object_collections(obj):
    colls = []
    for coll in bpy.data.collections:
        if obj.name in coll.objects:
            colls.append(coll)

    return colls

# -----------------------------------------------------------------------------------------------------------------------------
# Link an object to a collection
# Unlink all the linked collections

def put_object_in_collection(obj, collection=None):

    colls = get_object_collections(obj)
    for coll in colls:
        coll.objects.unlink(obj)
        
    coll = get_collection(collection, create=False)

    if coll is None:
        bpy.context.collection.objects.link(obj)
    else:
        coll.objects.link(obj)
        
    return obj

# -----------------------------------------------------------------------------------------------------------------------------
# Get a collection used by wrapanime
# The top collection is WrapAnime
# All other collections are children of WrapAnime
# The name is prefixed by "W " to avoid names collisions

def wrap_collection(name=None):
    
    # Make sure the top collection exists
    top = get_collection("WrapAnime", create=True)
    
    if name is None:
        return top
    
    # Prefix with "W "
    
    cname = name if name[:2] == "W " else "W " + name
    
    return create_collection(cname, parent=top)

# -----------------------------------------------------------------------------------------------------------------------------
# Get the top collection used by the add-on 

def control_collection():
    return wrap_collection("W Control")


# =============================================================================================================================
# Frames management

# -----------------------------------------------------------------------------------------------------------------------------
# Get a frame

def get_frame(frame_or_str, delta=0):
    
    if frame_or_str is None:
        return None
    
    if type(frame_or_str) is str:
        marker = bpy.context.scene.timeline_markers.get(frame_or_str)
        if marker is None:
            raise RuntimeError(
                wa_error_title % "get_frame" +
                F"Marker '{frame_or_str}' doesn't exist."
                )
        return marker.frame + delta
    
    return frame_or_str + delta


"""
# =============================================================================================================================
# MIGRATION
# Key frames mamangement
# Keyframes are managed by the couple data_path and index
# This is cumbersome
# Replaced here by the syntaxe "datapath.x"

# ----------------------------------------------------------------------------------------------------
# data_path, index
# Syntax name.x overrides index value if -1

def data_path_index(name, index=-1):
    
    if len(name) < 3:
        return name, index
    
    if name[-2] == ".":
        try:
            idx = ["x", "y", "z", "w"].index(name[-1])
        except:
            raise RuntimeError(
                wa_error_title % "data_path_index" + 
                f"{name}: suffix for index must be in (x, y, z, w), not '{name[-1]}'."
                )
        if index >= 0 and idx != index:
            raise RuntimeError(
                wa_error_title % "data_path_index" +
                f"Suffix of '{name}' gives index {idx} which is different from passed index {index}."
                )
            
        return name[:-2], idx
    
    return name, index

# ----------------------------------------------------------------------------------------------------
# Size of an attribute (gives possible values for index)

def attribute_size(obj, attr):
    return np.size(getattr(obj, attr))

# ----------------------------------------------------------------------------------------------------
# Is the object animated
    
def is_animated(obj):
    return obj.animation_data is not None
    
# ----------------------------------------------------------------------------------------------------
# Get animation_data. Create it if it doesn't exist

def animation_data(obj, create=True):
    animation = obj.animation_data
    if create and (animation is None):
        return obj.animation_data_create()
    else:
        return animation

# ----------------------------------------------------------------------------------------------------
# Get animation action. Create it if it doesn't exist
    
def animation_action(obj, create=True):
    animation = animation_data(obj, create)
    if animation is None:
        return None
    
    action = animation.action
    if create and (action is None):
        animation.action = bpy.data.actions.new(name="WA action")
    
    return animation.action

# ----------------------------------------------------------------------------------------------------
# Get fcurves. Create it if it doesn't exist

def get_fcurves(obj, create=True):
    
    aa = animation_action(obj, create=create)
    
    if aa is None:
        return None
    else:
        return aa.fcurves

# ----------------------------------------------------------------------------------------------------
# Check if a fcurve is an animation of a property

def is_fcurve_of(fcurve, name, index=-1):
    
    if fcurve.data_path == name:
        if (index == -1) or (fcurve.array_index < 0):
            return True

        return fcurve.array_index == index
    
    return False

# ----------------------------------------------------------------------------------------------------
# Return the animation curves of a property
# Since there could be more than one curve, an array, possibly empty, is returned

def get_acurves(obj, name, index=-1):
    
    name, index = data_path_index(name, index)

    acs = []
    fcurves = get_fcurves(obj, create=False)
    if fcurves is not None:
        for fcurve in fcurves:
            if is_fcurve_of(fcurve, name, index):
                acs.append(fcurve)
            
    return acs
    

# ----------------------------------------------------------------------------------------------------
# Delete a fcurve

def delete_acurves(obj, acurves):
    fcurves = get_fcurves(obj)
    try:
        for fcurve in acurves:
            fcurves.remove(fcurve)
    except:
        pass

# ----------------------------------------------------------------------------------------------------
# fcurve integral

def fcurve_integral(fcurve, frame_start=None, frame_end=None):
    
    if frame_start is None:
        frame_start= bpy.context.scene.frame_start
        
    if frame_end is None:
        frame_end= bpy.context.scene.frame_end
        
    # Raw algorithm : return all the values per frame
        
    vals = [fcurve.evaluate(i) for i in range(frame_start, frame_end+1)]
    vals = vals - fcurve.evaluate(frame_start)
    return np.cumsum(vals)


# EO MIGRATION
"""
    

# =============================================================================================================================
# Objects management utilities    

# -----------------------------------------------------------------------------------------------------------------------------
# Create an object

def create_object(name, what='CUBE', collection=None, parent=None, **kwargs):
    
    generics = ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'VOLUME', 'ARMATURE', 'LATTICE',
                'EMPTY', 'GPENCIL', 'CAMERA', 'LIGHT', 'SPEAKER', 'LIGHT_PROBE']
    typeds = ['CIRCLE', 'CONE', 'CUBE', 'GIZMO_CUBE', 'CYLINDER', 'GRID', 'ICOSPHERE', 'MONKEY', 'PLANE', 'TORUS', 'UVSPHERE',
              'BEZIERCIRCLE', 'BEZIERCURVE', 'NURBSCIRCLE', 'NURBSCURVE', 'NURBSPATH']
    
    
    if what in generics:
        
        bpy.ops.object.add(type=what, **kwargs)
    
    elif what == 'CIRCLE':
        bpy.ops.mesh.primitive_circle_add(**kwargs)
    elif what == 'CONE':
        bpy.ops.mesh.primitive_cone_add(**kwargs)
    elif what == 'CUBE':
        bpy.ops.mesh.primitive_cube_add(**kwargs)
    elif what == 'GIZMO_CUBE':
        bpy.ops.mesh.primitive_cube_add_gizmo(**kwargs)
    elif what == 'CYLINDER':
        bpy.ops.mesh.primitive_cylinder_add(**kwargs)
    elif what == 'GRID':
        bpy.ops.mesh.primitive_grid_add(**kwargs)
    elif what in ['ICOSPHERE', 'ICO_SPHERE']:
        bpy.ops.mesh.primitive_ico_sphere_add(**kwargs)
    elif what == 'MONKEY':
        bpy.ops.mesh.primitive_monkey_add(**kwargs)
    elif what == 'PLANE':
        bpy.ops.mesh.primitive_plane_add(**kwargs)
    elif what == 'TORUS':
        bpy.ops.mesh.primitive_torus_add(**kwargs)
    elif what in ['UVSPHERE', 'UV_SPHERE', 'SPHERE']:
        bpy.ops.mesh.primitive_uv_sphere_add(**kwargs)
        
        
    elif what in ['BEZIERCIRCLE', 'BEZIER_CIRCLE']:
        bpy.ops.curve.primitive_bezier_circle_add(**kwargs)
    elif what in ['BEZIERCURVE', 'BEZIER_CURVE', 'BEZIER']:
        bpy.ops.curve.primitive_bezier_curve_add(**kwargs)
    elif what in ['NURBSCIRCLE', 'NURBS_CIRCLE']:
        bpy.ops.curve.primitive_nurbs_circle_add(**kwargs)
    elif what in ['NURBSCURVE', 'NURBS_CURVE', 'NURBS']:
        bpy.ops.curve.primitive_nurbs_curve_add(**kwargs)
    elif what in ['NURBSPATH', 'NURBS_PATH']:
        bpy.ops.curve.primitive_nurbs_path_add(**kwargs)
        
    else:
        raise RuntimeError(
            wa_error_title % "create_object" +
            f"Invalid object creation name: '{what}' is not valid.",
            f"Valid codes are {generics + typeds}")
        
    obj             = bpy.context.active_object
    obj.name        = name
    obj.parent      = parent
    obj.location    = bpy.context.scene.cursor.location
    
    # Links exclusively to the requested collection
    
    if collection is not None:
        bpy.ops.collection.objects_remove_all()
        get_collection(collection).objects.link(obj)    

    return wrap(obj)

# -----------------------------------------------------------------------------------------------------------------------------
# Get an object by name or object itself
# The object can also be a WObject
# If otype is not None, the type of the object must the given value
            
def get_object(obj_or_name, mandatory=True, otype=None):
    
    if type(obj_or_name) is str:
        obj = bpy.data.objects.get(obj_or_name)
        
    elif hasattr(obj_or_name, 'name'):
        obj = bpy.data.objects.get(obj_or_name.name)
        
    else:
        obj = obj_or_name
        
    if (obj is None) and mandatory:
        raise RuntimeError(
            wa_error_title % "get_object" +
            f"Object '{obj_or_name}' doesn't exist")
        
    if (obj is not None) and (otype is not None):
        if obj.type != otype:
            raise RuntimeError(
                wa_error_title % "get_object" +
                    f"Object type error: '{otype}' is expected",
                    f"The type of the Blender object '{obj.name}' is '{obj.type}."
                    )
            
    return wrap(obj)

# -----------------------------------------------------------------------------------------------------------------------------
# Get an object and create it if it doesn't exist
# if create is None -> no creation
# For creation, the create argument must contain a valid object creation name
    
def get_create_object(obj_or_name, create=None, collection=None, **kwargs):
    
    obj = get_object(obj_or_name, mandatory = create is None)
    if obj is not None:
        return obj
    
    return wrap(create_object(obj_or_name, what=create, collection=collection, parent=None, **kwargs))


def get_control_object():
    
    wctl = get_create_object("W Control", create='EMPTY', collection=control_collection())
    ctl = wctl.wrapped
    
    # Ensure _RNA_UI prop exists
    rna = ctl.get('_RNA_UI')
    if rna is None:
        ctl['_RNA_UI'] = {}
        
    return ctl
        


# -----------------------------------------------------------------------------------------------------------------------------
# Copy modifiers

def copy_modifiers(source, target):
    
    for mSrc in source.modifiers:
        
        mDst = target.modifiers.get(mSrc.name, None)
        if not mDst:
            mDst = target.modifiers.new(mSrc.name, mSrc.type)

        # collect names of writable properties
        properties = [p.identifier for p in mSrc.bl_rna.properties if not p.is_readonly]

        # copy those properties
        for prop in properties:
            setattr(mDst, prop, getattr(mSrc, prop))    


# -----------------------------------------------------------------------------------------------------------------------------
# Duplicate an object and its hierarchy

def duplicate_object(obj, collection=None, link=False, modifiers=False, children=False):
    
    # ----- Object copy
    dupl = obj.copy()

    # ----- Data copy
    if obj.data is not None:
        if not link:
            dupl.data = obj.data.copy()
            
    # ----- Modifiers
    if modifiers:
        copy_modifiers(obj, dupl)

    # ----- Collection to place the duplicate into
    if collection is None:
        colls = get_object_collections(obj)
        for coll in colls:
            coll.objects.link(dupl)
    else:
        collection.objects.link(dupl)

    # ----- Children copy
    if children:
        for child in obj.children:
            duplicate_object(child, collection=collection, link=link).parent = dupl

    # ----- Done !
    return dupl

# -----------------------------------------------------------------------------------------------------------------------------
# Delete an object and its children

def delete_object(obj_or_name, children=True):
    
    wobj = get_object(obj_or_name, mandatory=False)
    if wobj is None:
        return
    
    obj = wobj.wrapped

    def add_to_coll(o, coll):
        for child in o.children:
            add_to_coll(child, coll)
        coll.append(o)

    coll = []
    if children:
        add_to_coll(obj, coll)
    else:
        coll = [obj]

    for o in coll:
        bpy.data.objects.remove(o)


# -----------------------------------------------------------------------------------------------------------------------------
# Smooth

def smooth_object(obj):

    mesh = obj.data
    for f in mesh.bm.faces:
        f.smooth = True
    mesh.done()

    return obj

# -----------------------------------------------------------------------------------------------------------------------------
# Hide / Show

def hide_object(obj, value=True, frame=None, viewport=True):
    
    if viewport:
        obj.hide_viewport = value
        
    obj.hide_render = value

    if frame is not None:
        iframe = get_frame(frame)
        if viewport:
            obj.keyframe_insert(data_path="hide_viewport", frame=iframe)
        obj.keyframe_insert(data_path="hide_render", frame=iframe)

def show_object(obj, value=False, frame=None, viewport=True):
    hide_object(obj, not value, frame=frame, viewport=viewport)

# -----------------------------------------------------------------------------------------------------------------------------
# Assign a texture

def set_material(obj, material_name):

    # Get material
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        return
        # mat = bpy.data.materials.new(name="Material")

    # Assign it to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return mat

        
        
# =============================================================================================================================
# Wrappers

# ---------------------------------------------------------------------------
# bpy_struct wrapper
# wrapped : bpy_struct


class WStruct():
    
    def __init__(self, wrapped):
        super().__setattr__("wrapped", wrapped)
    
    def __getattr__(self, name):
        if name in dir(self):
            return getattr(self, name)
        else:
            return getattr(self.wrapped, name)
        
    def __setattr__(self, name, value):
        if name in dir(self.wrapped):
            if not name in dir(self):
                setattr(self.wrapped, name, value)
        super().__setattr__(name, value)
        
    @property
    def class_name(self):
        return self.wrapped.__class__.__name__
    
    # ----------------------------------------------------------------------------------------------------
    # Ensure update
    
    def mark_update(self):
        #self.wrapped.id_data.update_tag(refresh={'OBJECT', 'DATA', 'TIME'})
        #self.wrapped.id_data.update_tag(refresh={'OBJECT', 'DATA', 'TIME'})
        #self.wrapped.id_data.update_tag(refresh={'TIME'})
        self.wrapped.id_data.update_tag()
    
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Keyframes
    
    # ----------------------------------------------------------------------------------------------------
    # data_path, index
    # Syntax name.x overrides index value if -1
    
    @staticmethod
    def data_path_index(name, index=-1):
        
        if len(name) < 3:
            return name, index
        
        if name[-2] == ".":
            try:
                idx = ["x", "y", "z", "w"].index(name[-1])
            except:
                raise RuntimeError(
                    wa_error_title % "WStruct.data_path_index" + 
                    f"{name}: suffix for index must be in (x, y, z, w), not '{name[-1]}'."
                    )
            if index >= 0 and idx != index:
                raise RuntimeError(
                    wa_error_title % "data_path_index" +
                    f"Suffix of '{name}' gives index {idx} which is different from passed index {index}."
                    )
                
            return name[:-2], idx
        
        return name, index

    # ----------------------------------------------------------------------------------------------------
    # Size of an attribute (gives possible values for index)
    
    def attribute_size(self, attr):
        return np.size(getattr(self.wrapped, attr))

    # ----------------------------------------------------------------------------------------------------
    # Is the object animated
        
    @property
    def is_animated(self):
        return self.wrapped.animation_data is not None
    
    # ----------------------------------------------------------------------------------------------------
    # Get animation_data. Create it if it doesn't exist
    
    def animation_data(self, create=True):
        animation = self.wrapped.animation_data
        if create and (animation is None):
            return self.wrapped.animation_data_create()
        else:
            return animation

    # ----------------------------------------------------------------------------------------------------
    # Get animation action. Create it if it doesn't exist
        
    def animation_action(self, create=True):
        animation = self.animation_data(create)
        if animation is None:
            return None
        
        action = animation.action
        if create and (action is None):
            animation.action = bpy.data.actions.new(name="WA action")
        
        return animation.action

    # ----------------------------------------------------------------------------------------------------
    # Get fcurves. Create it if it doesn't exist
    
    def get_fcurves(self, create=True):
        
        aa = self.animation_action(create)
        
        if aa is None:
            return None
        else:
            return aa.fcurves

    # ----------------------------------------------------------------------------------------------------
    # Check if a fcurve is an animation of a property
    
    @staticmethod
    def is_fcurve_of(fcurve, name, index=-1):
        
        if fcurve.data_path == name:
            if (index == -1) or (fcurve.array_index < 0):
                return True
    
            return fcurve.array_index == index
        
        return False

    # ----------------------------------------------------------------------------------------------------
    # Return the animation curves of a property
    # Since there could be more than one curve, an array, possibly empty, is returned
    
    def get_acurves(self, name, index=-1):
        
        name, index = self.data_path_index(name, index)
    
        acs = []
        fcurves = self.get_fcurves(create=False)
        if fcurves is not None:
            for fcurve in fcurves:
                if self.is_fcurve_of(fcurve, name, index):
                    acs.append(fcurve)
                
        return acs
    

    # ----------------------------------------------------------------------------------------------------
    # Delete a fcurve
    
    def delete_acurves(self, acurves):
        fcurves = self.get_fcurves()
        try:
            for fcurve in acurves:
                fcurves.remove(fcurve)
        except:
            pass

    # ----------------------------------------------------------------------------------------------------
    # fcurve integral
    
    @staticmethod
    def fcurve_integral(fcurve, frame_start=None, frame_end=None):
        
        if frame_start is None:
            frame_start= bpy.context.scene.frame_start
            
        if frame_end is None:
            frame_end= bpy.context.scene.frame_end
            
        # Raw algorithm : return all the values per frame
            
        vals = [fcurve.evaluate(i) for i in range(frame_start, frame_end+1)]
        vals = vals - fcurve.evaluate(frame_start)
        return np.cumsum(vals)
    
                
    # ----------------------------------------------------------------------------------------------------
    # Access to an animation curve
        
    def get_acurves_or_value(self, name, frame=None, index=-1):
        
        name, index =self. data_path_index(name, index)
        
        acurves = self.get_acurves(name, index)
        
        if len(acurves) == 0:
            val = getattr(self.wrapped, name)
            if index < 0:
                return val
            else:
                return val[index]
            
        frame = get_frame(frame)
        if frame is None:
            return acurves
        
        val = getattr(self.wrapped, name)
        for i, fcurve in enumerate(acurves):
            v = fcurve.evaluate(frame)
            if index >= 0:
                val[i] = v
                
        return val
        
    
    # ----------------------------------------------------------------------------------------------------
    # Get a keyframe at a given frame
        
    def get_kfs(self, name, frame, index=-1):
        
        acurves = self.get_acurves(name, index)
        frame = get_frame(frame)
        
        kfs = []
        for fcurve in acurves:
            for kf in fcurve.keyframe_points:
                if kf.co[0] == frame:
                    kfs.append(kfs)
                    break
            
        return kfs
    
    # ----------------------------------------------------------------------------------------------------
    # Create an animation curve
    
    def new_acurves(self, name, index=-1, reset=False):
        
        name, index = self.data_path_index(name, index)
        size = self.attribute_size(name)
        
        acurves = self.get_acurves(name, index)
        
        # Not an array, or a particular index in an array

        if (size == 1) or (index >= 0):
            if len(acurves) == 0:
                fcurves = self.get_fcurves()
                fcurve  = fcurves.new(data_path=name, index=index)
                acurves.append(fcurve)
            
        # All entries of an array
            
        else:
            if len(acurves) != size:
                fcurves = self.get_fcurves(create=True)
                for i in range(size):
                    if len(self.get_acurves(name, index=i)) == 0:
                        acurves.append(fcurves.new(data_path=name, index=i))
        
        # Reset
        
        if reset:
            for fcurve in acurves:
                count = len(fcurve.keyframe_points)
                for i in range(count):
                    fcurve.keyframe_points.remove(fcurve.keyframe_points[0], fast=True)
                    
        # Result
    
        return acurves
    
    # ----------------------------------------------------------------------------------------------------
    # Set an existing fcurve
        
    def set_acurves(self, name, acurves, index=-1):
        
        # Get / create the fcurves
        
        acs = self.new_acurves(name, index, reset=True)
        
        # Check the size
        if len(acs) != len(acurves):
            raise RuntimeError(
                wa_error_title % "set_acurves" + 
                f"The number of fcurves to set ({len(acs)}) doesn't match the number of passed fcurves ({len(acurves)}).\n" +
                f"name: {name}, index: {index}"
                )
            
        for f_source, f_target in zip(acurves, acs):
            
            kfp = f_source.keyframe_points
            if len(kfp) > 0:
                
                f_target.extrapolation = f_source.extrapolation
                f_target.keyframe_points.add(len(kfp))
                
                for kfs, kft in zip(kfp, f_target.keyframe_points):
                    kft.co            = kfs.co.copy()
                    kft.interpolation = kfs.interpolation
                    kft.amplitude     = kfs.amplitude
                    kft.back          = kfs.back
                    kft.easing        = kfs.easing
                    kft.handle_left   = kfs.handle_left
                    kft.handle_right  = kfs.handle_right
                    kft.period        = kfs.period
    
    # ----------------------------------------------------------------------------------------------------
    # Delete keyframes
    
    def del_kfs(self, name, frame0=None, frame1=None, index=-1):
        
        okframe0 = frame0 is not None
        okframe1 = frame1 is not None
        
        if okframe0:
            frame0 = get_frame(frame0)
        if okframe1:
            frame1 = get_frame(frame1)
            
        acurves = self.get_acurves(name, index)
        for fcurve in acurves:
            kfs = []
            for kf in fcurve.keyframe_points:
                ok = True
                if okframe0:
                    ok = kf.co[0] >= frame0
                if okframe1:
                    if kf.co[0] > frame1:
                        ok = False
                if ok:
                    kfs.append(kf)
            
            for kf in kfs:
                try:
                    fcurve.keyframe_points.remove(kf)
                except:
                    pass
            
    # ----------------------------------------------------------------------------------------------------
    # Insert a key frame
            
    def set_kfs(self, name, frame, value=None, interpolation=None, index=-1):
        
        frame = get_frame(frame)
        
        name, index = self.data_path_index(name, index)
        
        if value is not None:
            curr = getattr(self.wrapped, name)
            if index == -1:
                new_val = value
            else:
                new_val = curr
                new_val[index] = value
            setattr(self.wrapped, name, new_val)
            
        self.wrapped.keyframe_insert(name, index=index, frame=frame)
        
        if interpolation is not None:
            kfs = self.get_kfs(name, frame, index)
            for kf in kfs:
                kf.interpolation = interpolation
                
        if value is not None:
            setattr(self.wrapped, name, curr)


# ---------------------------------------------------------------------------
# Root wrapper
# wrapped = ID

class WID(WStruct):

    # ---------------------------------------------------------------------------
    # Evaluated ID
    
    @property
    def evaluated(self):
        if self.wrapped.is_evaluated:
            return self
        
        else:
            depsgraph   = bpy.context.evaluated_depsgraph_get()
            return self.__class__(self.wrapped.evaluated_get(depsgraph))
        
        
# ---------------------------------------------------------------------------
# Shape keys data blocks wrappers
# wrapped = Shapekey (key_blocks item)

class WShapekey(WStruct):
    
    @property
    def sk_name(name, step=None):
        return name if step is None else f"{name} {step:3d}"
        
    def __len__(self):
        return len(self.wrapped.data)
    
    def __getitem__(self, index):
        return self.wrapped.data[index]
    
    def check_attr(self, name):
        if name in dir(self.wrapped.data[0]):
            return
        raise RuntimeError(
            wa_error_title % "WShapeKey" +
            f"The attribut '{name}' doesn't exist for this shape key '{self.name}'."
            )

    @property
    def verts(self):
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count*3, np.float)
        data.foreach_get("co", a)
        return a.reshape((count, 3))
    
    @verts.setter
    def verts(self, value):
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count*3)
        data.foreach_set("co", a)

    @property
    def lefts(self):
        self.check_attr("handle_left")
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count*3, np.float)
        data.foreach_get("handle_left", a)
        return a.reshape((count, 3))
    
    @lefts.setter
    def lefts(self, value):
        self.check_attr("handle_left")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count*3)
        data.foreach_set("handle_left", a)

    @property
    def rights(self):
        self.check_attr("handle_right")
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count*3, np.float)
        data.foreach_get("handle_right", a)
        return a.reshape((count, 3))
    
    @rights.setter
    def rights(self, value):
        self.check_attr("handle_right")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count*3)
        data.foreach_set("handle_right", a)

    @property
    def radius(self):
        self.check_attr("radius")
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count, np.float)
        data.foreach_get("radius", a)
        return a
    
    @radius.setter
    def radius(self, value):
        self.check_attr("radius")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count)
        data.foreach_set("radius", a)

    @property
    def tilts(self):
        self.check_attr("tilt")
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count, np.float)
        data.foreach_get("tilt", a)
        return a
    
    @tilts.setter
    def tilts(self, value):
        self.check_attr("tilt")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count)
        data.foreach_set("tilt", a)


    
# ---------------------------------------------------------------------------
# Mesh mesh wrapper
# wrapped : data block of mesh object

class WMesh(WID):
    
    @property
    def owner(self):
        for obj in bpy.data.objects:
            if obj.data is not None:
                if obj.data.name == self.name:
                    return obj
        return None
    
    # Mesh vertices update
    
    def mark_update(self):
        super().mark_update()
        self.wrapped.update()
        
    # Vertices count
        
    @property
    def verts_count(self):
        return len(self.wrapped.vertices)
    
    # Vertices (uses verts not to override vertices attributes)
    
    @property
    def verts(self):
        verts = self.wrapped.vertices
        a    = np.empty(len(verts)*3, np.float)
        verts.foreach_get("co", a)
        return np.reshape(a, (len(verts), 3))
    
    @verts.setter
    def verts(self, vectors):
        verts = self.wrapped.vertices
        a     = to_shape(vectors, (len(verts)*3))
        verts.foreach_set("co", a)
        self.mark_update()
        
    # x, y, z vertices access    
        
    @property
    def xs(self):
        return self.verts[:, 0]
    
    @xs.setter
    def xs(self, values):
        locs = self.verts
        locs[:, 0] = to_shape(values, self.vcount)
        self.verts = locs
        
    @property
    def ys(self):
        return self.verts[:, 1]
        
    @ys.setter
    def ys(self, values):
        locs = self.verts
        locs[:, 1] = to_shape(values, self.vcount)
        self.verts = locs
        
    @property
    def zs(self):
        return self.vertices[:, 2]

    @zs.setter
    def zs(self, values):
        locs = self.verts
        locs[:, 2] = to_shape(values, self.vcount)
        self.verts = locs
        
    # edges as indices
        
    @property
    def edge_indices(self):
        edges = self.wrapped.edges
        return [e.key for e in edges]
    
    # edges as vectors
    
    @property
    def edge_vertices(self):
        return self.verts[np.array(self.edge_indices)]
    
    # polygons as indices
        
    @property
    def poly_indices(self):
        polygons = self.wrapped.polygons
        return [tuple(p.vertices) for p in polygons]
        
    # polygons as vectors
        
    @property
    def poly_vertices(self):
        polys = self.poly_indices
        verts = self.verts
        return [ [list(verts[i]) for i in poly] for poly in polys]
    
    # ---------------------------------------------------------------------------
    # Polygons centersand normals
    
    @property
    def poly_centers(self):
        polygons = self.wrapped.polygons
        a = np.empty(len(polygons)*3, np.float)
        polygons.foreach_get("center", a)
        return np.reshape(a, (len(polygons), 3))
        
    @property
    def normals(self):
        polygons = self.wrapped.polygons
        a = np.empty(len(polygons)*3, np.float)
        polygons.foreach_get("normal", a)
        return np.reshape(a, (len(polygons), 3))
    
    # ---------------------------------------------------------------------------
    # Set new points
    
    def new_geometry(self, verts, edges=[], polygons=[]):
        
        mesh = self.wrapped
        obj  = self.owner
        
        # Clear
        obj.shape_key_clear()
        mesh.clear_geometry()
        
        # Set
        mesh.from_pydata(verts, edges, polygons)
        
        # Update
        mesh.update()
        mesh.validate()
        
    # ---------------------------------------------------------------------------
    # Detach geometry to create a new mesh
    # polygons: an array of array of valid vertex indices
    
    def detach_geometry(self, polygons):

        verts     = self.verts

        new_inds  = np.full(len(verts), -1)
        new_verts = []
        new_polys = []
        for poly in polygons:
            new_poly = []
            for vi in poly:
                if new_inds[vi] == -1:
                    new_inds[vi] = len(new_verts)
                    new_verts.append(vi)
                new_poly.append(new_inds[vi])
            new_polys.append(new_poly)
            
        return verts[new_verts], new_polys
        
    # ---------------------------------------------------------------------------
    # Copy
    
    def copy_mesh(self, mesh, replace=False):
        
        wmesh = wrap(mesh)

        verts = wmesh.verts
        edges = wmesh.edge_indices
        polys = wmesh.poly_indices
            
        if not replace:
            x_verts = self.verts
            x_edges = self.edge_indices
            x_polys = self.poly_indices
            
            verts = np.concatenate((x_verts, verts))
                
            offset = len(x_verts)

            x_edges.extennd([(e[0] + offset, e[1] + offset) for e in edges])
            edges = x_edges
                
            x_polys.extend([ [p + offset for p in poly] for poly in polys])
            polys = x_polys

        self.new_geometry(verts, edges, polys)
    
# ---------------------------------------------------------------------------
# Spline wrapper
# wrapped : Spline

class WSpline(WStruct):
    
    @property
    def use_bezier(self):
        return self.wrapped.type == 'BEZIER'
    
    @property
    def count(self):
        if self.use_bezier:
            return len(self.wrapped.bezier_points)
        else:
            return len(self.wrapped.points)
        
    @property
    def blender_points(self):
        if self.use_bezier:
            return self.wrapped.bezier_points
        else:
            return self.wrapped.points
        
    @property        
    def verts(self):
        count  = self.count
        pts    = np.empty(count*3, np.float)
        self.blender_points.foreach_get("co", pts)
        return pts.reshape((count, 3))
    
    @property
    def handles(self):
        count  = self.count
        pts    = np.empty(count*3, np.float)
        lfs    = np.empty(count*3, np.float)
        rgs    = np.empty(count*3, np.float)

        bl_points = self.blender_points

        bl_points.foreach_get("co", pts)
        bl_points.foreach_get("handle_left", lfs)
        bl_points.foreach_get("hanlde_right", rgs)
        return pts.reshape((count, 3)), lfs.reshape((count, 3)), rgs.reshape((count, 3))
        
    # ---------------------------------------------------------------------------
    # Set the points and possibly handles for bezeir curves
    
    def set_verts(self, vectors, lefts=None, rights=None):
        
        nvectors = np.array(vectors)
        count = len(nvectors)

        bl_points = self.blender_points
        if len(bl_points) < count:
            bl_points.add(len(vectors) - len(bl_points))
            
        if len(bl_points) > count:
            raise RuntimeError(wa_error_title % "Spline.set_verts" +
                "The number of points to set is not enough\n" +
                f"Splines points: {len(bl_points)}\n" +
                f"Input points:   {count}")
            
        bl_points.foreach_set("co", np.reshape(nvectors, count*3))
        
        if self.use_bezier:
            
            if lefts is not None:
                pts = np.array(lefts).reshape(count*3)
                bl_points.foreach_set("handle_left", np.reshape(pts, count*3))
                
            if rights is not None:
                pts = np.array(rights).reshape(count*3)
                bl_points.foreach_set("handle_right", np.reshape(pts, count*3))
                
            if (lefts is None) and (rights is None):
                for bv in bl_points:
                    bv.handle_left_type  = 'AUTO'
                    bv.handle_right_type = 'AUTO'
                
        self.mark_update()
    
    
# ---------------------------------------------------------------------------
# Curve wrapper
# wrapped : Curve

class WCurve(WID):
    
    def __len__(self):
        return len(self.wrapped.splines)
    
    def __getitem__(self, index):
        return WSpline(self.wrapped.splines[index])
        
    def set_length(self, length, spline_type='BEZIER'):
        
        splines = self.wrapped.splines
        count = length - len(splines)
        if count == 0:
            return
        
        if count > 0:
            for i in range(count):
                splines.new(spline_type)
        else:
            for i in range(-count):
                splines.remove(splines[-1])
        
        self.wrapped.id_data.update_tag()
    
    @property
    def verts(self):
        verts = []
        for spline in self:
            verts.append(spline.verts)
        return verts

# ---------------------------------------------------------------------------
# Text wrapper
# wrapped : TextCurve

class WText(WID):
    
    @property
    def text(self):
        return self.wrapped.body
    
    @text.setter
    def text(self, value):
        self.wrapped.body = value
    
# ---------------------------------------------------------------------------
# Object wrapper
# wrapped: Object

class WObject(WID):
    
    def __init__(self, wrapped):
        super().__init__(wrapped)
        
    # ---------------------------------------------------------------------------
    # Data
    
    @property
    def object_type(self):
        data = self.wrapped.data
        if data is None:
            return 'Empty'
        else:
            return data.__class__.__name__
    
    @property
    def wdata(self):

        data = self.wrapped.data
        if data is None:
            return None
        
        name = data.__class__.__name__
        if name == 'Mesh':
            return WMesh(data)
        elif name == 'Curve':
            return WCurve(data)
        elif name == 'TextCurve':
            return WText(data)
        else:
            raise RuntimeError(
                wa_error_title % "WObject.wdata" +
                "Data class '{name}' not yet supported !"
                )
            
    def origin_to_geometry(self):

        wmesh = self.wdata
        if wmesh.class_name != "Mesh":
            raise RuntimeError(
                wa_error_title % "origin_to_geometry" +
                "origin_to_geometry can only be called with a Mesh objecs"
                )
            
        verts = wmesh.verts
        origin = np.sum(verts, axis=0)/len(verts)
        wmesh.verts = verts - origin
        
        self.location = np.array(self.location) + origin
    
    # ---------------------------------------------------------------------------
    # Location
    
    @property
    def location(self):
        return np.array(self.wrapped.location)
    
    @location.setter
    def location(self, value):
        self.wrapped.location = to_shape(value, 3)
    
    @property
    def x(self):
        return self.wrapped.location.x
    
    @x.setter
    def x(self, value):
        self.wrapped.location.x = value
        
    @property
    def y(self):
        return self.wrapped.location.y
    
    @y.setter
    def y(self, value):
        self.wrapped.location.y = value
        
    @property
    def z(self):
        return self.wrapped.location.z
    
    @z.setter
    def z(self, value):
        self.wrapped.location.z = value
        
    # ---------------------------------------------------------------------------
    # Scale
    
    @property
    def scale(self):
        return np.array(self.wrapped.scale)
    
    @scale.setter
    def scale(self, value):
        self.wrapped.scale = to_shape(value, 3)
    
    @property
    def sx(self):
        return self.wrapped.scale.x
    
    @sx.setter
    def sx(self, value):
        self.wrapped.scale.x = value
        
    @property
    def sy(self):
        return self.wrapped.scale.y
    
    @sy.setter
    def sy(self, value):
        self.wrapped.scale.y = value
        
    @property
    def sz(self):
        return self.wrapped.scale.z
    
    @sz.setter
    def sz(self, value):
        self.wrapped.scale.z = value
        
    # ---------------------------------------------------------------------------
    # Rotation in radians
    
    @property
    def rotation(self):
        return np.array(self.wrapped.rotation_euler)
    
    @rotation.setter
    def rotation(self, value):
        self.wrapped.rotation_euler = to_shape(value, 3)
    
    @property
    def rx(self):
        return self.wrapped.rotation_euler.x
    
    @rx.setter
    def rx(self, value):
        self.wrapped.rotation_euler.x = value
        
    @property
    def ry(self):
        return self.wrapped.rotation_euler.y
    
    @ry.setter
    def ry(self, value):
        self.wrapped.rotation_euler.y = value
        
    @property
    def rz(self):
        return self.wrapped.rotation_euler.z
    
    @rz.setter
    def rz(self, value):
        self.wrapped.rotation_euler.z = value

    # ---------------------------------------------------------------------------
    # Rotation in degrees
    
    @property
    def rotationd(self):
        return np.degrees(self.wrapped.rotation_euler)
    
    @rotationd.setter
    def rotationd(self, value):
        self.wrapped.rotation_euler = np.radians(to_shape(value, 3))
    
    @property
    def rxd(self):
        return degrees(self.wrapped.rotation_euler.x)
    
    @rxd.setter
    def rxd(self, value):
        self.wrapped.rotation_euler.x = radians(value)
        
    @property
    def ryd(self):
        return degrees(self.wrapped.rotation_euler.y)
    
    @ryd.setter
    def ryd(self, value):
        self.wrapped.rotation_euler.y = radians(value)
        
    @property
    def rzd(self):
        return degrees(self.wrapped.rotation_euler.z)
    
    @rzd.setter
    def rzd(self, value):
        self.wrapped.rotation_euler.z = radians(value)
        
    # ---------------------------------------------------------------------------
    # Rotation quaternion
    
    @property
    def rotation_quaternion(self):
        return np.array(self.wrapped.rotation_quaternion)
    
    @rotation_quaternion.setter
    def rotation_quaternion(self, value):
        self.wrapped.rotation_quaternion = Quaternion(value)
        
    # ---------------------------------------------------------------------------
    # Snapshot
    
    def snapshot(self, key="Wrap"):
        m = np.array(self.wrapped.matrix_basis).reshape(16)
        self.wrapped[key] = m
        
    def to_snapshot(self, key, mandatory=False):
        m = self.wrapped.get(key)
        if m is None:
            if mandatory:
                raise RuntimeError(
                    wa_error_title % "to_snapshot" +
                    f"The snapshot key '{key}' doesn't exist for object '{self.name}'."
                    )
            return
        
        m = np.reshape(m, (4, 4))
        self.wrapped.matrix_basis = np.transpose(m)
        
        self.mark_update()
        
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    # Shape keys management

    # -----------------------------------------------------------------------------------------------------------------------------
    # Indexed shape key name
    
    @staticmethod
    def sk_name(name, step=None):
        return name if step is None else f"{name} {step:3d}"

    # -----------------------------------------------------------------------------------------------------------------------------
    # Has been the shape_keys structure created ?
    
    @property
    def has_sk(self):
        return self.wrapped.data.shape_keys is not None
    
    @property
    def shape_keys(self):
        return self.wrapped.data.shape_keys
    
    @property
    def sk_len(self):
        sks = self.shape_keys
        if sks is None:
            return 0
        return len(sks.key_blocks)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Get a shape key
    # Can create it if it doesn't exist
    
    def get_sk(self, name, step=None, create=True):
    
        fname = self.sk_name(name, step)
        obj   = self.wrapped
        data  = obj.data
        
        if data.shape_keys is None:
            if create:
                obj.shape_key_add(name=fname)
                obj.data.shape_keys.use_relative = False
            else:
                return None
        
        # Does the shapekey exists?
        
        sk = data.shape_keys.key_blocks.get(fname)
        
        # No !
        
        if (sk is None) and create:
            
            eval_time = data.shape_keys.eval_time 
            
            if step is not None:
                # Ensure the value is correct
                data.shape_keys.eval_time = step*10
            
            sk = obj.shape_key_add(name=fname)
            
            # Less impact as possible :-)
            obj.data.shape_keys.eval_time = eval_time
            
        # Depending upon the data type
        
        return WShapekey(sk)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Create a shape
    
    def create_sk(self, name, step=None):
        return self.get_sk(name, step, create=True)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Does a shape key exist?
    
    def sk_exists(self, name, step):
        return self.get_sk(name, step, create=False) is not None
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Set the eval_time property to the shape key
    
    def set_on_sk(self, name, step=None):
        
        sk = self.get_sk(name, step, create=False)
        if sk is None:
            raise RuntimeError(
                wa_error_title % "WObject.set_on_sk" + 
                f"The shape key '{self.sk_name(name, step)}' doesn't exist in object '{self.name}'!")
    
        self.wrapped.data.shape_keys.eval_time = sk.frame
        return self.wrapped.data.shape_keys.eval_time
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Delete a shape key
    
    def delete_sk(self, name=None, step=None):
        
        if not self.has_sk:
            return
        
        if name is None:
            self.wrapped.shape_key_clear()
        else:
            sk = self.get_sk(name, step, create=False)
            if sk is not None:
                self.wrapped.shape_key_remove(sk)    


# ---------------------------------------------------------------------------
# Wrapper

def wrap(name):
    
    if name is None:
        return None
    
    if type(name) is str:
        obj = bpy.data.objects.get(name)
    else:
        obj = name
        
    if obj is None:
        raise RuntimeError(
            wa_error_title % "wrap" +
            f"Object named '{name}' not found"
            )
        
    if issubclass(type(obj), WStruct):
        return obj
        
    cname = obj.__class__.__name__
    if cname == "Object":
        return WObject(obj)
    elif cname == "Curve":
        return WCurve(obj)
    elif cname == "Mesh":
        return WMesh(obj)
    elif cname == "TextCurve":
        return WText(obj)
    elif cname == 'Spline':
        return WSpline(obj)
    else:
        raise RuntimeError(
            wa_error_title % "wrap" + 
            f"Blender class {cname} not yet wrapped !")
        
        
# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Objects collection

class Duplicator():

    DUPLIS = {}

    def __init__(self, model, length=None, linked=True, modifiers=False):
        
        # The model to replicate must exist
        mdl = get_object(model, mandatory=True)
        
        self.model         = mdl
        self.model_name    = mdl.name
        self.base_name     = f"Z_{self.model_name}"
        
            
        # Let's create the collection to host the duplicates
        
        coll_name  = self.model_name + "s"
        self.collection = wrap_collection(coll_name)
        
        self.linked        = linked
        self.modifiers     = modifiers
        
        if length is not None:
            self.set_length(length)
            
    # -----------------------------------------------------------------------------------------------------------------------------
    # Adjust the number of objects in the collection
    
    def set_length(self, length):
        
        count = length - len(self)
        
        if count > 0:
            for i in range(count):
                new_obj = duplicate_object(self.model, self.collection, self.linked, self.modifiers)
                if not self.linked:
                    new_obj.animation_data_clear()
                    
        elif count < 0:
            for i in range(-count):
                obj = self.collection.objects[-1]
                delete_object(obj)
                
    def __len__(self):
        return len(self.collection.objects)
    
    def __getitem__(self, index):
        return wrap(self.collection.objects[index])
    
    def mark_update(self):
        for obj in self.collection.objects:
            obj.update_tag()
        bpy.context.view_layer.update()
                
    # -----------------------------------------------------------------------------------------------------------------------------
    # The objects are supposed to all have the same parameters
    
    @property
    def rotation_mode(self):
        if len(self) > 0:
            return self[0].rotation_mode
        else:
            return 'XYZ'
        
    @rotation_mode.setter
    def rotation_mode(self, value):
        for obj in self.collection:
            obj.rotation_modes = value
        
    @property
    def euler_order(self):
        if len(self) > 0:
            return self[0].rotation_euler.order
        else:
            return 'XYZ'
        
    @euler_order.setter
    def euler_order(self, value):
        for obj in self.collection:
            obj.rotation_euler.order = value
        
    @property
    def track_axis(self):
        if len(self) > 0:
            return self[0].track_axis
        else:
            return 'POS_Y'
        
    @track_axis.setter
    def track_axis(self, value):
        for obj in self.collection:
            obj.track_axis = value
        
    @property
    def up_axis(self):
        if len(self) > 0:
            return self[0].up_axis
        else:
            return 'Z'
        
    @up_axis.setter
    def up_axis(self, value):
        for obj in self.collection:
            obj.up_axis = value
            
    # -----------------------------------------------------------------------------------------------------------------------------
    # Basics
    
    @property
    def locations(self):
        return getattrs(self.collection.objects, "location", 3, np.float)
        
    @locations.setter
    def locations(self, value):
        setattrs(self.collection.objects, "location", value, 3)

    @property
    def scales(self):
        return getattrs(self.collection.objects, "scale", 3, np.float)
        
    @scales.setter
    def scales(self, value):
        setattrs(self.collection.objects, "scale", value, 3)
            
    @property
    def rotation_eulers(self):
        return getattrs(self.collection.objects, "rotation_euler", 3, np.float)
        
    @rotation_eulers.setter
    def rotation_eulers(self, value):
        setattrs(self.collection.objects, "rotation_euler", value, 3)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Individual getters
        
    @property
    def xs(self):
        return self.locations[:, 0]
    
    @property
    def ys(self):
        return self.locations[:, 1]
    
    @property
    def zs(self):
        return self.locations[:, 2]
    
    @property
    def rxs(self):
        return self.rotation_eulers[:, 0]
    
    @property
    def rys(self):
        return self.rotation_eulers[:, 1]
    
    @property
    def rzs(self):
        return self.rotation_eulers[:, 2]
    
    @property
    def sxs(self):
        return self.scales[:, 0]
    
    @property
    def sys(self):
        return self.scales[:, 1]
    
    @property
    def szs(self):
        return self.scales[:, 2]
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Individual setters
    
    @xs.setter
    def xs(self, value):
        a = self.locations
        a[:, 0] = to_shape(value, len(a))
        self.locations = a

    @ys.setter
    def ys(self, value):
        a = self.locations
        a[:, 1] = to_shape(value, len(a))
        self.locations = a

    @zs.setter
    def zs(self, value):
        a = self.locations
        a[:, 2] = to_shape(value, len(a))
        self.locations = a

    @rxs.setter
    def rxs(self, value):
        a = self.rotation_eulers
        a[:, 0] = to_shape(value, len(a))
        self.rotation_eulers = a

    @rys.setter
    def rys(self, value):
        a = self.rotation_eulers
        a[:, 1] = to_shape(value, len(a))
        self.rotation_eulers = a

    @rzs.setter
    def rzs(self, value):
        a = self.rotation_eulers
        a[:, 2] = to_shape(value, len(a))
        self.locations = a

    @sxs.setter
    def sxs(self, value):
        a = self.scales
        a[:, 0] = to_shape(value, len(a))
        self.scales = a

    @sys.setter
    def sys(self, value):
        a = self.scales
        a[:, 1] = to_shape(value, len(a))
        self.scales = a

    @szs.setter
    def szs(self, value):
        a = self.scales
        a[:, 2] = to_shape(value, len(a))
        self.scales = a
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with a quaternion
    
    @property
    def matrix_locals(self):
        return getattrs(self.collection.objects, "matrix_local", (4, 4), np.float)
        
    @matrix_locals.setter
    def matrix_locals(self, value):
        setattrs(self.collection.objects, "matrix_local", value, (4, 4))
        
    def transform(self, tmat):
        mls  = self.matrix_locals
        new_mls =  wgeo.mul_tmatrices(mls, tmat)
        self.matrix_locals =new_mls
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with a quaternion
    
    def quat_orient(self, quat):
        if self.rotation_mode == 'QUATERNION':
            setattrs(self.collection.objects, "rotation_quaternion", quat, 4)
            #self.rotation_quaternions = quat
        elif self.rotation_mode == 'AXIS_ANGLE':
            setattrs(self.collection.objects, "rotation_axis_angle", wgeo.axis_angle(quat, True), 4)
            #self.rotation_axis_angles = wgeo.axis_angle(quat, True)
        else:
            setattrs(self.collection.objects, "rotation_euler", wgeo.q_to_euler(quat, self.euler_order), 3)
            #self.rotation_eulers = wgeo.q_to_euler(quat, self.euler_order)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with euler
    
    def euler_orient(self, euler):
        if self.rotation_mode == 'QUATERNION':
            setattrs(self.collection.objects, "rotation_quaternion", wgeo.e_to_quat(euler, self.euler_order), 4)
            #self.rotation_quaternions = wgeo.e_to_quat(euler, self.euler_order)
        elif self.rotation_mode == 'AXIS_ANGLE':
            setattrs(self.collection.objects, "rotation_axis_angle", wgeo.axis_angle(wgeo.e_to_quat(euler, self.euler_order)), 4)
            #self.rotation_axis_angles = wgeo.axis_angle(wgeo.e_to_quat(euler, self.euler_order), True)
        else:
            setattrs(self.collection.objects, "rotation_euler", euler, 3)
            #self.rotation_eulers = euler
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with matrix
    
    def matrix_orient(self, matrix):
        if self.rotation_mode in ['QUATERNION', 'AXIS_ANGLE']:
            self.quat_orient(wgeo.m_to_quat(matrix))
        else:
            self.euler_orient(wgeo.m_to_euler(matrix, self.euler_order))
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Track to a target location
    
    def track_to(self, location):
        locs = np.array(location) - self.locations
        q    = wgeo.q_tracker(self.track_axis, locs, up=self.up_axis)
        self.quat_orient(q)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient along a given axis
    
    def orient(self, axis):
        q    = wgeo.q_tracker(self.track_axis, axis, up=self.up_axis)
        self.quat_orient(q)
        
    # ---------------------------------------------------------------------------
    # Snapshot
    
    def snapshots(self, key="Wrap"):
        for wo in self:
            wo.snapshot(key)
        
    def to_snapshots(self, key="Wrap", mandatory=False):
        for wo in self:
            wo.to_snapshot(key, mandatory)
               

# ******************************************************************************************************************************************************
# Animation class

class Animation():
    
    def __init__(self):
        self.update_ready = False
    
    def setup(self):
        self.update_ready = False
    
    def setup_update(self):
        pass
    
    def update(self, frame):
        pass
    
    def run_update(self, frame):
        if not self.update_ready:
            self.setup_update()
            self.update_ready = True
            
        self.update(frame)
        
        
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Kernel handler
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

def cur_frame():
    return bpy.context.scene.frame_current_final

def cur_time():
    return bpy.context.scene.frame_current_final / bpy.context.scene.render.fps

def frame_from_time(time):
    return time * bpy.context.scene.render.fps

def time_from_frame(frame):
    frame / bpy.context.scene.render.fps


class Interval():
    def __init__(self, start=None, end=None):
        self.start = start
        self.end   = end
        
    def __repr__(self):
        return f"[{self.start}, {self.end}["
        
        
    def when(self, frame):

        if self.start is not None:
            if frame < get_frame(self.start):
                return -1
        
        if self.end is not None:
            if frame >= get_frame(self.end):
                return 1
            
        return 0
        

class Animator():
    
    def __init__(self, interval, action_before=None, action_during=None, action_after=None):
        
        self.interval      = interval
        
        self.action_before = action_before
        self.action_during = action_during
        self.action_after  = action_after
        
        self.objects       = []
        
    @classmethod
    def Hider(Cls, objects, after=None, before=None):
        hider = Cls(Interval(after, before))
        hider.objects = objects
        hider.action_before = hider.show
        hider.action_during = hider.hide
        hider.action_after  = hider.show
        return hider
        
    @classmethod
    def Shower(Cls, objects, after=None, before=None):
        shower = Cls(Interval(after, before))
        shower.objects = objects
        shower.action_before = shower.hide
        shower.action_during = shower.show
        shower.action_after  = shower.hide
        return shower
        
    def execute(self, frame):
        
        when = self.interval.when(frame)
        if when == -1:
            if self.action_before is not None:
                self.action_before(frame)
                
        elif when == 0:
            if self.action_during is not None:
                self.action_during(frame)
                
        else:
            if self.action_after is not None:
                self.action_after(frame)
                
    def hide(self, frame):
        for obj in self.objects:
            obj.hide_render   = True
            obj.hide_viewport = bpy.context.scene.wa_hide_viewport
        
    def show(self, frame):
        for obj in self.objects:
            obj.hide_render   = False
            obj.hide_viewport = False

class FunctionAnimator(Animator):
    
    def __init__(self, f, after=None, before=None):
        
        super().__init__(Interval(after, before), None, self.run_f)
        self.f = f
        
    def run_f(self, frame):
        if Engine.verbose:
            print(f"   Run function '{self.f.__name__}' in interval {self.interval}")
        self.f(frame)
        
        

class Engine():

    SETUP      = [] # Setup functions
    FUNCTIONS  = [] # Update functions
    ANIMATIONS = [] # Animations
    
    #frame_exec    = False
    verbose       = False
    #hide_viewport = True     # For hiding / showing objects
    
    #@staticmethod
    #@property
    #def frame_exec():
    #    return bpy.context.scene.wa_frame_exec
    
    @staticmethod
    def clear():
        Engine.FUNCTIONS = []
        Engine.SETUP     = []
        Engine.ANIMATIONS = []
        
    @staticmethod
    def register_animation(animation):
        Engine.ANIMATIONS.append(animation)
        #animation.setup()
        
    @staticmethod
    def register_setup(f):
        Engine.SETUP.append(f)
        
    @staticmethod
    def setup():
        print("Engine.setup...")
        for animation in Engine.ANIMATIONS:
            animation.setup()
            animation.update_ready = False
        print("Engine.setup done!")

        for f in Engine.SETUP:
            f()
        
    @staticmethod
    def register(f, after=None, before=None):
        Engine.FUNCTIONS.append(FunctionAnimator(f, after, before))
        
    @staticmethod
    def show_objects(objects, after=None, before=None):
        Engine.FUNCTIONS.append(Animator.Shower(objects, after, before))
        
    @staticmethod
    def hide_objects(objects, after=None, before=None):
        Engine.FUNCTIONS.append(Animator.Hider(objects, after, before))
        
    @staticmethod
    def execute():
        frame = cur_frame()
        
        if Engine.verbose:
            print(f"Engine exec, frame {frame:6.1f} in interval [self.interval]")
            
        for animation in Engine.ANIMATIONS:
            animation.run_update(frame)
    
        for anm in Engine.FUNCTIONS:
            anm.execute(frame)

    @staticmethod
    def run(go=True):
        bpy.context.scene.wa_frame_exec = go
        if go:
            Engine.execute()
            
def engine_handler(scene):
    if  bpy.context.scene.wa_frame_exec:
        Engine.execute()
        
# ==========================================================================================
# UI

class ClearParamsOperator(bpy.types.Operator):
    """Delete the user parameters"""
    bl_idname = "wrap.clear_params"
    bl_label = "Clear parameters"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        del_params()
        return {'FINISHED'}


class SetupOperator(bpy.types.Operator):
    """Execute the initial set up functions"""
    bl_idname = "wrap.setup"
    bl_label = "Setup"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        Engine.setup()
        return {'FINISHED'}


class ExecOperator(bpy.types.Operator):
    """Execute the udate function"""
    bl_idname = "wrap.exec"
    bl_label = "Update"

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        Engine.execute()
        return {'FINISHED'}

def ui_param_full_name(name, group):
    return f"{group}.{name}"

def ui_param_list(ctl):
    
    params = {}
    def add(group, name, key):
        ps = params.get(group)
        if ps is None:
            params[group] = [(name, key)]
        else:
            ps.append((name, key))
            
    for key in ctl.keys():
        if key[0] != "_":
            gn = key.split('.')
            if len(gn) == 1:
                add("_main", key, key)
            else:
                add("_main" if gn[0] == "" else gn[0], gn[1], key)
                
    return params

def scalar_param(name, default=0., min=0., max=1., group="", description="Wrapanime parameter"):
    
    ctl = get_control_object()
    rna = ctl['_RNA_UI']
    
    fname = ui_param_full_name(name, group)
    prm   = ctl.get(fname)
    if prm is None:
        ctl[fname] = default
        
    rna[fname] = {
        "description": description,
        "default":     default,
        "min":         min,
        "max":         max,
        "soft_min":    min,
        "soft_max":    max,
        }

def bool_param(name, default=True, group="", description="Wrapanime parameter"):
    
    ctl = get_control_object()
    rna = ctl['_RNA_UI']
    
    fname = ui_param_full_name(name, group)
    prm   = ctl.get(fname)
    if prm is None:
        ctl[fname] = default
        
    rna[fname] = {
        "description": description,
        "default":     default,
        }

def vector_param(name, default=(0., 0., 0.), group="", description="Wrapanime parameter"):
    
    ctl = get_control_object()
    rna = ctl['_RNA_UI']
    
    fname = ui_param_full_name(name, group)
    prm   = ctl.get(fname)
    if prm is None:
        ctl[fname] = default
        
    rna[fname] = {
        "description": description,
        "default":     default,
        }
    
def get_param(name, group=""):
    ctl = get_control_object()
    val = ctl.get(ui_param_full_name(name, group))
    if val is None:
        print(f"Wrapanime WARNING: param {name} doesn't exist.")
    return val

def del_params():
    ctl = get_control_object()
    keys = ctl.keys()
    for k in keys:
        del ctl[k]
    

class WAMainPanel(bpy.types.Panel):
    """Wrapanime commands"""
    bl_label        = "Commands"
    bl_category     = "Wrap"
    #bl_idname       = "SCENE_PT_layout"
    bl_space_type   = 'VIEW_3D'
    bl_region_type  = 'UI'
    #bl_context      = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout.operator("wrap.setup", icon='FACE_MAPS')
        layout.operator("wrap.exec", icon='FILE_REFRESH')

        if context.screen.is_animation_playing:
            layout.operator("screen.animation_play", text="Pause", icon='PAUSE')
        else:
            layout.operator("screen.animation_play", text="Play", icon='PLAY')

        row = layout.row()
        row.prop(scene, "wa_frame_exec", text="Frame change")
        #row.label(text=f"Frame {scene.frame_current_final:6.1f}")
        row.prop(scene, "wa_hide_viewport", text="Hide in VP")
        
        layout.operator("wrap.clear_params", icon='CANCEL')
       
    

class WAControlPanel(bpy.types.Panel):

    """User parameters to control animation"""
    bl_label        = "User parameters"
    bl_category     = "Wrap"
    #bl_idname       = "SCENE_PT_layout"
    bl_space_type   = 'VIEW_3D'
    bl_region_type  = 'UI'
    #bl_context      = "scene"

    def draw(self, context):
        layout = self.layout
        
        ctl = get_control_object()
        params = ui_param_list(ctl)
        
        def draw_group(prms):
            for pf in prms:
                name  = pf[0]
                fname = pf[1]
                
                if np.size(ctl[fname]) > 1:
                    box = layout.box()
                    box.label(text=name)
                    col = box.column()
                    col.prop(ctl,f'["{fname}"]',text = '')                
                else:
                    layout.prop(ctl,f'["{fname}"]', text=pf[0])
                    
        prms = params.get("_main")
        if prms is not None:
            draw_group(prms)
        
        for key,prms in params.items():
            if key != "_main":
                row = layout.row()
                row.label(text=key)
                draw_group(prms)


def menu_func(self, context):
    #self.layout.operator(AddMoebius.bl_idname, icon='MESH_ICOSPHERE')
    pass


def register():
    
    bpy.types.Scene.wa_frame_exec    = bpy.props.BoolProperty(description="Execute at frame change")
    bpy.types.Scene.wa_hide_viewport = bpy.props.BoolProperty(description="Hide in viewport when hiding render")
    
    bpy.utils.register_class(ClearParamsOperator)
    bpy.utils.register_class(SetupOperator)
    bpy.utils.register_class(ExecOperator)

    bpy.utils.register_class(WAMainPanel)
    bpy.utils.register_class(WAControlPanel)
   
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(engine_handler)
    
    # Enbsure control object is created
    get_control_object()


def unregister():
    bpy.app.handlers.frame_change_pre.remove(engine_handler)
    
    bpy.utils.unregister_class(WAMainPanel)
    bpy.utils.unregister_class(WAControlPanel)

    bpy.utils.unregister_class(ClearParamsOperator)
    bpy.utils.unregister_class(SetupOperator)
    bpy.utils.unregister_class(ExecOperator)


if __name__ == "__main__":
    register()


    
