#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:41:34 2021

@author: alain
"""

from math import degrees, radians
import numpy as np

import bpy
from mathutils import Quaternion

from .frames import get_frame
from .plural import to_shape


from .commons import wa_error_title
        
# =============================================================================================================================
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
 
    