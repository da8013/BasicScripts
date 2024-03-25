#-------------------------------#
# Author: Deepanshu Aggarwal
# Last Update: 1 Feb 2023
#
# -----*** Note ***-----
# - All the energy scales are in millielectronvolt [meV]
# - All the length scales are in nanometer [nm]
#-------------------------------#

"""
genFuncs (gf)
---

It provides the generic functions based on numpy and scipy.
"""

# importing the required libraries
import math
import numpy as np
from numpy import typing
from scipy import linalg as la, integrate, special
from scipy.spatial.transform import Rotation
from unittest import TestCase

# Tolerance for floats
TOL = 1e-8

# Defining a test object from TestCase class of unittest
test = TestCase()

## Defining various types for declaring arguments type or annotations
num = float|int # A number that can be a float or an int
ndarrd = typing.NDArray[np.int_|np.double] # A numpy ndarray with dtype=np.double
ndarrc = typing.NDArray[np.cdouble] # A numpy ndarray with dtype=np.double

##***** GENERAL FUNCTIONS *****

# Function (rotMat2dz) that generates a 2D rotation matrix about z-axis
def rotMat2dz(theta: num) -> ndarrd:
    """
    This function generates a 2D rotaton matrix about z-axis.
    Note: theta is in radians and is considered positive counterclockwise.
    """
    rotClass = Rotation.from_euler('z', theta, degrees=False)
    rotz2d = rotClass.as_matrix()[0:2,0:2]
    # returning the 2D rotation matrix about z-axis
    return rotz2d
#----------rotMat2dz-----------

# Function (rot2dz)
def rot2dz(theta: num|ndarrd, pointrot: ndarrd|list[num]|tuple[num, num],
           pointabt: ndarrd|list[num]|tuple[num, num] = (0,0)) -> ndarrd:
    """
    Rotates one (or more) 2D-vector(s) with one (or more) angle(s) about a point (x0,y0).

    params:
    ---
    theta : float| int | np.ndarray -- Angle(s) of rotation
    pointrot : tuple| list | np.ndarray -- vector(s) that has to be rotated
    pointabt : tuple | list | np.ndarray -- point of rotation
    """
    # Converting into numpy arrays
    theta = np.asarray_chkfinite(theta); pointrot = np.asarray_chkfinite(pointrot)
    pointabt = np.asarray_chkfinite(pointabt)

    # Defining new vectors w.r.t pointabt
    if pointrot.ndim == 1: # a 1D-array
        vecnew = np.hstack((pointrot - pointabt,0)) # A 1D-array with z-axis
    elif pointrot.ndim == 2: # a 2D-array
        vecnew = np.hstack((pointrot - pointabt, np.zeros((pointrot.shape[0],1))))
    else:
        raise ValueError("The no. of dimensions in pointrot is more than 2!")
    
    # Generating the rotation matrices for different angles in theta-array
    rotzClass = Rotation.from_euler('z', theta, degrees=False)
    # It may be only one 3*3 matrix or a set of 3*3 matrices equal to the number of elements in theta-array

    # Finding the number of dimensions in vecnew
    if vecnew.ndim == 1:
        vecnewRot = rotzClass.apply(vecnew)
        # Checking for norm equality
        if vecnewRot.ndim == 1:
            chknorm = np.abs(np.subtract(la.norm(vecnewRot),la.norm(vecnew))) < TOL
        else:
            chknorm = np.all(np.subtract(la.norm(vecnewRot,axis=1), la.norm(vecnew)) < TOL)
    else:
        if theta.size == 1: # it means there is only one 3*3 matrix in rotzClass
            vecnewRot = rotzClass.apply(vecnew)
            # Checking for norm equality
            chknorm = np.all(np.subtract(la.norm(vecnewRot,axis=1), la.norm(vecnew)) < TOL)
        else:
            raise NotImplementedError("Multiple vectors with multiple rotation is not implemented!")
    
    # testing for norm
    test.assertTrue(chknorm, msg="The norm of rotated and unrotated vecs are not same!")

    # Returning the array 'vecnewRot'
    if vecnewRot.ndim == 1:
        return vecnewRot[0:2]
    else:
        return vecnewRot[:,0:2]
#-----------rot2dz--------------

# Function for getting the real-space primitive lattice vectors
def realVec(lattConst: num, phi: num, theta: num = 0) -> tuple[ndarrd, ndarrd]:
    """
    Calculates the real-space primitive lattice vectors with lattice constant 'a'
    and angle 'phi' between a1 and a2.
    """
    # Direct-lattice primitive vectors (due to active rotation 'theta')
    a1 = rot2dz(theta, lattConst*np.array([1,0]))
    a2 = rot2dz(theta, lattConst*np.array([math.cos(phi), math.sin(phi)]))

    # Norm of input vectors
    isanormsame = abs(la.norm(a1) - la.norm(a2)) < TOL
    test.assertTrue(isanormsame, msg="The vectors a1 and a2 do not have equal norm!")

    # returning the vectors
    return (a1, a2)

# Function (recVec) - generates the reciprocal lattice vectors
def recVec(a1: ndarrd|list[num]|tuple[num, num], 
           a2: ndarrd|list[num]|tuple[num, num]) -> tuple[ndarrd, ndarrd]:
    """
    Calculates the reciprocal lattice vectors corresponding to the given vectors a1 and a2.

    params:
    ---
    a1 : 2-tuple | list | 1D numpy array
    a2 : 2-tuple | list | 1D numpy array

    return:
    ---
    A 2-tuple (b1,b2) of reciprocal lattice vectors.
    """
    # Converting a1 and a2 into numpy arrays
    a1 = np.asarray_chkfinite(a1); a2 = np.asarray_chkfinite(a2)

    # Now creating a matrix [a1,a2] of the primitive lattice vectors
    realMat = np.column_stack((a1,a2))
    
    # Finding the reciprocal space lattice vectors
    # Note: la.inv raises LinAlgErr if realMat is singular. 
    reciMat = np.transpose((2*math.pi)*la.inv(realMat))
    
    # The reciprocal lattice vectors
    b1 = reciMat[:,0].copy(); b2 = reciMat[:,1].copy()

    # Test-1 (Laue Conditions) for reciprocal lattice vectors
    chkLaue1 = np.logical_and(np.dot(a1,b2) < TOL, np.dot(a2,b1) < TOL)
    chkLaue2 = np.logical_and(abs(np.dot(a1,b1) - 2*math.pi) < TOL,
                              abs(np.dot(a2,b2) - 2*math.pi) < TOL)
    test.assertTrue(chkLaue1 or chkLaue2,
                    msg="The reciprocal lattice vectors are violating Laue condiitons!")

    # Test-2 (Product of area should be (2*pi)**2
    Areal = abs(la.det(realMat)); Areci = abs(la.det(reciMat))
    test.assertTrue(abs(Areal*Areci - (2*np.pi)**2) < TOL, msg="The area test is failed!")

    # Test-3 (Equal Norm test)
    isbrnormsame = abs(la.norm(b1) - la.norm(b2)) < TOL
    test.assertTrue(isbrnormsame, msg="The vectors b1 and b2 do not have equal norm!")

    # Returning the vectors b1 and b2
    return (b1,b2)
#--------------recVec----------------
