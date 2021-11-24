# -*- coding: utf-8 -*-
"""
@author: Gast Arbeiter

This script is a Python implementation
  of some ideas of Prof. I.V. Kochikov's Dr.Sci thesis.
  
It reads the input data in ".mol" format of DISP and ElDiff programs.
Essential are geometry, hessian and Z3-matrix (3rd derivative of energy wrt Cartesian coordinates)

Please note that by default the input units are Hartree and Angstrom.
The geometry and matrixes are supposed to be Cartesian.

"""
# import chemcoord as cc
# from chemcoord.xyz_functions import get_rotation_matrix
import numpy as np
from scipy.linalg import eigh
# from pygsm.wrappers import Molecule
# from pygsm.utilities import *
# from coordinate_systems import CartesianCoordinates,Topology,PrimitiveInternalCoordinates,DelocalizedInternalCoordinates
import copy

# settings:

# boltzmann_constant = 1.380649e-23 / 4.359744722207185e-18
boltzmann_constant = 1.380649e-23 * 2.293712278396345e17 # in Hartree/K # NIST
#  1H = 4.359744722207185e-18 J # NIST
# also k=1.380649e-23 # J / K # NIST
cm_1_toHartree = 4.556335252912088e-6 # NIST 
Hartree_to_cm1 = 2.194746313632043e5 # NIST 
emass_to_Dalton = 5.48579909065e-4 # NIST 
Bohr_to_A = 0.529177210903 # NIST




class SvionJ():
    """
    self.coordinates is used when calculating the vibrational corrections
        (there are different procedures for Cartesian of Internal)
    """
    def __init__(self,
                 mol_file,
                 moleculename = 'das Molekül',
                 temperature=298,
                 Cartesian = True,
                 Sayvetz = True,
                 nonlinear=True,
                 display=1):
        self.name = moleculename
        self.Cartesian = Cartesian
        self.Sayvetz = Sayvetz
        if Cartesian:
            self.coordinates = 'Cartesian'
        else:
            self.coordinates = 'Internal'
        self.temperature = temperature
        self.display = display
        self.nonlinear = nonlinear
        if self.display > 0:
            print('\nMolecule name = ', moleculename, '\n',
                  'Temperature = ', temperature, 'K \n',
                  'Processing by ', self.coordinates, 'coordinates \n')
        self.xyz, self.masses, self.atom_symbols = read_cartesian_from_mol(mol_file)
        self.number_of_atoms = np.shape(self.xyz)[0]
        self.masses = np.reshape(self.masses, (self.number_of_atoms))
        
        # shift to the center of mass:
        self.center_of_mass = (np.sum(np.reshape(copy.deepcopy(self.masses), (len(self.masses), 1)) *
                          self.xyz, axis=0)) / np.sum(self.masses)
        self.xyz -= self.center_of_mass
        self.hessian = read_hessian_from_mol(mol_file, self.number_of_atoms)
        
        self.Z3matrix = read_Z3_anharm_from_mol(mol_file, self.number_of_atoms)

        # @Test&Debug:
        # self.Z3matrix[:] = 0
        
        self.matrixMinv = np.column_stack((self.masses**-1,
                                           self.masses**-1,
                                           self.masses**-1))
        self.matrixMinv *= emass_to_Dalton
        self.matrixMinv = self.matrixMinv.flatten() * np.identity(np.size(self.matrixMinv))

        # matrixM is needed for tests        
        self.matrixM = np.column_stack((self.masses,
                                        self.masses,
                                        self.masses))
        self.matrixM /= emass_to_Dalton
        self.matrixM = self.matrixM.flatten() * np.identity(np.size(self.matrixM))

        # # Here is the use of internal coordinates, by now disabled
        # if not self.Cartesian:
        #     ELEMENT_TABLE = elements.ElementData()
        #     self.atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in self.atom_symbols]
        #     self.top = Topology.build_topology(
        #                 self.xyz,
        #                 self.atoms,
        #                 )
        #     # nifty.printcool("Building Primitive Internal Coordinates")
        #     self.primitive_internal_coordinates = PrimitiveInternalCoordinates.from_options(
        #                 xyz=self.xyz,
        #                 atoms=self.atoms,
        #                 addtr=False,  # Add TRIC (translation and rotation)
        #                 topology=self.top,
        #                 )
        #     self.matrixB = self.primitive_internal_coordinates.wilsonB(self.xyz)
        #     self.matrixB = matrix2numpy(self.matrixB)
        #
        #     # matrix G:
        #     # Bunker-Yansen (10-116)
        #     # G = B * Minv * B.T
        #     self.matrixG = np.linalg.multi_dot([self.matrixB,
        #                                         self.matrixMinv,
        #                                         self.matrixB.T])
        #     self.matrixG_inv = np.linalg.pinv(self.matrixG, 1e-5)
        #    
        #     # # A = Minv B.T Ginv
        #     # # book, p.74
        #     self.matrixA = np.linalg.multi_dot([self.matrixMinv,
        #                                         self.matrixB.T,
        #                                         self.matrixG_inv])
        #
        #     # Hessian in internal coordinates should be
        #     #   A.T * F * A
        #     self.hessian_int = np.linalg.multi_dot([self.matrixA.T,
        #                                             self.hessian,
        #                                             self.matrixA])
        #
        #     # self.frequencies_by_internal_coordinates, self.vibrational_forms_by_internal_coordinates = eigh(np.dot(self.hessian_int, self.matrixG))
        #     self.frequencies_by_internal_coordinates, self.vibrational_forms_by_internal_coordinates = eigh(np.dot(self.hessian_int, self.matrixG))
        #     # frequencies_ipygsm = np.real(np.lib.scimath.sqrt(frequencies_ipygsm))
        #     self.frequencies_by_internal_coordinates = np.lib.scimath.sqrt(self.frequencies_by_internal_coordinates)
        #
        #     #1 Hartree = 2.194746313632043e5 cm-1
        #     self.frequencies_by_internal_coordinates *= Hartree_to_cm1
        #     self.vibrational_forms_by_internal_coordinates_in_cartesians = np.dot(self.matrixA, self.vibrational_forms_by_internal_coordinates)
        #    
        #     if self.display > 0:
        #         with np.printoptions(precision=1, suppress=True):
        #             print('frequencies by internal coordinates = \n', self.frequencies_by_internal_coordinates)
        #
        #     self.default_vibrational_forms = np.dot(self.matrixA, self.vibrational_forms_by_internal_coordinates)
        #     self.default_frequencies = self.frequencies_by_internal_coordinates

        
        self.mw_hessian = np.linalg.multi_dot([self.matrixMinv**0.5, self.hessian, self.matrixMinv**0.5])
        # good:
        self.frequencies_by_cartesians, self.vibrational_forms_by_cartesians = eigh(self.mw_hessian)
        self.frequencies_by_cartesians = np.real(np.lib.scimath.sqrt(self.frequencies_by_cartesians))
        self.frequencies_by_cartesians *= Hartree_to_cm1
        
        self.reduced_masses_by_cartesians = np.zeros_like(self.frequencies_by_cartesians)
        for i in range(len(self.reduced_masses_by_cartesians)):
            self.reduced_masses_by_cartesians = 1/np.linalg.norm(self.vibrational_forms_by_cartesians[:,i])**2 # * emass_to_Dalton
        # reduced masses are in amu !
        #   To get it in Dalton, multipy it by emass_to_Dalton
        
        # @Test&Debug: (B-Y eq. 10-128)
        # np.dot(L, L.T) = G
        # G = np.dot(self.vibrational_forms_by_cartesians, self.vibrational_forms_by_cartesians.T)
        # print('\n @Test&Debug:')
        # print('matrix G from 10-128 = \n', G)
        # checked - yes, it's a diagonal matrix with 1s
        
        # At this step, mass-weighted displacements the normalization is
            # np.dot(self.vibrational_forms_by_cartesians, self.vibrational_forms_by_cartesians.T)
                # is the unity matrix.
        # Next step this will be lost.
        # Un-mass-weigh the displacements:
            # We get matrix A from this (in terms of Disser, p.185)
        # B-Y eq. 10-145 p.246
        self.vibrational_forms_by_cartesians = np.dot(self.matrixMinv**0.5, self.vibrational_forms_by_cartesians)

        
        if self.Sayvetz:
            _ = self.vibrational_analysis_with_Sayvetz()
            if self.nonlinear:
                self.default_vibrational_forms = self.vibrational_forms_by_Sayvetz[:,6:]
                self.default_frequencies = self.frequencies_Sayvetz[6:]
            else:
                self.default_vibrational_forms = self.vibrational_forms_by_Sayvetz[:,5:]
                self.default_frequencies = self.frequencies_Sayvetz[5:]
        else:
            self.default_vibrational_forms = self.vibrational_forms_by_cartesians
            self.default_frequencies = self.frequencies_by_cartesians
            if self.display > 0:
                with np.printoptions(precision=1, suppress=True):
                    print('frequencies by cartesian coordinates = \n', self.frequencies_by_cartesians)
                
        # @Test&Debug:
        # How to check that the vibrational forms are un-mass-weighted?
        #   The Cartesian displacements multiplied by masses
        #      should not displace the center of mass.
        #   The following thing gatta be near 0:
        # np.sum((np.dot(self.vibrational_forms_by_cartesians[:,-1], self.matrixM))) / np.linalg.norm(self.vibrational_forms_by_cartesians)
        # 

        

        self.number_of_normal_modes = np.shape(self.default_vibrational_forms)[1]
        self.atom_pairs = self.find_atom_pairs()
        self.Q2 = self.Q2_av()
        self.Q1 = self.Q1_av()
        self.QQQ = self.QkQlQm()
        
    def find_atom_pairs(self):
        """ returns a list of atomic pairs
            (atom numbers) in an order of increasing interatomic distances
                length: number_of_atoms * (number_of_atoms-1) / 2
            list structure of each pair:
                atom1#, atom2#, atom1symb, atom2symb, distance
        """
        atom_pairs = []
        for i in range(self.number_of_atoms):
            for j in range(i+1, self.number_of_atoms):
                R_ij = self.xyz[i,:] - self.xyz[j,:]
                r_e = np.linalg.norm(R_ij)
                atom_pairs.append([i, j,
                                     self.atom_symbols[i], self.atom_symbols[j],
                                     r_e])
        # sort by distance:
        atom_pairs = sorted(atom_pairs, key=lambda distance: distance[-1])
        return atom_pairs
    
    def h_and_H(self): # - rewrite ??? Should matrix A be different ?
        """ calculate h and H, see disser, p.137, top equations
        
        The vibrational forms should be cartesian displacements
        
        Returns h and H parameters for all normal modes,
           as a list with len=number_of_normal_modes
        
        ???
        about Dimensions:
            ??? default_vibrational_forms have the dimension of [Angstrom]
        """

        h_and_H = []
        for atom_pair in self.atom_pairs:
            # 1) find A matix for a given pair of atoms (i,j)
            A_ij = np.zeros((3, self.number_of_normal_modes))
            for normalmode in np.arange(self.number_of_normal_modes):
                A_i = self.default_vibrational_forms[3*atom_pair[0]: 3*atom_pair[0]+3, normalmode]
                A_j = self.default_vibrational_forms[3*atom_pair[1]: 3*atom_pair[1]+3, normalmode]
                A_ij[:,normalmode] = A_i - A_j
        
            # 2) find unit vector in direction between the 2 atoms and r_e:
            R_ij = self.xyz[atom_pair[0],:] - self.xyz[atom_pair[1],:]
            R_ij = R_ij.T
            r_e = atom_pair[-1]
            e_ij = R_ij / r_e # should be a unit vector
        
            # 3) calculate h and H, see disser, p.137, top equations
            h = np.zeros((self.number_of_normal_modes))
            H = np.zeros((self.number_of_normal_modes, self.number_of_normal_modes))
            
            for normalmode1 in np.arange(self.number_of_normal_modes):
                h[normalmode1] = np.dot(e_ij, A_ij[:, normalmode1]) / r_e
                for normalmode2 in np.arange(self.number_of_normal_modes):
                    H[normalmode1, normalmode2] = np.dot(A_ij[:, normalmode1], A_ij[:, normalmode2]) / r_e**2
            h_and_H.append([h, H])
        
        return h_and_H
    

    def Q2_av(self):
        """ TheDisser, p.136.
            Temperature in K 
            Planck constant = 1 in a.u.,
            therefore wavenumbers should be converted from cm-1 to Hartree.
        
            Returns <Q2> in units of Bohr**2 * emass
                """
            
        x = self.default_frequencies * cm_1_toHartree / (2 * boltzmann_constant * self.temperature)
        # here boltzmann_constant is in Hartree

        with np.errstate(divide='ignore', invalid='ignore'): 
            cothang = 1 / np.tanh(x)
            Q2_av = (1 / (2 * self.default_frequencies * cm_1_toHartree)) * cothang
        
        # units of <Q2> at this point should be in Bohr**2 * emass

        if not self.Sayvetz:
            if self.Cartesian:
                if self.nonlinear:
                    Q2_av[0:6] = 0
                else:
                    Q2_av[0:5] = 0

        if self.display > 1:
            with np.printoptions(precision=5, suppress=True):
                # print('sqrt(Q2_av) = \n', Q2_av**0.5)
                print('sqrt(Q2_av) (converted to A*Dalton**0.5) = \n', (Q2_av**0.5) *(Bohr_to_A**2 * emass_to_Dalton)**0.5)
        return Q2_av
    
    def Q1_av(self):
        """ 
            eq. 3.41 on page 136 Disser
            Q1_av is in Bohr * emass**0.5 !!!
            """

        delta_kll = np.zeros([self.number_of_normal_modes,
                              self.number_of_normal_modes])
        for k in range(self.number_of_normal_modes):
            for l in range(self.number_of_normal_modes):
                delta_kll[k, l] = self.delta_function_341_Cart_with_numerical_displacement(k, l, l)
        _q2av = np.reshape(copy.deepcopy(self.Q2), (1, self.number_of_normal_modes))

        Q1_av = _q2av * delta_kll 
        with np.errstate(divide='ignore', invalid='ignore'): 
            Q1_av = - np.sum(Q1_av, axis=1) / (2 * (self.default_frequencies*cm_1_toHartree)**2)
        if not self.Sayvetz:
            if self.Cartesian:
                if self.nonlinear:
                   Q1_av [0:6] = 0
                else:
                   Q1_av [0:5] = 0
        if self.display > 1:
            with np.printoptions(precision=5, suppress=True):
                print('Q1_av (converted to A*Dalton**0.5) = \n', Q1_av *(Bohr_to_A**2 * emass_to_Dalton)**0.5)

        return Q1_av

    
    def QkQlQm(self):
        """ see also eq. 3.41 on page 136 Disser

        The units here are Bohr and emass (not Dalton !)
         """
        QQQ = self.X_klm()
        # Q2 = self.Q2
        # Q1 = self.Q1
        for k in range(self.number_of_normal_modes):
            for l in range(self.number_of_normal_modes):
                for m in range(self.number_of_normal_modes):
                    QQQ[k, l, m] *= self.delta_function_341_Cart_with_numerical_displacement(k, l, m)
                    if k==l==m:
                        # QQQ[k, l, m] += (3 * self.Q2[k] *  self.Q1[k]
                        #                   / (Bohr_to_A**2 * emass_to_Dalton)**3 )
                        QQQ[k, l, m] += 3 * self.Q2[k] *  self.Q1[k]
                    elif k==l:
                        # QQQ[k, l, m] += (self.Q2[k] *  self.Q1[m]
                        #                   / (Bohr_to_A**2 * emass_to_Dalton)**3 )
                        QQQ[k, l, m] += self.Q2[k] *  self.Q1[m]
                    elif k==m:
                        # QQQ[k, l, m] += (self.Q2[k] *  self.Q1[l]
                        #                   / (Bohr_to_A**2 * emass_to_Dalton)**3 )
                        QQQ[k, l, m] += self.Q2[k] *  self.Q1[l]
                    elif l==m:
                        # QQQ[k, l, m] +=  (self.Q2[l] * self.Q1[k]
                        #                   / (Bohr_to_A**2 * emass_to_Dalton)**3 )
                        QQQ[k, l, m] +=  self.Q2[l] * self.Q1[k]
        if not self.Sayvetz:
            if self.Cartesian:
                if self.nonlinear:
                    QQQ[0:6, :, :] = 0
                    QQQ[:, 0:6, :] = 0
                    QQQ[:, :, 0:6] = 0
                else:
                    QQQ[0:5, :, :] = 0
                    QQQ[:, 0:5, :] = 0
                    QQQ[:, :, 0:5] = 0

        qqq_diagonal = np.zeros(self.number_of_normal_modes)
        for q in range(self.number_of_normal_modes):
            qqq_diagonal[q] = QQQ[q, q, q]
        if self.display > 1:
            with np.printoptions(precision=5, suppress=True):
                print('QQQ_diagonal (converted to (A*Dalton**0.5)**3) = \n',
                      qqq_diagonal *(Bohr_to_A**2 * emass_to_Dalton)**1.5)
        return QQQ


    def X_klm(self):
        """ equation 3.41 of Kochikov's Disser (p.136) 
            frequencies are in cm^-1  
            Also eq.36 from Novosadov 2004 (in Russian)
            
            Achtung: eq.3.41 for Xklm has an typo in 'sh(zk+zl-zl)'
            
            One more thing: the equation in K4k Disser seems to have a wrong sign (there should be minus before it !!111)
            
            """

        X_klm = np.zeros([self.number_of_normal_modes,
                          self.number_of_normal_modes,
                          self.number_of_normal_modes])

        _freq_k = np.reshape(copy.deepcopy(self.default_frequencies),
                             (self.number_of_normal_modes, 1, 1))
        _freq_l = np.reshape(copy.deepcopy(self.default_frequencies),
                             (1, self.number_of_normal_modes, 1))
        _freq_m = np.reshape(copy.deepcopy(self.default_frequencies),
                             (1, 1, self.number_of_normal_modes))
        _freq_k *= cm_1_toHartree 
        _freq_l *= cm_1_toHartree 
        _freq_m *= cm_1_toHartree 
    
        z_k = _freq_k / (2 * boltzmann_constant * self.temperature)
        z_l = _freq_l / (2 * boltzmann_constant * self.temperature)
        z_m = _freq_m / (2 * boltzmann_constant * self.temperature)

        with np.errstate(divide='ignore', invalid='ignore'):        
            X_klm = (
                (1 / (32 * boltzmann_constant * self.temperature)) *
                (1 / (_freq_k * _freq_l * _freq_m)) *
                (1 / (np.sinh(z_k) * np.sinh(z_l) * np.sinh(z_m))) *
                (np.sinh(z_k + z_l + z_m) / (z_k + z_l + z_m) + 
                 np.sinh(z_k - z_l + z_m) / (z_k - z_l + z_m) +
                 np.sinh(z_k + z_l - z_m) / (z_k + z_l - z_m) +
                 np.sinh(z_k - z_l - z_m) / (z_k - z_l - z_m) ) )
        if not self.Sayvetz:
            if self.Cartesian:
                if self.nonlinear:
                    X_klm[0:6, :, :] = 0
                    X_klm[:, 0:6, :] = 0
                    X_klm[:, :, 0:6] = 0
                else:
                    X_klm[0:5, :, :] = 0
                    X_klm[:, 0:5, :] = 0
                    X_klm[:, :, 0:5] = 0

        return -X_klm

    
    def delta_function_341_Cart_with_numerical_displacement(self,
                                                            i1, i2, i3):
        """ Calculate the 'delta_pqr' tensor on p. 136 and p.184 (Disser)
        i1, i2, i3 - p,q,r - indexes of normal modes for the delta_pqr ^tensor^
           in eq. 4.14 on p.184  """

        # numerical differentiation in curvilinear coordinates is disabled:
        # displacement_in_Q = np.zeros([3])
        # displacement_in_Q[0] = Q2_average[i1]**0.5
        # displacement_in_Q[1] = Q2_average[i2]**0.5
        # displacement_in_Q[2] = Q2_average[i3]**0.5
        # # make it half
        # displacement_in_Q /= 2
        
        # calculate sigmas with displacement
        displacement_i1 = self.default_vibrational_forms[:, i1]# * displacement_in_Q[0]
        displacement_i2 = self.default_vibrational_forms[:, i2]# * displacement_in_Q[1]
        displacement_i3 = self.default_vibrational_forms[:, i3]# * displacement_in_Q[2]

        with np.errstate(divide='ignore', invalid='ignore'):         
            sigma_i1 = displacement_i1# / displacement_in_Q[0]
            sigma_i2 = displacement_i2# / displacement_in_Q[1]
            sigma_i3 = displacement_i3# / displacement_in_Q[2]

    
        delta_123 = (self.Z3matrix *
                 np.reshape(sigma_i1, (3*self.number_of_atoms, 1, 1)) *
                 np.reshape(sigma_i2, (1, 3*self.number_of_atoms, 1)) *
                 np.reshape(sigma_i3, (1, 1, 3*self.number_of_atoms)) )
    
        
        # the second term:
        # tau_12, tau_13, tau_23 should be zero for normal coordinates in Cartesians
        return np.sum(delta_123)


    def greeks(self):
        """ Mostly from p.137 of K4k Disser"""
        
        datscaling = Bohr_to_A**2
        h_n_H = self.h_and_H()
        greeks = []
        for i in range(len(self.atom_pairs)):
            h = h_n_H[i][0]
            H = h_n_H[i][1]
            hi = np.sum(h * self.Q1) *datscaling**0.5
            fi = np.sum(h**2 * self.Q2) *datscaling**1
            psi = np.sum(np.diag(H) * self.Q2) *datscaling**1
            lam = np.sum(
                    np.reshape(copy.deepcopy(h), (self.number_of_normal_modes, 1, 1)) *
                    np.reshape(copy.deepcopy(H), (1, self.number_of_normal_modes, self.number_of_normal_modes)) *
                    self.QQQ ) *datscaling**1.5
            mu = np.sum(
                    np.reshape(copy.deepcopy(h), (self.number_of_normal_modes, 1, 1)) *
                    np.reshape(copy.deepcopy(h), (1, self.number_of_normal_modes, 1)) *
                    np.reshape(copy.deepcopy(h), (1, 1, self.number_of_normal_modes)) *
                    self.QQQ ) *datscaling**1.5
            sigma = fi * psi + 2 * np.sum(
                np.reshape(copy.deepcopy(h), (self.number_of_normal_modes, 1)) *
                np.reshape(copy.deepcopy(h), (1, self.number_of_normal_modes)) *
                H *
                np.reshape(copy.deepcopy(self.Q2), (self.number_of_normal_modes, 1)) *
                np.reshape(copy.deepcopy(self.Q2), (1, self.number_of_normal_modes))) *datscaling**2
            ro = psi**2 + 2 * np.sum(
                H**2 *
                np.reshape(copy.deepcopy(self.Q2), (self.number_of_normal_modes, 1)) *
                np.reshape(copy.deepcopy(self.Q2), (1, self.number_of_normal_modes))) *datscaling**2
            tau = 3 * fi**2 # page 85 Disser
            greeks.append([hi, fi, psi, lam, mu, sigma, ro, tau])
        return greeks

    def cumulants_distance_corrections_elkappa(self):
        """ see K4k disser, p.86
        el_and_kappa page 135, eq. 3.40
            """
        greeks = self.greeks()
        cumulants = []
        distance_corrections = []
        el_kappa = []
        
        if self.display>0:
            print('\n  Re,       Ra,       el,       kappa*1e6')
            
        for i in range(len(self.atom_pairs)):
            re = self.atom_pairs[i][-1]
            hi, fi, psi, lam, mu, sigma, ro, tau = greeks[i]
            c1 = re * (hi +
                       (psi-fi)/2 +
                       (mu-lam)/2 -
                       (ro-6*sigma+5*tau)/8)
            c2 = re**2 * (fi + lam - mu +
                          (ro-6*sigma+5*tau)/4 -
                          (psi-fi)**2/4 -
                          hi*(psi-fi))
            c3 = re**3 * (mu - 3*hi*fi +
                          1.5*(sigma-fi*psi) -
                          1.5*(tau-fi**2))
            
            # # @Test&Debug:
            # l_square = re**2 * (
            #     fi + lam - 2*mu +
            #     (ro-12*sigma+15*tau)/4 -
            #     (psi-3*fi)*(psi-5*fi)/4-
            #     hi*(psi-4*fi)) # ENDOF @Test&Debug
            
            rg = re * ( 1 + hi + (psi-fi)/2 +
                       (mu-lam)/2 -
                       (ro-6*sigma+5*tau)/8)
            ra = re * (1 + hi + (psi-3*fi)/2 +
                       (5*mu-3*lam)/2 -
                       (3*ro-30*sigma+35*tau)/8 +
                        (psi-3*fi)**2/4 + # this term is absent at Tarasov book, eq. Б.1, p. 562
                       hi*(psi-3*fi))
            # # @Test&Debug:
            # ra_c = re * (1 + hi + (psi-3*fi)/2 +  # Tarasoff
            #            (5*mu-3*lam)/2 -
            #            (3*ro-30*sigma+35*tau)/8 +
            #            hi*(psi-3*fi))
            # rg_c = ra_c + l_square/re # ENDOF @Test&Debug # Tarasoff
            
            lsquare = re**2 * (
                fi + lam - 2*mu +
                (ro-12*sigma+15*tau)/4 -
                (psi-3*fi)*(psi-5*fi)/4 -
                hi*(psi-4*fi))
            el=lsquare**0.5
            
            kappa = c3/6
            # kappa = (1/6) * re**3 * (
            #     mu - 3*hi*fi +
            #     (3*sigma-5*tau)/2 -
            #     1.5*fi*(psi-3*fi) )
            
            cumulants.append([c1, c2, c3])
            distance_corrections.append([re, rg, ra])
            el_kappa.append([el, kappa])
            if self.display>0:
                # with np.set_printoptions(formatter={'float': '{: 0.4f}'.format}):
                print('{:7.4f}   {:7.4f}   {:7.4f}   {:8.4f}'.format(re,  ra,   el,   kappa*1e6))
        
        return cumulants, distance_corrections, el_kappa
    

    
    def inertia_tensor(self):
        """ units: A**2 * Dalton """
        I_nertia_tensor = np.zeros((3,3))
        I_nertia_tensor[0,0] = np.sum(self.masses * (self.xyz[:,1]**2 + self.xyz[:,2]**2))
        I_nertia_tensor[0,1] = -np.sum(self.masses * self.xyz[:,0]*self.xyz[:,1])
        I_nertia_tensor[0,2] = -np.sum(self.masses * self.xyz[:,0]*self.xyz[:,2])
        I_nertia_tensor[1,0] = -np.sum(self.masses * self.xyz[:,1]*self.xyz[:,0])
        I_nertia_tensor[1,1] = np.sum(self.masses * (self.xyz[:,0]**2 + self.xyz[:,2]**2))
        I_nertia_tensor[1,2] = -np.sum(self.masses * self.xyz[:,1]*self.xyz[:,2])
        I_nertia_tensor[2,0] = -np.sum(self.masses * self.xyz[:,2]*self.xyz[:,0])
        I_nertia_tensor[2,1] = -np.sum(self.masses * self.xyz[:,2]*self.xyz[:,1])
        I_nertia_tensor[2,2] = np.sum(self.masses * (self.xyz[:,0]**2 + self.xyz[:,1]**2))
        return I_nertia_tensor
    
    def cartesian_internal_coordinates_with_Sayvetz(self):
        """ units: Dalton, Angstrom """
        matrixD = np.zeros((3*self.number_of_atoms, 6)) # 3*self.number_of_atoms))
        matrixD[:,0] = np.reshape(
            (self.masses,
             np.zeros(self.number_of_atoms),
             np.zeros(self.number_of_atoms)),
            (3*self.number_of_atoms), order='F')
        matrixD[:,1] = np.reshape(
            (np.zeros(self.number_of_atoms),
             self.masses,
             np.zeros(self.number_of_atoms)),
            (3*self.number_of_atoms), order='F')
        matrixD[:,2] = np.reshape(
            (np.zeros(self.number_of_atoms),
             np.zeros(self.number_of_atoms),
             self.masses),
            (3*self.number_of_atoms), order='F')
        
        matrixX = eigh(self.inertia_tensor())[1]
        Px = np.sum(matrixX[0,:] * self.xyz, axis=1)
        Py = np.sum(matrixX[1,:] * self.xyz, axis=1)
        Pz = np.sum(matrixX[2,:] * self.xyz, axis=1)
        
        D4x = (Py * matrixX[0,2] - Pz * matrixX[0,1]) * self.masses**0.5
        D4y = (Py * matrixX[1,2] - Pz * matrixX[1,1]) * self.masses**0.5
        D4z = (Py * matrixX[2,2] - Pz * matrixX[2,1]) * self.masses**0.5
        
        D5x = (Pz * matrixX[0,0] - Px * matrixX[0,2]) * self.masses**0.5
        D5y = (Pz * matrixX[1,0] - Px * matrixX[1,2]) * self.masses**0.5
        D5z = (Pz * matrixX[2,0] - Px * matrixX[2,2]) * self.masses**0.5
        
        D6x = (Px * matrixX[0,1] - Py * matrixX[1,0]) * self.masses**0.5
        D6y = (Px * matrixX[1,1] - Py * matrixX[1,0]) * self.masses**0.5
        D6z = (Px * matrixX[2,1] - Py * matrixX[2,0]) * self.masses**0.5
        
        matrixD[:,3] = np.reshape((D4x, D4y, D4z), (3*self.number_of_atoms), order='F')
        matrixD[:,4] = np.reshape((D5x, D5y, D5z), (3*self.number_of_atoms), order='F')
        matrixD[:,5] = np.reshape((D6x, D6y, D6z), (3*self.number_of_atoms), order='F')
        mean_norm_of_rotationals = (np.linalg.norm(matrixD[:,3]) + np.linalg.norm(matrixD[:,4]) + np.linalg.norm(matrixD[:,5])) / 3
        if np.linalg.norm(matrixD[:,3]) < 0.01*mean_norm_of_rotationals:
            matrixD = np.delete(matrixD, 3, 1)
            self.nonlinear = False
            print("""\n >>> achtung <<< \n\n the molecule is linear? \n Please check whether that's what you expect""")
        elif np.linalg.norm(matrixD[:,4]) < 0.01*mean_norm_of_rotationals:
            matrixD = np.delete(matrixD, 4, 1)
            self.nonlinear = False
            print("""\n >>> achtung <<< \n\n the molecule is linear? \n Please check whether that's what you expect""")
        elif np.linalg.norm(matrixD[:,5]) < 0.01*mean_norm_of_rotationals:
            matrixD = np.delete(matrixD, 5, 1)
            self.nonlinear = False
            print("""\n >>> achtung <<< \n\n the molecule is linear? \n Please check whether that's what you expect""")
        else:
            print("""the molecule is nonlinear? Please check whether that's what you expect""")
        
        # normalize:
        for i in range(np.shape(matrixD)[1]):
            matrixD[:,i] /= np.linalg.norm(matrixD[:,i])
        matrixD_big = np.random.rand(3*self.number_of_atoms, 3*self.number_of_atoms)
        matrixD_big[:, 0:np.shape(matrixD)[1]] = matrixD
        matrixD_big = gram_schmidt(matrixD_big.T)
        
        return matrixD_big.T

    def vibrational_analysis_with_Sayvetz(self):
        matrixD = self.cartesian_internal_coordinates_with_Sayvetz()
        self.hessian_iSayvetz = np.linalg.multi_dot([matrixD.T, self.mw_hessian, matrixD])
        self.frequencies_Sayvetz, self.vibrational_forms_by_Sayvetz = eigh(self.hessian_iSayvetz)
        self.frequencies_Sayvetz = np.real(np.lib.scimath.sqrt(self.frequencies_Sayvetz))
        self.frequencies_Sayvetz *= Hartree_to_cm1
        if self.display > 0:
            with np.printoptions(precision=1, suppress=True):
                if self.nonlinear:
                    f0=6
                else:
                    f0=5
                print('\n frequencies by Sayvetz')
                print('translational and rotational:', self.frequencies_Sayvetz[0:f0:])
                print('vibrational: \n', self.frequencies_Sayvetz[f0:])
        self.vibrational_forms_by_Sayvetz = np.linalg.multi_dot([self.matrixMinv**0.5, matrixD, self.vibrational_forms_by_Sayvetz])
        self.reduced_masses_by_Sayvetz = np.zeros_like(self.frequencies_Sayvetz)
        for i in range(len(self.reduced_masses_by_Sayvetz)):
            self.reduced_masses_by_Sayvetz = 1/np.linalg.norm(self.vibrational_forms_by_Sayvetz[:,i])**2 # * emass_to_Dalton
        # reduced_masses_by_Sayvetz are in [emass] !



def read_cartesian_from_mol(mol_file):
    with open(mol_file) as input_data:
        count=0
        
        # locate the beginning of atomic coordinates (cartesian):
        while True:
            line = input_data.readline()
            count += 1
            if line.lower().lstrip().startswith('[atoms]') == True:
                # print('line number', count, ', atoms !!')
                break
            if not line:
                print('where are the atomic coordinates? ([Atoms]) ??')
                break
        
        # read the number of atoms:
        while True:
            line = input_data.readline()
            count += 1
            if line.lower().lstrip().startswith('count') == True:
                number_of_atoms = line.lower().replace('count', '').replace('=', '')
                number_of_atoms = number_of_atoms.strip()
                number_of_atoms = int(number_of_atoms)
                print('number of atoms = ', number_of_atoms)
                break
            if line.strip() == '':
                raise SystemExit('number of atoms not found')
        cartesian_coordinates = np.zeros((number_of_atoms, 3))
        atomic_masses = np.zeros((number_of_atoms, 1))
        atomic_labels = np.empty(number_of_atoms, dtype='object')
        # read cartesian coordinates
        # not it's assumed that they are in Angstrom
        atom_no = 0
        while True:
            line = input_data.readline()
            count += 1
            currentline = line.split()

            if line.strip() == '':
                break
            elif (currentline[0]).isnumeric() == False:
                # print('count = ', count, ' ', currentline)
                continue
            else:
                # print('currentline[1:4]', currentline[1:4])

                cartesian_coordinates[atom_no, :] = np.asarray(currentline[1:4])
                atomic_labels[atom_no] = currentline[4]
                atomic_masses[atom_no] = np.asarray(currentline[5])
                atom_no += 1
            
        return cartesian_coordinates, atomic_masses, atomic_labels


def read_hessian_from_mol(filename, number_of_atoms):
    """ here it's assumed that the units are Angstrom and Hartree """
    
    hessian = np.zeros((3 * number_of_atoms, 3 * number_of_atoms))
    
    with open(filename) as input_data:
        
        # locate the beginning of hessian:
        while True:
            line = input_data.readline()
            if line.lower().strip().startswith('[matrix z]') == True:
                # skip two lines
                _ = input_data.readline()
                _ = input_data.readline()
                break
            if not line:
                print('where is the hessian? ([Matrix Z]) ??')
                break
        
        # read hessian:
        
        block_height = 3 * number_of_atoms
        current_column = 0
        starting_row = 0
        current_row = 0
        while True:

            line = input_data.readline()
            currentline = line.split()
            # print('currentline 1 = ', currentline)
            if line.strip() == '':
                break
            # elif (currentline[0]).isnumeric() == False:
            #     continue
            else:
                block_width = 0
                current_row = starting_row
                for i in np.arange(block_height):
                    # read current block line-by-line: 
                    # print('location = ', current_row, current_column)
                    # print('isgonna write', currentline[1:])
                    block_width = len(currentline) - 1 # because first goes some label
                    hessian[current_row, current_column:current_column+block_width] = np.asarray(currentline[1:])
                    current_row += 1

                    line = input_data.readline()
                    currentline = line.split()
                    # print('currentline 2 = ', currentline)
                # print('block_width = ', block_width)
                starting_row += block_width
                block_height -= block_width
                current_column += block_width

    # reconstruct the full matrix:
    hessian += hessian.T - np.diag(np.diag(hessian))
    
    return hessian


def read_Z3_anharm_from_mol(filename, number_of_atoms):
    """ Reads the third derivatives of potential energy wrt cartesian coordinates.
    Here it's assumed that the units are Angstrom and Hartree 
    """
    
    Z3matrix = np.zeros((3 * number_of_atoms, 3 * number_of_atoms, 3 * number_of_atoms))
    
    with open(filename) as input_data:
        
        # locate the beginning of Z3 matrix:
        while True:
            line = input_data.readline()
            if line.lower().strip().startswith('[matrix z3]') == True:
                break
            if not line:
                print('where is the Z3matrix? ( [Matrix Z3] ) ??')
                return
                break
        
        # locate the 'block' string:
        while True:
            line = input_data.readline()
            if line.lower().strip().startswith('block') == True:
                break
        
        # read Z3:
        for blocknumber in range(number_of_atoms*3):
            # current block would be the square matrix at (i, :, :)
            #   of the size blocknumber*blocknumber
            block_height = blocknumber + 1 # because it start with 0
            current_column = 0
            starting_row = 0
            current_row = 0
            
            while True:
                line = input_data.readline() 
                currentline = line.split()
                if line.strip() == '':
                    break
                # elif (currentline[0]).isnumeric() == False:
                #     break
                elif (currentline[0]).casefold().startswith('block') == True:
                    break
                else:
                    block_width = 0
                    current_row = starting_row
                    for i in np.arange(block_height):
                        # read current block line-by-line: 
                        # print('location = ', blocknumber, current_row, current_column)
                        # print('isgonna write', currentline[1:])
                        block_width = len(currentline) - 1 # because first goes some number
                        Z3matrix[blocknumber, current_row, current_column:current_column+block_width] = np.asarray(currentline[1:])
                        current_row += 1

                        line = input_data.readline()
                        currentline = line.split()
                    starting_row += block_width
                    block_height -= block_width
                    current_column += block_width

    # fill the remaining elements
    # the Z3 matrix is read in i,j,k sequence
    
    for i in range(number_of_atoms*3):
        for j in range(i, number_of_atoms*3):
            for k in range(j, number_of_atoms*3):
                    Z3matrix[i, j, k] = Z3matrix[k, j, i]
                    Z3matrix[i, k, j] = Z3matrix[k, j, i]
                    Z3matrix[j, k, i] = Z3matrix[k, j, i]
                    Z3matrix[j, i, k] = Z3matrix[k, j, i]
                    Z3matrix[k, i, j] = Z3matrix[k, j, i]

    return Z3matrix



# def matrix2numpy(datmatrix):
#     """
#         The function for processing in internal coordinates through pygsm.
#         Converts the matrix from the pygsm format to numpy """
#     if type(datmatrix) == np.ndarray:
#         return datmatrix
#     else:
#         a, b = datmatrix.shape
#         dat_I = np.identity(max(datmatrix.shape))
#         if a < b:
#             m2n = datmatrix.dot(datmatrix, dat_I)
#         else:
#             m2n = datmatrix.dot(dat_I, datmatrix)
#         return m2n

def gram_schmidt(vectors):
    """ Gram-Schmidt Orthogonization 
    Normalizes by default and assumes the vectors are in rows.
    If a vector is already there it is omitted from the normalization"""
    basis = []
    for v in vectors:
        w = v - sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)



if __name__ == '__main__':
    
    CHCl3 = SvionJ('eggzamplezz/chcl3_int.mol', 'CHCl3', Sayvetz=True)
    _ = CHCl3.cumulants_distance_corrections_elkappa()
   
    CS2 = SvionJ('eggzamplezz/cs2_cart.mol', 'CS2')
    _ = CS2.cumulants_distance_corrections_elkappa()


    # how to check that Xkkk works? :
    #  K4k disser, p.85, bottom equation
    # Xkkk = -((CHCl3.Q2**2 * CHCl3.default_frequencies**2 * cm_1_toHartree**2 /
    #           ((Bohr_to_A**2 * emass_to_Dalton)**2) - (1/6)) /
    #           (CHCl3.default_frequencies**4 * cm_1_toHartree**4) )
    # with np.errstate(divide='ignore', invalid='ignore'): 
    #     Xkkk = -((CHCl3.Q2**2 * CHCl3.default_frequencies**2 * cm_1_toHartree**2 /
    #           1 - (1/6)) /
    #           (CHCl3.default_frequencies**4 * cm_1_toHartree**4) )
    
    # xklm_c = CHCl3.X_klm()
    # # and the following should be 1:
    # Xkkk[-9] / xklm_c[-9, -9, -9]

