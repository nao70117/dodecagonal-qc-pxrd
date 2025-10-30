#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCdiff - Python library for Quasi-Crystal diffraction
# Copyright (c) 2022 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import numpy as np
import math

PI = np.pi

#===========================
# Atomic form factor (X-ray)
#===========================

# parameters for atomic scattering factor[1]
# [1]New Analytical Scattering-Factor Functions for Free Atoms and  Ions, D. Waasamaier et.al
aff_parameters = {}
#asf_parameters['atom'] = [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c]
aff_parameters['Sr'] = [17.730219, 1.563060, 9.795867, 14.310868, 6.099763, 0.120574, 2.620025, 135.771315, 0.600053, 0.120574, 1.140251]
aff_parameters['Te'] = [6.660302, 33.031656, 6.940756, 0.025750, 19.847015, 5.065547, 1.557175, 84.101613, 17.802427, 0.487660, -0.806668]
aff_parameters['W'] = [31.507901 , 1.629485, 15.682498 , 9.446448 , 37.960127, 0.000898 , 4.885509, 59.980675, 16.792113, 0.160798, -32.864576]
aff_parameters['Re'] = [31.888456, 1.549238, 16.117103 , 9.233474, 42.390296, 0.000689, 5.211669, 54.516371, 16.767591, 0.152815, -37.412681]
aff_parameters['Os'] = [32.210298, 1.473531, 16.678440, 9.049695 , 48.559907, 0.000519, 5.455839, 50.210201, 16.735532, 0.145771, -43.677984]
aff_parameters['Ir'] = [32.004437, 1.353767, 1.975454, 81.014172 , 17.070104, 0.128093, 15.939454, 7.461196, 5.990003,  26.659403, 4.018893]
aff_parameters['Pt'] = [31.273891, 1.316992, 18.445441, 8.797154, 17.063745, 0.124741, 5.555933, 40.177994, 1.575270,  1.316997, 4.050394]
aff_parameters['Ti'] = [16.630795, 0.110704, 19.386615, 7.181401, 32.808570, 1.119730, 1.747191, 98.660262, 6.356862,  26.014978, 4.066939]
aff_parameters['Pb'] = [16.419567, 0.105499, 32.738592, 1.055049, 6.530247, 25.025890, 2.342742, 80.906596, 19.916475,  6.664449, 4.049824]
aff_parameters['Bi'] = [16.282274, 0.101180, 32.725137, 1.002287, 6.678302, 25.714145, 2.694750, 77.057550, 20.576559,  6.291882, 4.040914]
aff_parameters['Po'] = [16.289164, 0.098121, 32.807170, 0.966265, 21.095164, 6.046622, 2.505901, 76.598071 , 7.254589,  28.096128, 4.046556]
aff_parameters['La'] = [19.966018, 3.197408, 27.329654, 0.003446, 11.018425, 19.955492, 3.086696, 141.381979, 17.335454,  0.341817, -21.745489]
aff_parameters['Ce'] = [17.355121, 0.328369, 63.988498, 0.002047, 20.546650, 3.088196, 3.130670, 134.907661, 11.353665,  18.832961, -38.386017]
aff_parameters['Pr'] = [21.551311, 2.995675, 17.161729, 0.312491, 11.903859, 17.716705, 2.679103, 152.192827, 9.564197,  0.010468, -3.871068]
aff_parameters['Nd'] = [17.331244, 0.300269, 62.783923, 0.001320, 12.160097, 17.026001, 2.663483, 148.748986, 22.239951,  2.910268, -57.189844]
aff_parameters['Pm'] = [17.286388, 0.286620, 51.560161, 0.001550, 12.478557, 16.223755, 2.675515, 163.984513, 22.960947,  2.796480, -45.973681]
aff_parameters['Sm'] = [23.700364, 2.689539, 23.072215, 0.003491, 12.777782, 15.495437, 2.6842171, 139.862475, 17.204366,  0.274536, -17.452166]




def form_factor(element, k):
    """
    calculate atomic form factor for x-ray
    input:
    
    :param str element:
    :param float k = 2sin(th)/lambda in Ang^{-1}.
    """
    k=k/2.0 # = sin(th)/lambda in Ang^{-1}.
    p = aff_parameters[element]
    sum_ = 0.0
    for i in range(5):
        sum_ += p[2*i]*math.exp(-1.0*p[2*i + 1]*k**2)
    return sum_ + p[10]

#================================
# Neutron scattering lengths and cross sections
#================================

# [1] https://www.ncnr.nist.gov/resources/n-lengths/
neutron_table = {}
#neutron_table['Isotope'] = [conc, Coh_b, Inc_b, Coh_xs, Inc_xs, Scatt_xs, Abs_xs]
neutron_table['Tb3+'] = [100, 7.38, -0.17, 6.84, 0.004, 6.84, 23.4]
neutron_table['Au']   = [100, 7.63, -1.84, 7.32, 0.43,  7.75, 98.65]
neutron_table['Ge']   = ['-', 8.185,  '-', 8.42, 0.18,  8.6,   2.2]

def scattering_lengths_and_cross_section(ion, wvl):
    """
    Neutron scattering lengths and cross sections
    # Isotope
    # conc, Natural abundance (For radioisotopes the half-life is given instead)
    # Coh_b, coherent scattering length, in fm
    # Inc_b, incoherent scattering length, in fm
    # Coh_xs, coherent scattering cross section, in barn
    # Inc_xs, incoherent scattering cross section, in barn
    # Scatt_xs, scattering cross section, in barn
    # Abs_xs, absorption cross section for 2200 m/s neutrons
    # Note: 1fm=1E-15 m, 1barn=1E-24 cm^2, scattering lengths and cross sections in parenthesis are uncertainties.
    input: 
    
    :param str ion:
    :param float  wvl: neutron wavelength in Ang.
    """
    return neutron_table[ion]

# [2] International Tables for Crystallography (2006). Vol. C, Chapter 4.4. page.445


#================================
# Magnetic form factors in the dipole approximation (Neutron)
# ("双極子近似による磁気形状因子" in Japanese)
#================================
# References
# [1] International Tables for Crystallography (2006). Vol. C, Chapter 4.4. page.454
# [2] Prof. S.J.Sato's Webpage, http://sato.issp.u-tokyo.ac.jp/ibuka/magform.html

# Table 4.4.5.3. <j_0> form factors for rare-earth ions
j0_parameters = {}
# j0_parameters['ion'] = [A a B b C c D e]
j0_parameters['Fe2+'] = [0.0263, 34.960, 0.3668, 15.943, 0.6188, 5.594, -0.0119, 0.1437]
j0_parameters['Tb2+'] = [0.0547, 25.509, 0.3171, 10.591, 0.6490, 3.517, -0.0212, 0.0342]
j0_parameters['Tb3+'] = [0.0177, 25.510, 0.2921, 10.577, 0.7133, 3.512, -0.0231, 0.0512]

# Table 4.4.5.7. <j_2> form factors for rare-earth ions
j2_parameters = {}
# j2_parameters['ion'] = [A a B b C c D e]
j2_parameters['Fe2+'] = [1.6490, 16.559, 1.9064, 6.133, 0.5206, 2.137, 0.0035, 0.0335]
j2_parameters['Tb2+'] = [0.6688, 18.491, 1.2487, 6.822, 0.8888, 2.275, 0.0215, 0.0439]
j2_parameters['Tb3+'] = [0.2892, 18.497, 1.1678, 6.797, 0.9437, 2.257, 0.0232, 0.0458]

# total spin angular momentum for 3d transition metals
# total spin angular momentum (全スピン角運動量), S
# Landé g factor, gfact
table1_parameters = {}
# electrons, ions, S, gfact
# 3d transition metals
table1_parameters['Ti3+'] = ['3d1', 1/2, 4/5]
table1_parameters['V4+']  = ['3d1', 1/2, 4/5]
table1_parameters['V3+']  = ['3d2',   1, 2/3]
table1_parameters['Cr3+'] = ['3d3', 3/2, 2/5]
table1_parameters['V2+']  = ['3d3', 3/2, 2/5]
table1_parameters['Mn3+'] = ['3d4',   2, '-']
table1_parameters['Cr2+'] = ['3d4',   2, '-']
table1_parameters['Fe3+'] = ['3d5', 5/2, 3/2]
table1_parameters['Mn2+'] = ['3d5', 5/2,   2]
table1_parameters['Fe2+'] = ['3d6',   2,   2]
table1_parameters['Co2+'] = ['3d7', 3/2, 4/3]
table1_parameters['Ni2+'] = ['3d8',   1, 5/4]
table1_parameters['Cu2+'] = ['3d9', 1/2, 6/5]
# for rare-earth ions
table1_parameters['Ce3+'] = ['4f1', 1/2, 6/7]
table1_parameters['Pr3+'] = ['4f2',   1, 4/5]
table1_parameters['Nd3+'] = ['4f3', 3/2,8/11]
table1_parameters['Pm3+'] = ['4f4',   2, 3/5]
table1_parameters['Sm3+'] = ['4f5', 5/2, 2/7]
table1_parameters['Eu3+'] = ['4f6',   3,   0]
table1_parameters['Gd3+'] = ['4f7', 7/2,   2]
table1_parameters['Tb3+'] = ['4f8',   3, 3/2]
table1_parameters['Dy3+'] = ['4f9', 5/2, 4/3]
table1_parameters['Ho3+'] = ['4f10',  2, 5/4]
table1_parameters['Er3+'] = ['4f11',3/2, 6/5]
table1_parameters['Tm3+'] = ['4f12',  1, 7/6]
table1_parameters['Yb3+'] = ['4f13',1/2, 8/7]

# for Lanthanides ions
# total spin angular momentum (全スピン角運動量), S
# total orbital angular momentum (全軌道角運動量), L
# total angular momentum (全角運動量), J
table2_parameters = {}
# ions, electrons, S, L, J
table2_parameters['Ce3+'] = ['4f1', 1/2, 3, 5/2]
table2_parameters['Pr3+'] = ['4f2',   1, 5,   4]
table2_parameters['Nd3+'] = ['4f3', 3/2, 6, 9/2]
table2_parameters['Pm3+'] = ['4f4',   2, 6,   4]
table2_parameters['Sm3+'] = ['4f5', 5/2, 5, 5/2]
table2_parameters['Sm2+'] = ['4f6',   3, 3,   0]
table2_parameters['Eu3+'] = ['4f6',   3, 3,   0]
table2_parameters['Eu2+'] = ['4f7', 7/2, 0, 7/2]
table2_parameters['Gd3+'] = ['4f7', 7/2, 0, 7/2]
table2_parameters['Tb3+'] = ['4f8',   3, 3,   6]
table2_parameters['Dy3+'] = ['4f9', 5/2, 5,15/2]
table2_parameters['Ho3+'] = ['4f10',  2, 6,   8]
table2_parameters['Er3+'] = ['4f11',3/2, 6,15/2]
table2_parameters['Tm3+'] = ['4f12',  1, 5,   6]
table2_parameters['Yb3+'] = ['4f13',1/2, 3, 7/2]

def j0(ion, s):
    """
    calculates j0 magnetic form factors, see ITC-vol-C, page 460, Eq (4.4.5.2)
    inputs:
    
    :param str ion:
    :param float s: value of sin(theta)/lambda in Ang^{-1}.
    """
    p = j0_parameters[ion]
    #print(p)
    sum_ = 0.
    for i in range(3):
        sum_ += p[2*i]*math.exp(-1.*p[2*i+1]*s**2)
    return sum_ + p[6]

def j2(ion, s):
    """
    calculates j2 magnetic form factors, see ITC-vol-C, page 460, Eq (4.4.5.3)
    inputs:
    
    :param str ion:
    :param float s: value of sin(theta)/lambda in Ang^{-1}.
    """
    p = j2_parameters[ion]
    sum_ = 0.
    for i in range(3):
        sum_ += p[2*i]*s**2*math.exp(-1.*p[2*i+1]*s**2)
    return sum_ + p[6]*s**2

def magnetic_form_factor_1(ion,s):
    """
    The momentum transfer dependence of <j0>, <j2>, <j0>^2 for 3d transition metals.
    Within the dipole approximation (spherical symmetry).
    gfact: Landé g factor (Lovesey, 1984)
    see ITC-vol-C, page 592, Eq (6.1.2.17)
    
    inputs:
    :param str ion:
    :param float s: value of sin(theta)/lambda in Ang^{-1}.
    """
    _, mS, gfact = table1_parameters[ion]
    return j0(ion,s)+j2(ion,s)*(1-2/gfact)

def magnetic_form_factor_2(ion,s):
    """
    calculates magnetic form factor for Lanthanides ions.
    input:
    
    :param str ion:
    :param float s: value of sin(theta)/lambda in Ang^{-1}.
    """
    _, mS, mL, mJ = table2_parameters[ion]
    b1 = mJ*(mJ+1) + mL*(mL+1) - mS*(mS+1)
    b2 = 3*mJ*(mJ+1) + mS*(mS+1) - mL*(mL+1)
    return j0(ion,s)+j2(ion,s)*b1/b2

def magnetic_form_factor(ion, s, flag):
    """
    calculates magnetic form factors within the dipole approximation.
    
    input:
    :param str ion:
    :param float s: value of sin(theta)/lambda in Ang^{-1}.
    :param str flag: 
            (0) when the orbital angular momentum is quenched and can be replaced by g(g-2)S_i with g-factor and spin angular momentum in the each ion i. 
                see ITC-vol-C, page 592, Eq (6.1.2.17)
            (1) when the magnetic moment of the ion i can be characterized by the total angular momentum J_i.
                add reherences!
    """
    if flag == 1:
        return magnetic_form_factor_2(ion,k)
    else:
        return magnetic_form_factor_1(ion,k)

if __name__ == '__main__':
    
    #----------
    # X-ray
    #----------
    element = 'Tb'
    
    th = 20 # in degree
    th = th/180*PI # in rad
    wvl = 1.5418 # in Ang.
    k = 2*math.sin(th)/wvl
    
    aff = form_factor(element, k)
    print(aff)
    
    aff = form_factor(element, 0.0)
    print(aff)
    
    #----------
    # Neutron
    #----------
    import os
    import matplotlib.pyplot as plt
    
    ion = 'Tb3+'
    #ion = 'Fe2+'
    
    # plot
    ss = np.arange(0.0, 1.5, 0.001)
    j0_ = []
    j2_ = []
    j0sq_ = []
    for i in range(len(ss)):
        j0_.append(j0(ion,ss[i]))
        j2_.append(j2(ion,ss[i]))
        j0sq_.append(j0(ion,ss[i])**2)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots()
    ax.plot(ss, j0_, lw = 1, c = 'r')
    ax.plot(ss, j2_, lw = 1, c = 'b')
    ax.plot(ss, j0sq_, lw = 1, c = 'm')
    #ax.tick_params(labelleft = False)
    ax.set_xlabel("sin(th) [Å^{-1}]")
    ax.set_ylabel("arb. unit")
    path = 'tmp'
    fname = ion
    if os.path.isdir('%s'%(path))==False:
      os.mkdir('%s'%(path))
    fig.savefig('%s/%s.png'%(path,fname))
    
    # Magnetic form factors within the dipole approximation.
    #   flag: (0) when the orbital angular momentum is quenched and can be replaced by g(g-2)S_i
    #             with g-factor and spin angular momentum in the each ion i. 
    #             see ITC-vol-C, page 592, Eq (6.1.2.17)
    #         (1) when the magnetic moment of the ion i can be characterized by the total angular momentum J_i.
    #flag = 0
    flag = 1
    mff = magnetic_form_factor(ion, k, flag)
    print(mff)
    