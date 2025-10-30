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
aff_parameters['Mg'] = [4.708971, 4.875207, 1.194814, 108.50607, 1.558157, 0.111516, 1.170413, 48.292407, 3.239403, 1.928171, 0.126842]
aff_parameters['Al'] = [4.730796, 3.620931, 2.313951, 43.051166, 1.541980, 0.095960, 1.117564, 108.93231, 3.154754, 1.555918, 0.139509]
aff_parameters['Si'] = [5.275329, 2.631338, 3.191038, 33.730728, 1.511514, 0.081119, 1.356849, 86.288640, 2.519114, 1.170087, 0.145073]
aff_parameters['Ca'] = [8.593655, 10.46064, 1.477324, 0.041891, 1.436254, 81.390382, 1.182839, 169.847839, 7.113258, 0.688098, 0.196255]
aff_parameters['Sc'] = [1.476566, 53.13102, 1.487278,  0.035325, 1.600187, 137.319495, 9.177463, 9.098031, 7.099750, 0.602102, 0.157765]
aff_parameters['Ti'] = [9.818524, 8.001879, 1.522646, 0.029763, 1.703101, 39.885423, 1.768774, 120.158000, 7.082555, 0.532405, 0.102473]
aff_parameters['V']  = [10.47357, 7.081940, 1.547881, 0.026040, 1.986381, 31.909672, 1.865616, 108.022044, 7.056250, 0.474882, 0.067744]
aff_parameters['Cr'] = [11.00706, 6.366281, 1.555477, 0.023987, 2.985293, 23.244838, 1.347855, 105.774500, 7.034779, 0.429369, 0.065510]
aff_parameters['Mn'] = [11.70954, 5.597120, 1.733414, 0.017800, 2.673141, 21.788419, 2.023368, 89.517915, 7.003180, 0.383054, -0.147293]
aff_parameters['Fe'] = [12.31109, 5.009415, 1.876623, 0.014461, 3.066177, 18.743041, 2.070451, 82.767874, 6.975185, 0.346506, -0.304931]
aff_parameters['Co'] = [12.91451, 4.507138, 2.481908, 0.009126, 3.466894, 16.438130, 2.106351, 76.987317, 6.960892, 0.314418, -0.936572]
aff_parameters['Ni'] = [13.52186, 4.077277, 6.947285, 0.286763, 3.866028, 14.622634, 2.135900, 71.966078, 4.284731, 0.004437, -2.762697]
aff_parameters['Cu'] = [14.01419, 3.738280, 4.784577, 0.003744, 5.056806, 13.034982, 1.457971, 72.554793, 6.932996, 0.265666, -3.254477]
aff_parameters['Zn'] = [14.74100, 3.388232, 6.907748, 0.243315, 4.642337, 11.903689, 2.191766, 63.312130, 38.424042, 0.000397, -36.915828]
aff_parameters['Ga'] = [15.75894, 3.121754, 6.841123, 0.226057, 4.121016, 12.482196, 2.714681, 66.203622, 2.395246, 0.007238, -0.87395]
aff_parameters['Ge'] = [16.54061, 2.866618, 1.567900, 0.012198, 3.727829, 13.432163, 3.345098, 58.866046, 6.785079, 0.210974, 0.018726]
aff_parameters['As'] = [17.02564, 2.597739, 4.503441, 0.003012, 3.715904, 14.272119, 3.937200, 50.437997, 6.790175, 0.193015, -2.984117]
aff_parameters['Se'] = [17.35407, 2.349787, 4.653248, 0.002550, 4.259489, 15.579460, 4.136455, 45.181201, 6.749163, 0.177432, -3.160982]
aff_parameters['Y']  = [17.79204, 1.429691, 10.25325, 13.132816, 5.714949, 0.112173, 3.170516, 108.197029, 0.918251, 0.112173, 1.131787]
aff_parameters['Zr'] = [17.05977, 1.310692, 10.91103, 12.319285, 5.821115, 0.104353, 3.512513, 91.777544, 0.746965, 0.104353, 1.124859]
aff_parameters['Nb'] = [17.95839, 1.211590, 12.06305, 12.246687, 5.007015, 0.098615, 3.287667, 75.011944, 1.531019, 0.098615, 1.123452]
aff_parameters['Mo'] = [6.236218, 0.090780, 17.98771, 1.108310, 12.973127, 11.468720, 3.451426, 66.684153, 0.210899, 0.090780, 1.108770]
aff_parameters['Tc'] = [17.84096, 1.005729, 3.428236, 41.901383, 1.373012, 119,320541, 12.947364, 9.781542, 6.335469, 0.083391, 1.074784]
aff_parameters['Ru'] = [6.271624, 0.077040, 17.90673, 0.928222, 14.123269, 9.555345, 3.746008, 35.860678, 0.908235, 123.552247, 1.043992]
aff_parameters['Rh'] = [6.216648, 0.070789, 17.91973, 0.856121, 3.854252, 33.889484, 0.840326, 121.686688, 15.173498, 9.029517, 0.995452]
aff_parameters['Pd'] = [6.121511, 0.062549, 4.784063, 0.784031, 16.631683, 8.751391, 4.318258, 34.489983, 13.246773, 0.784031, 0.883099]
aff_parameters['Ag'] = [6.073874, 0.055333, 17.15543, 7.896512, 4.173344, 28.443739, 0.852238, 110.376108, 17.988685, 0.716809, 0.756603]
aff_parameters['Cd'] = [6.080986, 0.048990, 18.01946, 7.273646, 4.018197, 29.119283, 1.303510, 95.831208, 17.974669, 0.661231, 0.605304]
aff_parameters['In'] = [6.196477, 0.042072, 18.81618, 6.695665, 4.050479, 31.009791, 1.638929, 103.284350, 17.962912, 0.610714, 0.333097]
aff_parameters['Sn'] = [19.32517, 6.118104, 6.281571, 0.036915, 4.498866, 32.529047, 1.856934, 95.037182, 17.917310, 0.565651, 0.119024]
aff_parameters['Sb'] = [5.394956, 33.32652, 6.549570, 0.030974, 19.650681, 5.564929, 1.827820, 87.130965, 17.867833, 0.523992, -0.290506]
aff_parameters['Eu'] = [17.18619, 0.261678, 37.15683, 0.001995, 13.103387, 14.787360, 2.707246, 134.816293, 24.419271, 2.581883, -31.586687]
aff_parameters['Gd'] = [24.89811, 2.435028, 17.10495, 0.246961, 13.222581, 13.996325, 3.266152, 110.863093, 48.995214, 0.001383, -43.505684]
aff_parameters['Tb'] = [25.91001, 2.373912, 32.34413, 0.002034, 13.765117, 13.481969, 2.751404, 125.836511, 17.064405, 0.236916, -26.851970]
aff_parameters['Dy'] = [26.67178, 2.282593, 88.68757, 0.000665, 14.065445, 12.920230, 2.768497, 121.937188, 17.067782, 0.225531, -83.279831]
aff_parameters['Ho'] = [27.15019, 2.169660, 16.99981, 0.215414, 14.059334, 12.213148, 3.386979, 100.506781, 46.546471, 0.001211, -41.165283]
aff_parameters['Er'] = [28.17488, 2.120995, 82.49326, 0.000640, 14.624002, 11.915256, 2.802756, 114.529936, 17.018515, 0.207519, -77.135221]
aff_parameters['Tm'] = [28.92589, 2.046203, 76.17379, 0.000656, 14.904704, 11.465375, 2.814812, 111.411979, 16.998117, 0.199376, -70.839813]
aff_parameters['Yb'] = [29.67676, 1.977630, 65.62406, 0.000720, 15.160854 ,11.044622 ,2.830288 ,108.139150 ,16.997850 ,0.192110 ,-60.313812]
aff_parameters['Lu'] = [30.12286, 1.883090, 15.09934, 10.342764, 56.314899, 0.000780, 3.540980, 89.559248, 16.943730, 0.183849, -51.049417]
aff_parameters['Hf'] = [30.61703, 1.795613, 15.14535, 9.934469, 54.933548, 0.000739, 4.096253, 76.189707, 16.896157, 0.175914, -49.719838]
aff_parameters['Ta'] = [31.06635, 1.708732, 15.34182, 9.618455, 49.278296, 0.000760, 4.577665, 66.346202, 16.828321, 0.168002, -44.119025]
aff_parameters['Au'] = [16.77738, 0.122737, 19.31715, 8.621570, 32.979682, 1.256902, 5.595453, 38.008821, 10.576854, 0.000601, -6.279078]

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
    