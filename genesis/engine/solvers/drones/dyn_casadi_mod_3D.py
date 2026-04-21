import os
import sys
sys.path.append(os.environ["FLIGHTMARE_PATH"] + "/flightpy/opt_control")

import casadi as ca
import spatial_casadi as sc
import numpy as np
# import yaml
import matplotlib.pyplot as plt
from lib.integrators import rk4, euler_newton
from lib.utils import quat_dot

def simulate(x0, u, N, tf, dyn_fun):
  x = x0
  times = np.linspace(start=0, stop= tf, num=N, endpoint=False)
  # print("Simulation timestep : " + str(np.mean(np.diff(times))))
  x_history = np.zeros((len(times),len(x0)))
  x_dot_history = np.zeros((len(times),len(x0)))
  u_history = u

  for k in range(len(times)-1):
    x_history[k] = x
    x_dot = np.array(dyn_fun(x,u[k])).flatten()
    x_dot_history[k] = x_dot
    # print("State derivative at time " + str(times[k]) + ": " + str(x_dot))
    x = rk4(dyn_fun,x,u[k],times[k+1]-times[k])
    # print("State at time " + str(times[k]) + ": " + str(x))

  return times, x_history, u_history, x_dot_history

class IndoorUAV3D:
  def __init__(self, filename = ""):
    self.m = 0.132 # Drone mass in [kg]
    self.g = 9.801
    self.pi = 3.14159
    self.rho = 1.225
    self.nu = 1.5e-5

    # Wing geometric paramters
    self.b_center = 0.027                                        
    self.b_root_inner = (0.11*2) + self.b_center
    self.c_root_inner = 0.18
    self.b_root_outer = (0.155*2) + self.b_center
    self.c_root_outer = 0.15
    self.s_one_root = 0.0265

    # Wing mass parameters
    self.m_outer_wing = 0.0055

    self.s_wing_root = (2*self.s_one_root)+(self.b_center*self.c_root_inner)
    self.mac_root = (1/self.s_wing_root)*(self.b_root_inner*self.c_root_inner*self.c_root_inner + (self.b_root_outer - self.b_root_inner)*self.c_root_outer*self.c_root_outer)
    self.geo_ctr_y_root = (1/self.s_wing_root)*(self.b_root_inner*self.c_root_inner*self.b_root_inner/4 + (self.b_root_outer - self.b_root_inner)*self.c_root_outer*(self.b_root_outer + self.b_root_inner)/4)
    self.geo_ctr_x_root = (1/self.s_wing_root)*(self.b_root_inner*self.c_root_inner*self.c_root_inner/2 + (self.b_root_outer - self.b_root_inner)*self.c_root_outer*(self.c_root_outer/2))

    #Tail geometric parameters
    self.k_alpha_elev = 0.7 #Relation betwween rudder angle and effective AOA (d(alpha_eff)/d(tau_ele))
    self.k_alpha_rudder = 0.7 #Relation betwween rudder angle and effective AOA (d(alpha_eff)/d(tau_rud))
    self.s_hor_tail = 0.021495
    self.s_vert_tail = 0.0125625
    self.c_hor_tail = 0.10
    self.c_vert_tail = 0.15
    self.c_ele = 0.058
    self.c_rud = 0.096
    self.b_hor_tail = 0.20
    self.b_vert_tail = 0.15

    #Slipstream effectiveness parameters
    self.K_slip_tail = 1.0
    self.K_slip_wing = 1.0
    
    #Reynolds degradation parameters
    self.Re_ref = 100000
    self.M_Re = 2.5

    # Aerodynamic parameters
    self.M_smooth = 0.2 # Stall transition smoothing

    # Wing lift/drag parameters
    self.alpha_stall_wing = 14.0*(self.pi/180) #Defined in Radians
    self.cl_alpha_wing_2D = (2*self.pi) # + 5.37)/2 #5.73 #5.73 #2D cl_alpha_coefficient
    self.c_l_0_wing = 0.0 #C_L_0 of airfoil
    self.c_d_0_wing = 0.05 #Zero AOA drag coefficient of drone
    self.c_d_0_tail = 0.2
    self.c_m_0_wing = 0.0 #Zero AOA drag coefficient of drone (M_wing_0 = 0.5*rho*c_wing*c_m_0*S_wing*V**2), TBI
    self.c_m_y_drone_static = -0.0017 #-0.0004819 #Identified from optimization #Static moment offset coefficient (M_offset = 0.5*rho*c_m_drone*V**2), TBI
    self.c_m_x_drone_static = 0.0 
    self.c_m_z_drone_static = 0.0

    # Tail lift/drag parameters
    self.alpha_stall_tail = 20.0*(self.pi/180)

    #Position offsets from Wing leading edge
    self.pos_cg_body = ca.DM([-.085, 0., 0.]) # Distance of C.G. from wing leading edge
    self.fixed_cg = False # If True, C.G. is fixed at pos_cg_theta_sw_0, if False, C.G. is updated based on wing geometry
    self.pos_outer_wing_left_hinge = ca.DM([-0.0145, 0.1735, 0.0]) # Position of left outer wing hinge from body frame origin
    self.pos_outer_wing_right_hinge = ca.DM([-0.0145, -0.1735, 0.0]) # Position of right outer wing hinge from body frame origin
    self.pos_ele_le = ca.DM([-.339, .0, .0]) #np.array([-.1, .0, .0]) #
    self.pos_rud_le = ca.DM([-.344, 0., .075])
    self.pos_wing_left_le = ca.DM([.0, 0.0145, .0])
    self.pos_wing_right_le = ca.DM([.0, -0.0145, .0])
    self.pos_prop = ca.DM([.18, 0., 0.0])

    # self.I_ext = ca.DM([(917668.0, 0., 39330.0), (0., 3543007.0, 0.), (39330.0, 0., 4389834.0)])/(10**9)    # Inertia
    # self.I_ext_inv = ca.inv(self.I_ext)                       # Inertia inverse

    self.I_body_cg = ca.DM([(398235.18, 0.0, -148894.47), (0.0, 3669185.14, 0.0), (-148894.47, 0.0, 3989882.30)])/(10**9)  # Inertia about C.G. in body frame
    self.I_outer_wing_l_cg_sw = ca.DM([(10555.81, 2787.67, 0.0), (2787.67, 8703.71, 0.0), (0.0, 0.0, 19245.67)])/(10**9)  # Inertia of outer left wing about its C.G. in swept wing frame    
    self.I_outer_wing_r_cg_sw = ca.DM([(10555.81, -2787.67, 0.0), (-2787.67, 8703.71, 0.0), (0.0, 0.0, 19245.67)])/(10**9)  # Inertia of outer right wing about its C.G. in swept wing frame    
    self.r_outer_wing_left_cg_hinge_sw = ca.DM([-0.04482, 0.04632, 0.0])
    self.r_outer_wing_right_cg_hinge_sw = ca.DM([-0.04482, -0.04632, 0.0])

    self.R_prop = 0.075
    self.sw_min = -5.0                                # min sweep angle [deg]
    self.sw_max = 75.0                                # max sweep angle [deg]
    self.ele_min = -26.0                           # min elevator angle [deg]
    self.ele_max = 26.0                               # max elevator angle [deg]
    self.rud_min = -30.0                               # min rudder angle (positive means clockwise deflection) [deg]
    self.rud_max = 30.0                                # max rudder angle (negative means counter clock-wise deflection) [deg]
    
    # Servo Model parameters
    self.sweep_offset = 0.5 #Servo offset for normalized range (Typically -1.0 to 0.5)
    self.A_sw = ca.DM([(0.0, 1.0), (-974.5, -44.1)])
    self.B_sw = ca.DM([(-4.71),(1174.0)])
    self.C_sw = ca.DM([(1.0, 0.0)])
    self.ele_tau = 0.02449                          # elevator servo time constant [s]
    self.rud_tau = 0.02449                          # rudder servo time constant [s]

    # Motor Model parameters
    self.throttle_offset = 0.05
    self.motor_omega_map = -0.6713 #-0.529 #-0.504
    self.motor_tau_inv = 2.054 #2.20 #1.709
    self.T_min = 0                                        # min thrust [N]
    self.T_max = 0.9 #1.0 #1.01 #1.01                          # max thrust [N]
    self.delay_mot = 0.0 #0.175                           # motor input delay [s]
                                                     
  def thrust_slipstream(self, thrust, vel_u):
    prop_wake = (-vel_u + ca.sqrt(vel_u**2 + 2*thrust/(self.rho*self.pi*self.R_prop**2)))/2
    V_slip_wing = self.K_slip_wing*prop_wake
    V_slip_tail = self.K_slip_tail*prop_wake

    return V_slip_wing, V_slip_tail
  
  def wing_geometry(self, theta_sw_deg):
    #Ensure theta_sw_deg is defined in degrees °
    b_outer = (-0.0201*theta_sw_deg**2+0.0904*theta_sw_deg+165.15)/1000 #Fit from Excel, in m
    s_outer = (-142.97*theta_sw_deg + 16290)/(10**6) #Fit from Excel, in m2
    ac_x_outer = self.c_root_outer/6 #Assuming neutral point of a double traingular wing with base chord of 0.15 m //(0.15/4)+(2*b_outer*std::tan(theta_sw_deg*M_PI/180)/6); // Assuming outer triangle M.A.C based on base chord of 0.15 m: (2*0.15*s_outer/12) with pointed tip wing assumption
    cpx_theta_0 = 59.55/1000
    cpy_theta_0 = 62.78/1000
    cpx_outer = ((-0.003*theta_sw_deg**2)+0.7136*theta_sw_deg)/1000 + cpx_theta_0 #Fit of x centroid from Excel, in m
    cpy_outer = (((-0.00397*theta_sw_deg**2)-0.251*theta_sw_deg)/1000 + cpy_theta_0) + self.b_root_outer/2 #Fit of y centroid from Excel, in m, with offset of half root chord

    b = self.b_root_outer + 2*b_outer
    S = self.s_wing_root + 2*s_outer
    AR = (b**2)/S
    ac_x = (((self.mac_root/4)*self.s_wing_root)+(ac_x_outer*2*s_outer))/S
    cpx_wing = ((self.geo_ctr_x_root*self.s_wing_root)+(cpx_outer*2*s_outer))/S
    cpy_wing = b/4 #((self.geo_ctr_y_root*self.s_wing_root)+(cpy_outer*2*s_outer))/S
    cpz_wing = 0.0

    S_one_wing = S/2
    b_one_wing = b/2

    cp_wing_vec = ca.vertcat(cpx_wing, cpy_wing, cpz_wing) #Centroid vector of wing in body frame relative to leading edge of wing offset from C.G.

    return AR, S_one_wing, b_one_wing, ac_x, cp_wing_vec
  
  def total_cg_and_inertia(self, theta_sw_l_deg, theta_sw_r_deg):
  
    # Rotation matrices from swept wing frame to body frame
    R_swept_to_body_l = sc.Rotation.from_euler('z', theta_sw_l_deg*(self.pi/180)).as_matrix()
    R_swept_to_body_r = sc.Rotation.from_euler('z', -theta_sw_r_deg*(self.pi/180)).as_matrix()

     # Calculate corrected C.G. location based on wing sweep
    r_wing_l_body = self.pos_outer_wing_left_hinge + R_swept_to_body_l @ self.r_outer_wing_left_cg_hinge_sw
    r_wing_r_body = self.pos_outer_wing_right_hinge + R_swept_to_body_r @ self.r_outer_wing_right_cg_hinge_sw
    if self.fixed_cg:
      pos_cg = self.pos_cg_body
    else:
      pos_cg = (self.pos_cg_body * (self.m - 2*self.m_outer_wing) + (r_wing_l_body * self.m_outer_wing) + (r_wing_r_body * self.m_outer_wing)) / self.m

    # Calculate transformed inertia about hinge point of left and right wings based on sweep angles into body frame
    I_wing_l_body = R_swept_to_body_l @ self.I_outer_wing_l_cg_sw @ R_swept_to_body_l.T #Inertia of left outer wing about hinge in body frame
    I_wing_r_body = R_swept_to_body_r @ self.I_outer_wing_r_cg_sw @ R_swept_to_body_r.T #Inertia of right outer wing about hinge in body frame

    # Convert wing inertias from hinge point to center of gravity using parallel axis theorem
    r_wing_l_to_cg = -r_wing_l_body + pos_cg
    r_wing_r_to_cg = -r_wing_r_body + pos_cg

    I_wing_l_body_cg = I_wing_l_body + self.m_outer_wing*(ca.norm_2(r_wing_l_to_cg)**2*ca.DM.eye(3) - r_wing_l_to_cg @ r_wing_l_to_cg.T)
    I_wing_r_body_cg = I_wing_r_body + self.m_outer_wing*(ca.norm_2(r_wing_r_to_cg)**2*ca.DM.eye(3) - r_wing_r_to_cg @ r_wing_r_to_cg.T)

    I_body_cg = self.I_body_cg + (self.m - 2*self.m_outer_wing)*(ca.norm_2(pos_cg - self.pos_cg_body)**2*ca.DM.eye(3) - (pos_cg - self.pos_cg_body) @ (pos_cg - self.pos_cg_body).T)
    
    I_total_body_cg = I_body_cg +  I_wing_l_body_cg + I_wing_r_body_cg

    return pos_cg, I_total_body_cg
  
  def sigmoid(self,x,x_cut,M):

    sig = (1 + ca.exp(-M*(180/self.pi)*(x-x_cut)) + ca.exp(M*(180/self.pi)*(x+x_cut)))/((1 + ca.exp(-M*(180/self.pi)*(x-x_cut)))*(1 + ca.exp(M*(180/self.pi)*(x+x_cut))))

    return sig
  
  def wing_coefficients(self, alpha, alpha_stall, AR, ac_x, geo_x, pos_cg_from_le, f_re = 1, c_d_0 = 0.0, c_l_dyn_pre_st = 0.0):

    # alpha between -90° and 90°
 
    k_cd = 1 - 0.41*(1-ca.exp(-17/AR))
    w_pos = ca.cos(self.pi*((alpha-alpha_stall)/(self.pi-2*alpha_stall))-self.pi/2)
    w_neg = ca.cos(self.pi*((-alpha-alpha_stall)/(self.pi-2*alpha_stall))-self.pi/2)
    w = (1/(1+ca.exp(-20*(alpha-alpha_stall))))*w_pos + (1/(1+ca.exp(-20*(-alpha-alpha_stall))))*w_neg  #self.sigmoid(alpha,-alpha_stall,10)*w_pos + self.sigmoid(alpha,-alpha_stall,10)*w_neg
    c_l_st = f_re * (2*ca.sin(alpha)*ca.cos(alpha)*(1-w*(1-k_cd)))
    c_d_st = 2*f_re*(ca.sin(alpha)**2)*((1-w*(1-k_cd)))
    #d_x_ac_cg_st = pos_cg_from_le + (w*(geo_x - ac_x) + ac_x)

    c_l_lin =  f_re * (self.c_l_0_wing + ((self.cl_alpha_wing_2D*AR)/(2+ca.sqrt(AR**2 + 4)))*(alpha))
    c_d_quad = c_d_0 + (c_l_lin**2)/(self.pi*AR)
    #d_x_ac_cg_lin = pos_cg_from_le + ac_x

    sig = self.sigmoid(alpha,alpha_stall,self.M_smooth) # Efficient declaration
    c_l = (1-sig)*(c_l_lin + c_l_dyn_pre_st) + sig*c_l_st
    c_d = (1-sig)*c_d_quad + sig*c_d_st
    #d_x_m = (1-self.sigmoid(alpha,alpha_stall,self.M_smooth))*d_x_ac_cg_lin + self.sigmoid(alpha,alpha_stall,self.M_smooth)*d_x_ac_cg_st #(Unit: m)
    d_x_m = - (ac_x + (2*ca.sqrt(alpha**2)/self.pi)*(geo_x - ac_x)) - pos_cg_from_le #Position of aerodynamic center in body frame relative to C.G. (in body frame)
    #mac = ac_x*4
    # print(mac)
    #d_x_m = pos_cg_from_le + (0.5 - 0.175*(1-2*ca.sqrt(alpha**2)/self.pi))*ac_x*4
    #print(d_x_m)

    return c_l, c_d, d_x_m

  def wing_aerodynamics(self, V_slip_wing, vel, ome, AR, S, ac_x, geo_center_vec, pos_cg):

    S_slip = self.c_root_inner*(self.R_prop)
    S_free_wing = S - S_slip

    vel_inb = -(vel + ca.cross(ome, pos_cg + geo_center_vec)) #Body frame wing inbound velocity corrected for quasi-steady rotation 
    vel_slip_inb = vel_inb - ca.vertcat(V_slip_wing, 0, 0) #Slipstream velocity in body frame 

    V = ca.norm_2(vel_inb)  # velocity norm
    V_slip = ca.norm_2(vel_slip_inb)  # velocity norm

    #Reynolds degradation for wing segments
    f_re_wing = self.reynolds_wing_degradation(V, ac_x*4)
    f_re_slip = self.reynolds_wing_degradation(V_slip, self.mac_root)

    alpha_wing_inb = ca.atan2(vel_inb[2], -vel_inb[0]) # angle of attack, valid when vel_u > 0
    alpha_slip_inb = ca.atan2(vel_slip_inb[2], -vel_slip_inb[0]) # angle of attack, valid when vel_u > 0
    beta_wing = ca.asin(vel_inb[1]/V) #angle of side slip, valid when V > 0
    beta_wing_slip = ca.asin(vel_slip_inb[1]/V_slip) #angle of side slip, valid when V > 0

    # Dynamic coefficients
    c_l_dyn_qs_wing = (self.pi/2)*(-ome[1]*ac_x*4)/V
    c_l_dyn_qs_slip = (self.pi/2)*(-ome[1]*ac_x*4)/V_slip

    # print(alpha_wing_inb)
    # print(alpha_slip_inb)

    wing_cl, wing_cd, wing_dxm = self.wing_coefficients(alpha_wing_inb, self.alpha_stall_wing, AR, ac_x, geo_center_vec[0], pos_cg[0], f_re = f_re_wing, c_d_0 = self.c_d_0_wing, c_l_dyn_pre_st = c_l_dyn_qs_wing)
    slip_cl, slip_cd, slip_dxm = self.wing_coefficients(alpha_slip_inb, self.alpha_stall_wing, AR, ac_x, geo_center_vec[0], pos_cg[0], f_re = f_re_slip, c_d_0 = self.c_d_0_wing, c_l_dyn_pre_st = c_l_dyn_qs_slip)
    F_x_slip = (0.5*self.rho*S_slip*V_slip**2)*(ca.cos(beta_wing_slip)**2)*(slip_cl*ca.sin(alpha_slip_inb) - slip_cd*ca.cos(alpha_slip_inb))
    F_x_wing = (0.5*self.rho*S_free_wing*V**2)*(ca.cos(beta_wing)**2)*(wing_cl*ca.sin(alpha_wing_inb) - wing_cd*ca.cos(alpha_wing_inb)) #In body frame (forward x)
    F_z_slip = (0.5*self.rho*S_slip*V_slip**2)*(ca.cos(beta_wing_slip)**2)*(slip_cl*ca.cos(alpha_slip_inb) + slip_cd*ca.sin(alpha_slip_inb))
    F_z_wing = (0.5*self.rho*S_free_wing*V**2)*(ca.cos(beta_wing)**2)*(wing_cl*ca.cos(alpha_wing_inb) + wing_cd*ca.sin(alpha_wing_inb)) #In body frame (upward z)
    F_y_wing = 0.0
    F_y_slip = 0.0

    F = ca.vertcat(F_x_wing, F_y_wing, F_z_wing) + ca.vertcat(F_x_slip, F_y_slip, F_z_slip) #Force vector in body frame
    r_ac_avg = ca.vertcat((wing_dxm*S_free_wing+slip_dxm*S_slip)/S, geo_center_vec[1], geo_center_vec[2]) #Position vector of aerodynamic center in body frame relative to Wing L.E. (in body frame)

    #r_ac_slip = ca.vertcat(slip_dxm, geo_center_vec[1], geo_center_vec[2]) #Position vector of aerodynamic center in body frame relative to Wing L.E. (in body frame)
    M = ca.cross(r_ac_avg, F) + ca.vertcat(0.0,(0.5*self.rho*(4*ac_x)*self.c_m_0_wing*S*V**2),0.0)
    eps_down_tail = ca.vertcat(0.0, 0.0, 0.0) #ca.if_else(alpha_wing_inb < self.alpha_stall_wing, ((2*wing_cl)/(self.pi*AR)), 0) #positive when Cl positive, when V > 0, original: 2*c_l / pi*AR, with cl here assumed as wing_cl and V as free stream velocity

    return F, M, eps_down_tail
  
  def reynolds_wing_degradation(self, V_inb, mac_wing):

    Re = V_inb*mac_wing/(self.nu)

    f = ca.if_else(Re > self.Re_ref, 1.0, 1 - ((self.Re_ref - Re)/(self.Re_ref))**self.M_Re) #(Re/self.Re_ref)**(self.M_Re)
    # print(f)
    
    return f
  
  def hor_tail_aerodynamics(self, V_slip_tail, vel, ome, vel_eps_down_tail, pos_cg, ele):

    pos_ele_cg_le = (self.pos_ele_le - pos_cg)
    pos_ele_cg_cp = (pos_ele_cg_le + ca.vertcat(-self.c_ele/2, 0.0, 0.0)) #Position of elevator center of pressure in body frame relative to C.G.
    # print(d_x_ac_cg)

    vel_inb_slip = ca.vertcat(-V_slip_tail, 0, 0) #Slipstream velocity in body frame
    vel_inb = -(vel + ca.cross(ome, pos_ele_cg_cp)) + vel_eps_down_tail + vel_inb_slip #Body frame tail inbound velocity corrected for quasi-steady rotation and downwash

    V = ca.norm_2(vel_inb)
    alpha_hor_tail_inb = ca.atan2(vel_inb[2], -vel_inb[0]) #- eps_down_tail # angle of attack, valid when vel_u > 0
    #beta_hor_tail_inb = ca.asin(vel_inb[1]/V) #angle of side slip, valid when V > 0
    #print(eps_down_tail)
    
    #print(ele)
    alpha_ele_eff = self.k_alpha_elev*ele + alpha_hor_tail_inb
    #c_l, c_d, d_x_t = self.wing_coefficients(alpha_eff, self.alpha_stall_tail, AR_tail, self.c_hor_tail/4, self.c_hor_tail/2, pos_ele_le_cg, f_re = 1.0, c_d_0 = self.c_d_0_tail, c_l_dyn_pre_st=0.0) 
    c_l = 2*ca.sin(alpha_ele_eff)*ca.cos(alpha_ele_eff)
    c_d = self.c_d_0_tail + 2*(ca.sin(alpha_ele_eff)**2)
    d_x_t = self.c_hor_tail/4 + (2*ca.sqrt(alpha_ele_eff**2)/self.pi)*(self.c_hor_tail/4)
    pos_ele_cg_ac = pos_ele_cg_le + ca.vertcat(-d_x_t, 0.0, 0.0)

    F_l = 0.5*self.rho*self.s_hor_tail*c_l*V**2
    F_d = 0.5*self.rho*self.s_hor_tail*c_d*V**2

    F_x = F_l*ca.sin(alpha_ele_eff) - F_d*ca.cos(alpha_ele_eff)
    F_z = F_l*ca.cos(alpha_ele_eff) + F_d*ca.sin(alpha_ele_eff)
    
    F = ca.vertcat(F_x, 0.0, F_z) #Force vector in body frame
    M = ca.cross(pos_ele_cg_ac, F)  #Moment vector in body frame

    return F, M
  
  def vertical_tail_aerodynamics(self, V_slip_tail, vel, ome, pos_cg, rud):

    pos_rud_cg_le = (self.pos_rud_le - pos_cg)
    pos_rud_cg_cp = (pos_rud_cg_le + ca.vertcat(-self.c_rud/2, 0.0, 0.0))

    vel_inb = -(vel + ca.cross(ome, pos_rud_cg_cp)) - ca.vertcat(V_slip_tail, 0, 0) #Body frame tail inbound velocity corrected for quasi-steady rotation (neglect downwash)

    V = ca.norm_2(vel_inb)
    alpha_vert_tail_inb = ca.asin(vel_inb[1]/V) # angle of attack, valid when vel_u > 0

    alpha_rud_eff = - (self.k_alpha_rudder*rud) + alpha_vert_tail_inb
    c_l = 2*ca.sin(alpha_rud_eff)*ca.cos(alpha_rud_eff)
    c_d = 2*(ca.sin(alpha_rud_eff)**2)
    d_x_t = self.c_vert_tail/4 + (2*ca.sqrt(alpha_rud_eff**2)/self.pi)*(self.c_vert_tail/4)
    pos_rud_cg_ac = pos_rud_cg_le + ca.vertcat(-d_x_t, 0.0, 0.0)

    F_l = 0.5*self.rho*self.s_vert_tail*c_l*V**2
    F_d = 0.5*self.rho*self.s_vert_tail*c_d*V**2

    F_y = F_l*ca.cos(alpha_rud_eff) + F_d*ca.sin(alpha_rud_eff)
    F_x = F_l*ca.sin(alpha_rud_eff) - F_d*ca.cos(alpha_rud_eff)

    F = ca.vertcat(F_x, F_y, 0.0) #Force vector in body frame
    M = ca.cross(pos_rud_cg_ac, F)  #Moment vector in body frame

    return F, M
  
  def sweep_angle_from_normalized(self, theta_sw_norm):
    theta_sw_norm_scaled = (theta_sw_norm + (self.sweep_offset/2))/(1-self.sweep_offset/2)
    theta_sweep_deg = ((self.sw_max - self.sw_min)*(theta_sw_norm_scaled)/2 + (self.sw_max + self.sw_min)/2) #In degrees

    return theta_sweep_deg
  
  def sweep_servo_dynamics(self, x, u):
    
    #print(x)
    #print(u)
    # x = [theta_sw_norm, theta_sw_norm_dot]
    # x0 = theta_sw_norm_between -1.0 and 1.0
    x_sw_dot = self.A_sw @ x + self.B_sw @ u
    theta_sw_norm = self.C_sw @ x
    
    theta_sweep_deg = self.sweep_angle_from_normalized(theta_sw_norm)

    # print(u)
    # print(x)
    # print(theta_sw_norm)
    # print(theta_sweep_deg)

    return theta_sweep_deg, x_sw_dot
  
  def tail_first_order_servo_dynamics(self, x, u, min, max, tau):
    # x_ele_dot = u
    x_dot = (1/tau)*(u - x)
    x_norm = x

    x_ang_rad = ((max - min)*(x_norm)/2 + (max + min)/2)*(self.pi/180) #In radians

    return x_ang_rad, x_dot
      
  def motor_throttle_to_omega(self, u_thr):

    c_0 = self.motor_omega_map
    c_1 = ((self.motor_omega_map*(self.throttle_offset**2 - 1) + 1)/(1-self.throttle_offset))
    c_2 = 1 - c_0 - c_1
    omega_norm = c_0 * ((u_thr)**2) + c_1 * u_thr + c_2

    # print(u_thr)
    # print(omega_norm)
    #omega_norm = self.motor_omega_map * (u_thr - self.throttle_offset)**2 + (1/(1-self.throttle_offset) - self.motor_omega_map*(1+self.throttle_offset))*(u_thr- self.throttle_offset)

    return omega_norm
  
  def motor_dynamics(self, omega_mot_norm, u_thr):

    # u_thr = np.clip(u_thr, self.throttle_offset, 1.0)

    #u_thr is between u_thr_offset and 1 (must be clipped)
    omega_mot_des_norm = self.motor_throttle_to_omega(u_thr)
    omega_mot_norm_dot = self.motor_tau_inv * (omega_mot_des_norm - omega_mot_norm)

    thrust = (omega_mot_norm**2)*(self.T_max - self.T_min) + self.T_min
    F = ca.vertcat(thrust, 0.0, 0.0) #Force vector in body frame (thrust in forward x)

    #Implement a delay in MPC defining function

    return F, omega_mot_norm_dot
  
  def force_moment_cg_total(self, x, x_act_pos, u=ca.vertcat(0.5, 0.0, 0.0, 0.0, 0.0)):
    # State vector x ordered as: [pos_x, pos_y, pos_z, vel_u, vel_v, vel_w, q_x, q_y, q_z, q_w, ome_p, ome_q, ome_r, omega_mot_norm, x_sw_sym_l_0, x_sw_sym_l_1, x_sw_sym_r_0, x_sw_sym_r_1, x_ele, x_rud]
    # Control vector u ordered as: [u_thr, u_sw_l, u_sw_r, u_ele_dot, u_rud_dot]
    pos_x, pos_y, pos_z, vel_u, vel_v, vel_w, q_x, q_y, q_z, q_w, ome_p, ome_q, ome_r = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]
    omega_mot_norm, x_sw_ang_l, x_sw_ang_r, x_ele, x_rud = ca.fmax(ca.fmin(x_act_pos[0], 1.0), 0.0), ca.fmax(ca.fmin(x_act_pos[1], 1.0 - self.sweep_offset), -1.0), ca.fmax(ca.fmin(x_act_pos[2], 1.0 - self.sweep_offset), -1.0), ca.fmax(ca.fmin(x_act_pos[3], 1.0), -1.0), ca.fmax(ca.fmin(x_act_pos[4], 1.0), -1.0) #Clamp states to min/max values: 0 to 1 for motor omega norm, -1 to 0.5 for sweeps, -1 to 1 for elevator and rudder
    u_thr, u_sw_l, u_sw_r, u_ele, u_rud = ca.fmax(ca.fmin(u[0], 1.0), self.throttle_offset), ca.fmax(ca.fmin(u[1], 1.0 - self.sweep_offset), -1.0), ca.fmax(ca.fmin(u[2], 1.0 - self.sweep_offset), -1.0), ca.fmax(ca.fmin(u[3], 1.0), -1.0), ca.fmax(ca.fmin(u[4], 1.0), -1.0) #Clamp inputs to min/max values: 0 to 1 for throttle, -1 to 0.5 for sweeps, -1 to 1 for elevator and rudder

    quat = ca.vertcat(q_x, q_y, q_z, q_w) #Quaternion vector
    vel_body = ca.vertcat(vel_u, vel_v, vel_w) #Velocity vector in body frame
    omega_body = ca.vertcat(ome_p, ome_q, ome_r) #Angular velocity vector in body frame
    
    # Quaternion to rotation matrix
    r = sc.Rotation.from_quat(quat) #Rotation matrix from quaternion
    R = r.as_matrix() #Rotation matrix from quaternion

    # COORDINATES USING X,Y,Z: Forward, Left, Up
    # Pitch angle defined as positive about positive Y (downwards pitch: positive)

    # Actuator model update
    ele_rad, x_ele_dot = self.tail_first_order_servo_dynamics(x_ele, u_ele, self.ele_min, self.ele_max, self.ele_tau)
    rud_rad, x_rud_dot = self.tail_first_order_servo_dynamics(x_rud, u_rud, self.rud_min, self.rud_max, self.rud_tau)

    theta_sw_l_deg = self.sweep_angle_from_normalized(x_sw_ang_l)
    theta_sw_r_deg = self.sweep_angle_from_normalized(x_sw_ang_r)

    # Wing geometries
    AR_l, S_l, b_l, ac_x_l, geo_cp_l = self.wing_geometry(theta_sw_l_deg)
    AR_r, S_r, b_r, ac_x_r, geo_cp_r = self.wing_geometry(theta_sw_r_deg)
    # Negate right geo_cp y-value to account for right wing being mirrored
    geo_cp_r[1] = -geo_cp_r[1]

    # Total C.G. location and inertia
    pos_cg, I_total_body_cg = self.total_cg_and_inertia(theta_sw_l_deg, theta_sw_r_deg) #Position of C.G. in body frame relative to leading edge of wing offset from C.G. & Inertia with respect to C.G, including both wing sweeps

    # Graviational force in body frame
    F_grav = ca.inv(R) @ ca.vertcat(0.0, 0.0, -self.m*self.g) #Gravitational force in body frame (downward z in world frame acting)
    # Thrust Force
    F_thrust, omega_mot_norm_dot = self.motor_dynamics(omega_mot_norm, u_thr)
    # Wing and tail slipsteam velocities
    V_slip_wing, V_slip_tail = self.thrust_slipstream(F_thrust[0], vel_u)
    # Left wing forces and moments
    F_wing_left, M_cg_wing_left, vel_down_tail_left = self.wing_aerodynamics(V_slip_wing, vel_body, omega_body, AR_l, S_l, ac_x_l, geo_cp_l, pos_cg)
    # Right wing forces and moments
    F_wing_right, M_cg_wing_right, vel_down_tail_right = self.wing_aerodynamics(V_slip_wing, vel_body, omega_body, AR_r, S_r, ac_x_r, geo_cp_r, pos_cg)
    # Horizontal tail forces and moments
    F_hor_tail, M_cg_hor_tail = self.hor_tail_aerodynamics(V_slip_tail, vel_body, omega_body, vel_down_tail_left + vel_down_tail_right, pos_cg, ele_rad)
    # Vertical tail forces and moments
    F_vert_tail, M_cg_vert_tail = self.vertical_tail_aerodynamics(V_slip_tail, vel_body, omega_body, pos_cg, rud_rad)

    # Total forces and moments in body frame
    F_total = F_wing_left + F_wing_right + F_hor_tail + F_vert_tail + F_thrust #Force vector in body frame
    M_total = M_cg_wing_left + M_cg_wing_right + M_cg_hor_tail + M_cg_vert_tail + ca.cross(self.pos_prop - pos_cg, F_thrust) + 0.5*self.rho*(vel_u**2 + vel_v**2 + vel_w**2)*ca.vertcat(self.c_m_x_drone_static, self.c_m_y_drone_static, self.c_m_z_drone_static) #Moment vector in body frame

    Lift_total = ca.dot((R @ F_total), ca.vertcat(0.0, 0.0, 1.0)) #Lift force in body frame (upward z in world frame acting)
    Drag_total = ca.dot((R @ F_total), ca.vertcat(-1.0, 0.0, 0.0)) #Drag force in body frame (forward x in world frame acting)

    return F_total, M_total, Lift_total, Drag_total, F_grav, pos_cg, I_total_body_cg
  
  def accelerations_only(self, x, x_act_pos):
    # State vector x ordered as: [pos_x, pos_y, pos_z, vel_u, vel_v, vel_w, q_x, q_y, q_z, q_w, ome_p, ome_q, ome_r, omega_mot_norm, x_sw_sym_l_0, x_sw_sym_l_1, x_sw_sym_r_0, x_sw_sym_r_1, x_ele, x_rud]
    # Actuator state vector x_act ordered as: [omega_mot_norm, x_sw_ang_l, x_sw_ang_r, x_ele, x_rud]

    vel_u, vel_v, vel_w, ome_p, ome_q, ome_r = x[3], x[4], x[5], x[10], x[11], x[12]
    vel_body = ca.vertcat(vel_u, vel_v, vel_w) #Velocity vector in body frame
    omega_body = ca.vertcat(ome_p, ome_q, ome_r) #Angular velocity vector in body frame

    F_total, M_total, Lift_total, Drag_total, F_grav, pos_cg, I_total_body_cg = self.force_moment_cg_total(x, x_act_pos)

    uvw_dot = (F_total + F_grav)/self.m - ca.cross(omega_body, vel_body) #State derivative in body frame (acceleration in body frame, rigid body assumption)
    omega_dot = ca.inv(I_total_body_cg) @ (M_total - ca.cross(omega_body, I_total_body_cg @ omega_body)) #State derivative in body frame (angular acceleration in body frame, rigid body assumption)

    return ca.vertcat(uvw_dot, omega_dot)
  
  def actuator_dynamics(self, x_act, u):
    omega_mot_norm, x_sw_sym_l_0, x_sw_sym_l_1, x_sw_sym_r_0, x_sw_sym_r_1, x_ele, x_rud = x_act[0], x_act[1], x_act[2], x_act[3], x_act[4], x_act[5], x_act[6] #ca.fmax(ca.fmin(x_act[0], 1.0), 0.0), ca.fmax(ca.fmin(x_act[1], 1.0 - self.sweep_offset), -1.0), x_act[2], ca.fmax(ca.fmin(x_act[3], 1.0 - self.sweep_offset), -1.0), x_act[4], ca.fmax(ca.fmin(x_act[5], 1.0), -1.0), ca.fmax(ca.fmin(x_act[6], 1.0), -1.0) #Clamp states to min/max values: 0 to 1 for motor omega norm, -1 to 0.5 for sweeps, -1 to 1 for elevator and rudder
    u_thr, u_sw_l, u_sw_r, u_ele, u_rud = ca.fmax(ca.fmin(u[0], 1.0), self.throttle_offset), ca.fmax(ca.fmin(u[1], 1.0 - self.sweep_offset), -1.0), ca.fmax(ca.fmin(u[2], 1.0 - self.sweep_offset), -1.0), ca.fmax(ca.fmin(u[3], 1.0), -1.0), ca.fmax(ca.fmin(u[4], 1.0), -1.0) #Clamp inputs to min/max values: 0 to 1 for throttle, -1 to 0.5 for sweeps, -1 to 1 for elevons and rudder

    x_sw_l = ca.vertcat(x_sw_sym_l_0, x_sw_sym_l_1)
    theta_sw_l, x_sw_dot_l = self.sweep_servo_dynamics(x_sw_l, u_sw_l)
    x_sw_r = ca.vertcat(x_sw_sym_r_0, x_sw_sym_r_1)
    theta_sw_r, x_sw_dot_r = self.sweep_servo_dynamics(x_sw_r, u_sw_r)
    ele_rad, x_ele_dot = self.tail_first_order_servo_dynamics(x_ele, u_ele, self.ele_min, self.ele_max, self.ele_tau)
    rud_rad, x_rud_dot = self.tail_first_order_servo_dynamics(x_rud, u_rud, self.rud_min, self.rud_max, self.rud_tau)
    thrust, omega_mot_norm_dot = self.motor_dynamics(omega_mot_norm, u_thr)

    return ca.vertcat(omega_mot_norm_dot, x_sw_dot_l[0], x_sw_dot_l[1], x_sw_dot_r[0], x_sw_dot_r[1], x_ele_dot, x_rud_dot)
  
  def dynamics(self, x, u, case=None):  #x_sw_sym is a 2x1 column

    # State vector x ordered as: [pos_x, pos_y, pos_z, vel_u, vel_v, vel_w, q_x, q_y, q_z, q_w, ome_p, ome_q, ome_r, omega_mot_norm, x_sw_sym_l_0, x_sw_sym_l_1, x_sw_sym_r_0, x_sw_sym_r_1, x_ele, x_rud]
    # Control vector u ordered as: [u_thr, u_sw_l, u_sw_r, u_ele_dot, u_rud_dot]

    pos_x, pos_y, pos_z, vel_u, vel_v, vel_w, q_x, q_y, q_z, q_w, ome_p, ome_q, ome_r = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]
    omega_mot_norm, x_sw_sym_l_0, x_sw_sym_l_1, x_sw_sym_r_0, x_sw_sym_r_1, x_ele, x_rud = x[13], x[14], x[15], x[16], x[17], x[18], x[19] #ca.fmax(ca.fmin(x[13], 1.0), 0.0), ca.fmax(ca.fmin(x[14], 1.0 - self.sweep_offset), -1.0), x[15], ca.fmax(ca.fmin(x[16], 1.0 - self.sweep_offset), -1.0), x[17], ca.fmax(ca.fmin(x[18], 1.0), -1.0), ca.fmax(ca.fmin(x[19], 1.0), -1.0) #Clamp states to min/max values: 0 to 1 for motor omega norm, -1 to 0.5 for sweeps, -1 to 1 for elevator and rudder
    u_thr, u_sw_l, u_sw_r, u_ele, u_rud = ca.fmax(ca.fmin(u[0], 1.0), self.throttle_offset), ca.fmax(ca.fmin(u[1], 1.0 - self.sweep_offset), -1.0), ca.fmax(ca.fmin(u[2], 1.0 - self.sweep_offset), -1.0), ca.fmax(ca.fmin(u[3], 1.0), -1.0), ca.fmax(ca.fmin(u[4], 1.0), -1.0) #Clamp inputs to min/max values: 0 to 1 for throttle, -1 to 0.5 for sweeps, -1 to 1 for elevons and rudder
    
    quat = ca.vertcat(q_x, q_y, q_z, q_w) #Quaternion vector
    vel_body = ca.vertcat(vel_u, vel_v, vel_w) #Velocity vector in body frame
    omega_body = ca.vertcat(ome_p, ome_q, ome_r) #Angular velocity vector in body frame
    
    # Quaternion to rotation matrix
    r = sc.Rotation.from_quat(quat) #Rotation matrix from quaternion
    R = r.as_matrix() #Rotation matrix from quaternion

    # COORDINATES USING X,Y,Z: Forward, Left, Up
    # Pitch angle defined as positive about positive Y (downwards pitch: positive)

    # Actuator model update
    ele_rad, x_ele_dot = self.tail_first_order_servo_dynamics(x_ele, u_ele, self.ele_min, self.ele_max, self.ele_tau)
    rud_rad, x_rud_dot = self.tail_first_order_servo_dynamics(x_rud, u_rud, self.rud_min, self.rud_max, self.rud_tau)
    x_sw_l = ca.vertcat(x_sw_sym_l_0, x_sw_sym_l_1)
    x_sw_r = ca.vertcat(x_sw_sym_r_0, x_sw_sym_r_1)
    theta_sw_l_deg, x_sw_dot_l = self.sweep_servo_dynamics(x_sw_l, u_sw_l)
    theta_sw_r_deg, x_sw_dot_r = self.sweep_servo_dynamics(x_sw_r, u_sw_r)

    # Wing geometries
    AR_l, S_l, b_l, ac_x_l, geo_cp_l = self.wing_geometry(theta_sw_l_deg)
    AR_r, S_r, b_r, ac_x_r, geo_cp_r = self.wing_geometry(theta_sw_r_deg)
    # Negate right geo_cp y-value to account for right wing being mirrored
    geo_cp_r[1] = -geo_cp_r[1]

    # Total C.G. location and inertia
    pos_cg, I_total_body_cg = self.total_cg_and_inertia(theta_sw_l_deg, theta_sw_r_deg) #Position of C.G. in body frame relative to leading edge of wing offset from C.G. & Inertia with respect to C.G, including both wing sweeps

    # Graviational force in body frame
    F_grav = ca.inv(R) @ ca.vertcat(0.0, 0.0, -self.m*self.g) #Gravitational force in body frame (downward z in world frame acting)
    # Thrust Force
    F_thrust, omega_mot_norm_dot = self.motor_dynamics(omega_mot_norm, u_thr)
    # Wing and tail slipsteam velocities
    V_slip_wing, V_slip_tail = self.thrust_slipstream(F_thrust[0], vel_u)
    # Left wing forces and moments
    F_wing_left, M_cg_wing_left, vel_down_tail_left = self.wing_aerodynamics(V_slip_wing, vel_body, omega_body, AR_l, S_l, ac_x_l, geo_cp_l, pos_cg)
    # Right wing forces and moments
    F_wing_right, M_cg_wing_right, vel_down_tail_right = self.wing_aerodynamics(V_slip_wing, vel_body, omega_body, AR_r, S_r, ac_x_r, geo_cp_r, pos_cg)
    # Horizontal tail forces and moments
    F_hor_tail, M_cg_hor_tail = self.hor_tail_aerodynamics(V_slip_tail, vel_body, omega_body, vel_down_tail_left + vel_down_tail_right, pos_cg, ele_rad)
    # Vertical tail forces and moments
    F_vert_tail, M_cg_vert_tail = self.vertical_tail_aerodynamics(V_slip_tail, vel_body, omega_body, pos_cg, rud_rad)


    # Total forces and moments in body frame
    F_total = F_wing_left + F_wing_right + F_hor_tail + F_vert_tail + F_thrust + F_grav #Force vector in body frame
    M_total = M_cg_wing_left + M_cg_wing_right + M_cg_hor_tail + M_cg_vert_tail + ca.cross(self.pos_prop - pos_cg, F_thrust) + 0.5*self.rho*(vel_u**2 + vel_v**2 + vel_w**2)*ca.vertcat(self.c_m_x_drone_static, self.c_m_y_drone_static, self.c_m_z_drone_static) #Moment vector in body frame

    q_dot = quat_dot(quat, omega_body)

    uvw_dot = F_total/self.m - ca.cross(omega_body, vel_body) #State derivative in body frame (acceleration in body frame, rigid body assumption)
    omega_dot = ca.inv(I_total_body_cg) @ (M_total - ca.cross(omega_body, I_total_body_cg @ omega_body)) #State derivative in body frame (angular acceleration in body frame, rigid body assumption)

    pos_dot = R @ vel_body #Position derivative in body frame (position in world frame)

    if case == 'casadi':
      state_dot = ca.vertcat(pos_dot, uvw_dot, q_dot, omega_dot, omega_mot_norm_dot, x_sw_dot_l[0], x_sw_dot_l[1], x_sw_dot_r[0], x_sw_dot_r[1], x_ele_dot, x_rud_dot)
    elif case == 'casadi_2D':
      state_dot = ca.vertcat(pos_dot[0], pos_dot[2], uvw_dot[0], uvw_dot[2], omega_body[1], omega_dot[1], omega_mot_norm_dot, x_sw_dot_l[0], x_sw_dot_l[1], x_ele_dot)
    else:
      state_dot = [pos_dot[0], pos_dot[1], pos_dot[2], uvw_dot[0], uvw_dot[1], uvw_dot[2], q_dot[0], q_dot[1], q_dot[2], q_dot[3], omega_dot[0], omega_dot[1], omega_dot[2], omega_mot_norm_dot, x_sw_dot_l[0], x_sw_dot_l[1], x_sw_dot_r[0], x_sw_dot_r[1], x_ele_dot, x_rud_dot]

    return state_dot
    
if __name__ == "__main__":

  # #Plot Lift,Drag coefficients and Moment arm curves
  alphas =  np.linspace(-90,90,60)*(np.pi/180) #np.array([-90,-45,0,5,10,15,20])*(3.14/180) np.array([90,45,20,0,-8])*(3.14/180)
  theta_sw_syms = [-5.0, 45.0, 75.0] #np.array([-7.0, 10.0, 30.0, 45.0, 75.0])
  elevator_angs = np.array([0])*(3.14/180) #np.array([-23, -15, -5, 0, 5, 15, 23])*(3.14/180) #
  ome_vecs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) #, [0.0, 0.0, 0.0]]) #np.zeros((1,3)) #np.array([[0.0, -3.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
  vel_norm = 5.0
  betas = [0] #np.linspace(-15,15,50)*(np.pi/180) #np.array([-20, 0, 20])  #, 0, 20]) #15*(3.14/180) #Side slip angle in radians
  phis = [0] #[-np.pi/6, 0, np.pi/6] #np.linspace(-45,45,50)*(np.pi/180)
  
  obj = IndoorUAV3D()

  simulate_variable_trajectory = True

  ### Trajectory simulation ###################################################

  # Simulate dynamics for a sample input and an initial sample state
  x_sw_l_0 = -0.25
  x_sw_r_0 = -0.25
  
  u0 = np.array([[0.75, -0.25, -0.25, -0.1, 0.0]]).reshape((1,5)) #Throttle, left wing sweep, right wing sweep, elevator, rudder
  omega_mot_norm_0 = obj.motor_throttle_to_omega(u0[0,0])
  x_sw_l_0 = np.array((ca.inv(obj.A_sw) @ -(obj.B_sw @ u0[0,1]))).flatten()
  x_sw_r_0 = np.array((ca.inv(obj.A_sw) @ -(obj.B_sw @ u0[0,2]))).flatten()
  x0 = np.array([0,0,0,vel_norm,0,0,0,0,0,1,0,0,0,omega_mot_norm_0,x_sw_l_0[0],x_sw_l_0[1],x_sw_r_0[0],x_sw_r_0[1],u0[0,3],u0[0,4]])

  # print(x0)

  N_steps = 100
  t_end = 1.5

  if not simulate_variable_trajectory:
    u = np.repeat(u0, N_steps, axis=0)
  else:
    ele_amp = -0.5
    ele_mean = -0.1
    ele_freq = 2.0
    rud_amp = 0.0
    rud_mean = 0.0
    rud_freq = 5.0
    a_sw_amp = 0.0
    sw_mean = -0.25
    sw_rand_freq = 10.0
    a_sw_freq = 0.0
    thr_mean = 0.75
    thr_rand_range = 0.7
    thr_rand_freq = 5.0
    sw_rand_range = 0.75
    u = np.repeat(u0, N_steps, axis=0)
    for i in range(N_steps):
      if i % int(N_steps/(sw_rand_freq*t_end)) == 0:
        d_sw_sym = np.random.uniform(-sw_rand_range, sw_rand_range)
      if i % int(N_steps/(thr_rand_freq*t_end)) == 0:
        d_thr_sym = np.random.uniform(-thr_rand_range, thr_rand_range)

      u[i,0] = np.clip(thr_mean + d_thr_sym, 0.05, 1.0) #Add some random variation to throttle input to create a variable trajectory for testing
      u[i,1] = np.clip(sw_mean + d_sw_sym + a_sw_amp*np.sin(2 * np.pi * a_sw_freq * (i / N_steps) * t_end), -1.0, 0.5)  #Add some random variation to left wing sweep input to create a variable trajectory for testing
      u[i,2] = np.clip(sw_mean + d_sw_sym - a_sw_amp*np.sin(2 * np.pi * a_sw_freq * (i / N_steps) * t_end), -1.0, 0.5)  #Add some random variation to right wing sweep input to create a variable trajectory for testing
      u[i,3] = np.clip(ele_mean + ele_amp*np.sin(2 * np.pi * ele_freq * (i / N_steps) * t_end), -1.0, 1.0) #Vary elevator input sinusoidally to create a variable trajectory for testing
      u[i,4] = np.clip(rud_mean + rud_amp*np.sin(2 * np.pi * rud_freq * (i / N_steps) * t_end), -1.0, 1.0) #Vary rudder input sinusoidally to create a variable trajectory for testing

  t_sim, x_sim, u_sim, x_dot_sim = simulate(x0, u, N_steps, t_end, obj.dynamics)

  alpha = np.arctan(-x_sim.transpose()[5]/x_sim.transpose()[3])*(180/3.14)

  plt.plot(x_sim.transpose()[0], x_sim.transpose()[2])
  #plt.plot(x_sim.transpose()[0], x_sim.transpose()[18])


  ### FORCE ESTIMATION ###################################################
  # fig,ax = plt.subplots(1)

  # # Simulate dynamics one step given a sample input and a zero sample state
  # for ome_vec in ome_vecs:
  #   for beta in betas:
  #     for aoa in alphas:
  #       for phi in phis:
  #       # vel_w = -vel_u * np.tan(10*(3.14/180)) #np.sin(beta)*vel_norm
  #       # vel_u = np.sqrt((vel_norm**2)/(1+np.tan(10*(3.14/180))**2)) #np.sqrt((vel_norm**2)/(1+np.tan(aoa)**2))
  #       # vel_v = -vel_norm * np.sin(aoa)
  #       # vel_u = np.sqrt(vel_norm**2 - vel_w**2)
  #         r_eul = sc.Rotation.from_euler('XYZ', np.array([phi, -aoa, beta]), degrees=False)
  #         R_eul = r_eul.as_matrix() #Rotation matrix from Euler angles (yaw, pitch, roll)
  #         q_eul = r_eul.as_quat() #Quaternion vector
  #         vel_global = np.array([1,0,0])*vel_norm
  #         vel_body = R_eul.T @ vel_global #Velocity vector in body frame
  #         vel_u = vel_norm*np.cos(aoa)*np.cos(beta) #vel_body[0]
  #         vel_v = -vel_norm*np.sin(beta) #vel_body[1]
  #         vel_w = -vel_norm*np.cos(beta)*np.sin(aoa) #vel_body[2]
  #         # print(vel_body)
  #         x0 = np.array([0,0,0,vel_u,vel_v,vel_w,q_eul[0],q_eul[1],q_eul[2],q_eul[3],ome_vec[0],ome_vec[1],ome_vec[2],0,-1.0,0.0,-1.0,0.0,0.0,0.0])
  #         u0 = np.array([0.0, -1.0, -1.0, 0.0, 0.0])

  #         F_tot, M_tot, Lift_tot, Drag_tot, F_grav = obj.force_moment_cg_total(x0, u0)
  #         F_tot = np.array(F_tot).flatten()
  #         M_tot = np.array(M_tot).flatten()
  #         Fx = F_tot[0]
  #         Fy = F_tot[1]
  #         Fz = F_tot[2]

  #         sig = obj.sigmoid(aoa, obj.alpha_stall_wing, obj.M_smooth)
  #         # Lift = Fx*np.sin(np.arctan(-vel_w/vel_u)) + Fz*np.cos(np.arctan(-vel_w/vel_u))
  #         # Drag = -1*(Fx*np.cos(np.arctan(-vel_w/vel_u)) - Fz*np.sin(np.arctan(-vel_w/vel_u)))

  #         ax.scatter(aoa*(180/3.14), np.array(Lift_tot).flatten()) #np.array(Lift_tot).flatten()) #np.array(M_tot).flatten()[0], label='Mx')
      
  # plt.show()

  # x_dot = obj.dynamics(x0, u0[0])
  # a = obj.accelerations_only(x0[:13], [x0[13], x0[14], x0[16], x0[18], x0[19]])
  # act = obj.actuator_dynamics(x0[13:20], u0[0])

  # # print(x_dot)
  # # print(a)
  # # print(act)
  # plt.plot(x_sim.transpose()[0], alpha)
  # plt.plot(t_sim, x_dot_sim.transpose()[11])

  # #Compute jacobian of accelerations_only with respect to actuator state x_act
  # x_sym = ca.SX.sym('x', len(x0[:13]))
  # x_act_sym = ca.SX.sym('x_act', len(u0[0]))
  # a_sym = obj.accelerations_only(x_sym, x_act_sym)
  # a_fun = ca.Function('a_fun', [x_sym, x_act_sym], [a_sym])
  # a_jac_xact = ca.jacobian(a_sym, x_act_sym)
  # a_jac_xact_fun = ca.Function('a_jac_xact_fun', [x_sym, x_act_sym], [a_jac_xact])

  # a_jac_xact_val = a_jac_xact_fun(x0[:13], x0[[13,14,16,18,19]])
  # print(a_jac_xact_val)