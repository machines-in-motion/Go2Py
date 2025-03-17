from Go2Py.sim.mujoco import Go2Sim
import numpy as np
import os
import sys
import mim_solvers
import friction_utils
import mujoco
import pinocchio as pin
import crocoddyl
import pinocchio
import mujoco.viewer
import time
import mujoco as mj


class Go2MPC:
    def __init__(self, assets_path, HORIZON=250, friction_mu = 0.2, dt = 0.02):
        self.assets_path = assets_path
        self.HORIZON = HORIZON
        self.max_iterations = 500
        self.dt = dt
        self.urdf_path = os.path.join(assets_path, 'urdf/go2_with_arm.urdf')
        self.xml_path = os.path.join(assets_path, 'mujoco/go2_with_arm.xml')
        self.pin_robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.assets_path, pin.JointModelFreeFlyer())
        self.pinRef = pin.LOCAL_WORLD_ALIGNED
        self.friction_mu = friction_mu 
        self.rmodel = self.pin_robot.model
        self.rdata = self.pin_robot.data

        self.mpc_to_unitree_name_map = \
        {'FL_HAA_joint':'FL_hip_joint',
         'FL_HFE_joint':'FL_thigh_joint',
         'FL_KFE_joint':'FL_calf_joint',
         'FR_HAA_joint':'FR_hip_joint',
         'FR_HFE_joint':'FR_thigh_joint',
         'FR_KFE_joint':'FR_calf_joint',
         'HL_HAA_joint':'RL_hip_joint',
         'HL_HFE_joint':'RL_thigh_joint',
         'HL_KFE_joint':'RL_calf_joint',
         'HR_HAA_joint':'RR_hip_joint',
         'HR_HFE_joint': 'RR_thigh_joint',
         'HR_KFE_joint': 'RR_calf_joint',
         'Joint1':'Joint1',
         'Joint2':'Joint2',
         'Joint3':'Joint3',
         'Joint4':'Joint4',
         'Joint5':'Joint5',
         'Joint6':'Joint6'
         }
        self.unitree_to_mpc_name_map = {val:key for key, val in self.mpc_to_unitree_name_map.items()}
        self.unitree_joint_order = ['FR', 'FL', 'RR', 'RL']

        mpc_to_unitree_idx_map = {}
        unitree_id = 0
        for foot_name in self.unitree_joint_order:
            for actuator in ['_hip_joint', '_thigh_joint', '_calf_joint']:
                unitree_actuator_name = foot_name+actuator
                mpc_joint_name = self.unitree_to_mpc_name_map[unitree_actuator_name]
                mpc_joint_id = self.rmodel.getJointId(mpc_joint_name)
                mpc_to_unitree_idx_map[mpc_joint_id-2]=unitree_id
                unitree_id+=1

        for unitree_actuator_name in ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']:
                mpc_joint_name = self.unitree_to_mpc_name_map[unitree_actuator_name]
                mpc_joint_id = self.rmodel.getJointId(mpc_joint_name)
                mpc_to_unitree_idx_map[mpc_joint_id-2]=unitree_id
                unitree_id+=1

        self.mpc_to_unitree_idx = np.zeros(18).astype(np.int32)    # mpc_state[mpc_to_unitree_idx] -> state/command in unitree order 
        self.unitree_to_mpc_idx = np.zeros(18).astype(np.int32)        # unitree_state[unitree_to_mpc] -> state/command in mpc order
        for mpc_idx, unitree_idx in mpc_to_unitree_idx_map.items():
            self.mpc_to_unitree_idx[mpc_idx] = unitree_idx
            self.unitree_to_mpc_idx[unitree_idx] = mpc_idx

        # set contact frame_names and_indices
        ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT', 'Link6']
        self.lfFootId = self.rmodel.getFrameId(ee_frame_names[0])
        self.rfFootId = self.rmodel.getFrameId(ee_frame_names[1])
        self.lhFootId = self.rmodel.getFrameId(ee_frame_names[2])
        self.rhFootId = self.rmodel.getFrameId(ee_frame_names[3])
        self.armEEId = self.rmodel.getFrameId(ee_frame_names[4])
        self.running_models = []
        self.constraintModels = []
        
        self.ccdyl_state = crocoddyl.StateMultibody(self.rmodel)
        self.ccdyl_actuation = crocoddyl.ActuationModelFloatingBase(self.ccdyl_state)
        self.nu = self.ccdyl_actuation.nu

    def initialize(self, q0=np.array([0.0, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0] 
                    +4*[0.0, 0.77832842, -1.56065452] + [0.0, 0.3, -0.3, 0.0, 0.0, 0.0]
                        )):
        q0[11+2]=0.0
        self.q0 = q0.copy()
        self.x0 =  np.concatenate([q0, np.zeros(self.rmodel.nv)])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        self.rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        self.rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        self.lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        self.lhFootPos0 = self.rdata.oMf[self.lhFootId].translation 
        self.armEEPos0 = self.rdata.oMf[self.armEEId].translation
        self.armEEOri0 = self.rdata.oMf[self.armEEId].rotation
        self.supportFeetIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
        # self.supportFeePos = [self.lfFootPos0, self.rfFootPos0, self.lhFootPos0, self.rhFootPos0]
        self.xs = [self.x0]*(self.HORIZON + 1)
        self.createProblem()
        self.createSolver()
        self.us = self.solver.problem.quasiStatic([self.x0]*self.HORIZON) 

    def createProblem(self):
        #First compute the desired state of the robot
        comRef = (self.rfFootPos0 + self.rhFootPos0 + self.lfFootPos0 + self.lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, self.q0)[2].item() 
        eeDes = self.armEEPos0
        comDes = []
        dt = self.dt
        # radius = 0.065
        radius = 0.0
        for t in range(self.HORIZON+1):
            comDes_t = comRef.copy()
            w = (2 * np.pi) * 0.2 # / T
            comDes_t[0] += radius * np.sin(w * t * dt) 
            comDes_t[1] += radius * (np.cos(w * t * dt) - 1)
            comDes += [comDes_t]
        body_com_ref = comDes
        arm_eff_ref = [eeDes]*(self.HORIZON+1)
        # Now define the model
        for t in range(self.HORIZON+1):
            self.contactModel = crocoddyl.ContactModelMultiple(self.ccdyl_state, self.nu)
            costModel = crocoddyl.CostModelSum(self.ccdyl_state, self.nu)

            # Add contacts
            for frame_idx in self.supportFeetIds:
                support_contact = crocoddyl.ContactModel3D(self.ccdyl_state, frame_idx, np.array([0., 0., 0.]), self.pinRef, self.nu, np.array([0., 0.]))
                self.contactModel.addContact(self.rmodel.frames[frame_idx].name + "_contact", support_contact) 

            # Contact for the EE
            support_contact = crocoddyl.ContactModel3D(self.ccdyl_state, self.armEEId, np.array([0., 0., 0.0]), pin.LOCAL_WORLD_ALIGNED, self.nu, np.array([0., 0.]))
            self.contactModel.addContact(self.rmodel.frames[self.armEEId].name + "_contact", support_contact) 
            # Add state/control regularization costs
            state_reg_weight, control_reg_weight = 1e-1, 1e-3

            freeFlyerQWeight = [0.]*3 + [500.]*3
            freeFlyerVWeight = [10.]*6
            legsQWeight = [0.01]*(self.rmodel.nv - 6)
            legsWWeights = [1.]*(self.rmodel.nv - 6)
            stateWeights = np.array(freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights)    
            stateResidual = crocoddyl.ResidualModelState(self.ccdyl_state, self.x0, self.nu)

            stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
            stateReg = crocoddyl.CostModelResidual(self.ccdyl_state, stateActivation, stateResidual)

            if t == self.HORIZON:
                costModel.addCost("stateReg", stateReg, state_reg_weight*self.dt)
            else:
                costModel.addCost("stateReg", stateReg, state_reg_weight)

            if t != self.HORIZON:
                ctrlResidual = crocoddyl.ResidualModelControl(self.ccdyl_state, self.nu)
                ctrlReg = crocoddyl.CostModelResidual(self.ccdyl_state, ctrlResidual)
                costModel.addCost("ctrlReg", ctrlReg, control_reg_weight)      


            # Body COM Tracking Cost
            com_residual = crocoddyl.ResidualModelCoMPosition(self.ccdyl_state, body_com_ref[t], self.nu)
            com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            com_track = crocoddyl.CostModelResidual(self.ccdyl_state, com_activation, com_residual) # What does it correspond to exactly?
            # if t == self.HORIZON:
            #     # costModel.addCost("comTrack", com_track, 1e5)
            #     costModel.addCost("comTrack", com_track, 1e5)
            # else:
            #     costModel.addCost("comTrack", com_track, 1e1)

            # End Effecor Position Tracking Cost
            # ef_residual = crocoddyl.ResidualModelFrameTranslation(self.ccdyl_state, self.armEEId, arm_eff_ref[t], self.nu) # Check this cost term            
            # ef_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            # ef_track = crocoddyl.CostModelResidual(self.ccdyl_state, ef_activation, ef_residual)

            ef_rotation_residual = crocoddyl.ResidualModelFrameRotation(self.ccdyl_state, self.armEEId, self.armEEOri0, self.nu)
            ef_rot_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            ef_rot_track = crocoddyl.CostModelResidual(self.ccdyl_state, ef_rot_activation, ef_rotation_residual)
            if t == self.HORIZON:
                # costModel.addCost("efTrack", ef_track, 1e5)
                costModel.addCost("efRotTrack", ef_rot_track, 1e5)
                
            else:
                # costModel.addCost("efTrack", ef_track, 1e1)
                costModel.addCost("efRotTrack", ef_rot_track, 1e1)
            # Force tracking term
            self.ef_des_force = pin.Force()
            self.ef_des_force.linear[2] = 0
            contact_force_residual = crocoddyl.ResidualModelContactForce(self.ccdyl_state, self.armEEId, self.ef_des_force, 3, self.nu)
            contact_force_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
            contact_force_track = crocoddyl.CostModelResidual(self.ccdyl_state, contact_force_activation, contact_force_residual)
            if t == self.HORIZON:
                costModel.addCost("contact_force_track", contact_force_track, 1e5)
            else:
                costModel.addCost("contact_force_track", contact_force_track, 1e1)
                
    
            # Friction Cone Constraints
            constraintModelManager = crocoddyl.ConstraintModelManager(self.ccdyl_state, self.ccdyl_actuation.nu)
            if(t != self.HORIZON):
                for frame_idx in self.supportFeetIds:
                    name = self.rmodel.frames[frame_idx].name + "_contact"
                    residualFriction = friction_utils.ResidualFrictionCone(self.ccdyl_state, name, self.friction_mu, self.ccdyl_actuation.nu)
                    constraintFriction = crocoddyl.ConstraintModelResidual(self.ccdyl_state, residualFriction, np.array([0.]), np.array([np.inf]))
                    constraintModelManager.addConstraint(name + "friction", constraintFriction)
            #friction model for the arm
            # name = self.rmodel.frames[self.armEEId].name + "_contact"
            # residualFriction = friction_utils.ResidualFrictionCone(self.ccdyl_state, name, self.friction_mu, self.ccdyl_actuation.nu)
            # constraintFriction = crocoddyl.ConstraintModelResidual(self.ccdyl_state, residualFriction, np.array([0.]), np.array([np.inf]))
            # constraintModelManager.addConstraint(name + "friction", constraintFriction)
            
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.ccdyl_state, self.ccdyl_actuation, self.contactModel, costModel, constraintModelManager, 0., True)
            model = crocoddyl.IntegratedActionModelEuler(dmodel, self.dt)
            self.running_models += [model]
        self.ocp = crocoddyl.ShootingProblem(self.x0, self.running_models[:-1], self.running_models[-1])
        
    def createSolver(self):
        solver = mim_solvers.SolverCSQP(self.ocp)
        solver.max_qp_iters = 25
        solver.with_callbacks = True
        solver.use_filter_line_search = False
        solver.termination_tolerance = 1e-2
        solver.eps_abs = 1e-2
        solver.eps_rel = 0.
        self.solver = solver

    def getSolution(self, k=None):
        if k is None: 
            x_idx = 1
            u_idx = 0
        else:
            x_idx = k
            u_idx = k
        t = self.xs[x_idx][:3]
        quat = self.xs[x_idx][3:7]
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]
        q = self.xs[x_idx][7:25]
        eta = self.xs[x_idx][25:25+6]
        dq = self.xs[x_idx][25+6:]
        constraint_norm = self.solver.constraint_norm
        return dict(
            position=t,
            orientation=np.array([qw, qx, qy, qz]), #Mujoco and uniree quaternion order
            velocity = eta[:3],
            omega = eta[3:],
            q = q[self.mpc_to_unitree_idx],
            dq = dq[self.mpc_to_unitree_idx], 
            tau = self.us[u_idx][[self.mpc_to_unitree_idx]],
            constraint_norm = constraint_norm
        )
    
    def updateAndSolve(self, t, quat, q, v, omega, dq):
        q_ = np.zeros(self.rmodel.nq)
        dq_ = np.zeros(self.rmodel.nv)
        qw = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]
        q_[:3] = t
        q_[3:7] = np.array([qx, qy, qz, qw])
        q_[7:] = q[self.unitree_to_mpc_idx]
        dq_[:3] = v
        dq_[3:6] = omega
        dq_[6:] = dq[self.unitree_to_mpc_idx]
        pin.framesForwardKinematics(self.rmodel, self.rdata, q_)
        x = np.hstack([q_, dq_])
        self.solver.problem.x0 = x
        self.xs = list(self.solver.xs[1:]) + [self.solver.xs[-1]]
        self.xs[0] = x
        self.us = list(self.us[1:]) + [self.us[-1]] 
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()
    
    def solve(self):
        self.solver.solve(self.xs, self.us, self.max_iterations)
        self.xs, self.us = self.solver.xs, self.solver.us
        return self.getSolution()


def getForceSensor(model, data, site_id):
    site_id = mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE, 'EF_force_site')
    world_R_sensor = data.xmat[site_id].reshape(3,3).T
    force_in_body = data.sensordata[-3:].reshape(3,1)
    force_in_world = world_R_sensor@force_in_body
    return force_in_world


# Instantiate the simulator
robot=Go2Sim(with_arm=True)
map = np.zeros((1200, 1200))
map[:,649:679] = 400
robot.updateHeightMap(map)

# Instantiate the solver
assets_path = '/home/Go2py/Go2Py/assets'
mpc = Go2MPC(assets_path, HORIZON=20, friction_mu=0.1)
mpc.initialize()
mpc.solve()
m = list(mpc.solver.problem.runningModels) + [mpc.solver.problem.terminalModel]

# Reset the robot
state = mpc.getSolution(0)
robot.pos0 = state['position']
robot.rot0 = state['orientation']
robot.q0 = state['q']
robot.reset()

# Solve for as many iterations as needed for the first step
mpc.max_iterations=500

measured_forces = []
forces = np.linspace(0, 25, 500)
breakpoint()
for i in range(500):
    # set the force setpoint
    for action_model in m:
        action_model.differential.costs.costs['contact_force_track'].cost.residual.reference.linear[:] = np.array([-forces[i], 0, 0])

    state = robot.getJointStates()
    q = state['q']
    dq = state['dq']
    t, quat = robot.getPose()
    v = robot.data.qvel[:3]
    omega = robot.data.qvel[3:6]
    q = np.hstack([q, np.zeros(2)])
    dq = np.hstack([dq, np.zeros(2)])
    solution = mpc.updateAndSolve(t, quat, q, v, omega, dq)
    # Reduce the max iteration count to ensure real-time execution
    mpc.max_iterations=100
    q = solution['q']
    dq = solution['dq']
    tau = solution['tau'].squeeze()
    kp = np.ones(18)*0.
    kv = np.ones(18)*0.
    # Step the physics

    for j in range(int(mpc.dt//robot.dt)):
        robot.setCommands(q, dq, kp, kv, tau)
        robot.step()

    force_site_to_sensor_idx = {'FL_force_site': 0, 'FR_force_site': 1, 'RL_force_site': 2, 'RR_force_site': 3, 'EF_force_site': 4}
    force_sensor_site = 'EF_force_site'
    measured_forces.append(getForceSensor(robot.model, robot.data, force_sensor_site).squeeze().copy())
measured_forces = np.array(measured_forces)

# Visualize the measured force against the desired
import matplotlib.pyplot as plt
plt.plot(measured_forces[:300,0], '*')
plt.plot(forces,'k')
plt.show()
robot.close()
