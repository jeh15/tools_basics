import os

import numpy as np
import mujoco
import mujoco.viewer

import time


def main(argv=None):
    filepath = os.path.join(os.path.dirname(__file__), 'arm.xml')
    model = mujoco.MjModel.from_xml_path(filepath)

    # Update timestep:
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    # Update the initial state
    data.qpos = np.array([0.0, 0.0, -np.pi / 2])
    data.qvel = np.array([0.0, 0.0, 0.0])
    data.ctrl = np.array([0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    # Dynamic Matrix:
    M = np.zeros((model.nv, model.nv))


    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Matrices:
            mujoco.mj_fullM(model, M, data.qM)
            C = data.qfrc_bias

            # Get End Effector Position:
            end_effector_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, 'end_effector')
            end_effector_body_id = model.site_bodyid[end_effector_site_id]

            end_effector_position = data.site_xpos[end_effector_site_id]

            # Compute Jacobians:
            jac_p = np.zeros((3, model.nv))
            jac_r = np.zeros((3, model.nv))
            jacdot_p = np.zeros((3, model.nv))
            jacdot_r = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac_p, jac_r, end_effector_position, end_effector_body_id)
            mujoco.mj_jacDot(model, data, jacdot_p, jacdot_r, end_effector_position, end_effector_body_id)

            end_effector_velocity = jac_p @ data.qvel

            # Desired Acceleration:
            desired_position = np.array([0.0, 0.0, 0.3])
            desired_velocity = np.array([0.0, 0.0, 0.0])

            # PD Controller:
            kp = 10.0
            kd = 1.0
            desired_acceleration = kp * (desired_position - end_effector_position) + kd * (desired_velocity - end_effector_velocity)

            # Compute Torque:
            qdd = np.linalg.pinv(jac_p) @ (desired_acceleration - jacdot_p @ data.qvel)
            u = M @ qdd + C

            # Update the control:
            data.ctrl = u

            # Simulate the system for a few steps:
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Sleep for a bit to visualize the simulation:
            time.sleep(0.002)


if __name__ == "__main__":
    main()
