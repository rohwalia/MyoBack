import numpy as np
import mujoco
import cv2
import matplotlib.pyplot as plt
import csv
import opensim as osim

def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid.mp4', fourcc, 200, (width, height))
    for image in images:
        video.write(image)
    video.release()

def main(res, sites):
    images = []
    height = 480
    width = 640
    camera_id = "front_camera"

    model_path = './myosuite/myosuite/simhive/myo_sim/back/myoback_v2.0.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    print(len(data.qpos))

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)  # Get joint name
        qpos_address = model.jnt_qposadr[joint_id]  # Get the qpos index for the joint
        print(f"Joint Name: {joint_name}, qpos index: {qpos_address}")

    kf = model.keyframe('default-pose')
    data.qpos = kf.qpos

    mujoco.mj_forward(model, data)

    # Initialize qpos_flex
    qpos_flex = np.zeros((res, model.nq))  # res is the number of steps, model.nq is the number of generalized coordinates

    joint_val = np.linspace(-1, 0.3, res)[::-1]

    qpos_flex[:, 0] = 0.03305523 * joint_val
    qpos_flex[:, 1] = 0.01101841 * joint_val
    qpos_flex[:, 2] = 0.6 * joint_val
    qpos_flex[:, 3] = 0.03*joint_val
    qpos_flex[:, 4] = (3.89329927e-04 * joint_val**4 - 4.18762151e-03 * joint_val**3 -
                        1.86233838e-02 * joint_val**2 + 5.78749087e-02 * joint_val)
    qpos_flex[:, 5] = 0.64285726 * joint_val
    qpos_flex[:, 9] = 0.185 * joint_val
    qpos_flex[:, 12] = 0.204 * joint_val
    qpos_flex[:, 15] = 0.231 * joint_val
    qpos_flex[:, 18] = 0.255 * joint_val
    # qpos_flex[:, 6] = joint_val

    sites_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in sites]
    print(sites_id)
    kinematics_mj = []
    # sites_id2 = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in ["LTpT_T4_r_LTpT_t4_R-P1"]]
    # kinematics_mj2 = []

    model_osim = osim.Model('/home/rwalia/converter_osim_xml/Lumbar_C_210.osim')
    state = model_osim.initSystem()
    coordinate_name = "flex_extension"
    flex_extension_coord = model_osim.getCoordinateSet().get(coordinate_name)
    parent_frame = model_osim.getBodySet().get("lumbar1")
    kinematics_osim = []

    # print(data.xmat[sites_id[0]].reshape(3, 3))

    for i in range(res):
        data.qpos = qpos_flex[i]
        mujoco.mj_forward(model, data)
        pos = [np.copy(data.xpos[site_id]) for site_id in sites_id]
        # pos2 = [np.copy(data.site_xpos[site_id]+np.dot(data.xmat[sites_id[0]].reshape(3, 3), [0.0457852, -0.00705986, -0.00282])) for site_id in sites_id2]
        kinematics_mj.append(pos)
        # kinematics_mj2.append(pos2)
        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

        flex_extension_coord.setValue(state, joint_val[i])
        model_osim.realizePosition(state)
        position_frame = parent_frame.getPositionInGround(state)
        kinematics_osim.append([np.array([position_frame[0], position_frame[1], position_frame[2]])])


    kinematics_mj = np.array(kinematics_mj)
    kinematics_osim = np.array(kinematics_osim)
    # kinematics_mj2 = np.array(kinematics_mj2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # First subplot
    ax1.plot(-np.degrees(joint_val), kinematics_mj[:,0][:,2], label="Mujoco")
    ax1.plot(-np.degrees(joint_val), kinematics_osim[:,0][:,1], label="OSIM")
    ax1.set_xlabel("Flex extension [°]")
    ax1.set_ylabel("Z-axis")
    ax1.legend()

    # Second subplot
    ax2.plot(-np.degrees(joint_val), kinematics_mj[:,0][:,0], label="Mujoco")
    ax2.plot(-np.degrees(joint_val), kinematics_osim[:,0][:,0], label="OSIM")
    ax2.set_xlabel("Flex extension [°]")
    ax2.set_ylabel("X-axis")
    ax2.legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

    create_vid(images)


if __name__ == '__main__':
    main(res=100, sites=["lumbar1"])