import numpy as np
import mujoco
import cv2

def create_vid(images):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('vid_dynamic.mp4', fourcc, 200, (width, height))
    for image in images:
        video.write(image)
    video.release()

def main(res):
    images = []
    height = 480
    width = 640
    camera_id = "front_camera"

    model_path = './myo_sim/back/myobacklegs-Exoskeleton.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=height, width=width)
    
    renderer.update_scene(data, camera=camera_id)
    images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    kf = model.keyframe('default-pose')
    data.qpos = kf.qpos

    ctrl_data = np.zeros(len(data.ctrl))
        
    for i in range(res):
        data.ctrl = ctrl_data
        mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=camera_id)
        images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    create_vid(images)


if __name__ == '__main__':
    main(res=500)


 
