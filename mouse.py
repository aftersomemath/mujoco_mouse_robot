import multiprocessing as mp
from multiprocessing import Process, Queue
import os

import cv2
import mujoco
import mujoco.viewer as viewer
import numpy as np

# The "vision process" renders images in a totally seperate process from the MuJoCo viewer/physics thread
# It is needed on MacOS because currently MuJoCo does not support OpenGL rendering off of the main
# thread on MacOS.
#
# On MacOS all live visualization (e.g. cv2.imshow() calls) need to occur in this process, because the
# physics thread (and the on_control callback) cannot have a UI
def create_vision_process(m, cam_name, cam_res):
    mp.set_start_method('spawn')
    req_q = Queue()
    ret_q = Queue()
    p = Process(target=vision_process,
                args=(req_q, ret_q, m, cam_name, cam_res),
                daemon=True)
    p.start()
    return req_q, ret_q, p

# req_q is a multiprocessing Queue which will contain MuJoCo "data" objects for rendering
# ret_q is a multiprocessing Queue which will return the image
def vision_process(req_q, ret_q, m, cam_name, cam_res):
    # Make all the things needed to render a simulated camera
    gl_ctx = mujoco.GLContext(*cam_res)
    gl_ctx.make_current()

    scn = mujoco.MjvScene(m, maxgeom=100)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

    vopt = mujoco.MjvOption()
    vopt.geomgroup[1] = 0 # Group 1 is the mocap markers for visualization
    pert = mujoco.MjvPerturb()

    ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)

    viewport = mujoco.MjrRect(0, 0, *cam_res)

    image = np.empty((cam_res[1], cam_res[0], 3), dtype=np.uint8)
    while True:
        d = req_q.get()

        mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(viewport, scn, ctx)
        mujoco.mjr_readPixels(image, None, viewport, ctx)
        image = cv2.flip(image, 0) # OpenGL renders with inverted y axis
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow('image', image)
        cv2.waitKey(1)

        ret_q.put(None)

# on_control callback for physics simulation
last_render_t = 0.0
def control(m, d, req_q, res_q):
  try:
    d.actuator('lwheel').ctrl[0] = -1
    d.actuator('rwheel').ctrl[0] = 1

    # Render the camera on the mouse using MacOS workaround
    global last_render_t
    if d.time - last_render_t > 1/60.0:
      last_render_t = d.time

      req_q.put(d)
      result = res_q.get() # For now result is None, it should be the result of processing the image

  except Exception as e:
    print(e)

def load_callback(m=None, d=None, xml_path=None, req_q=None, res_q=None):
  mujoco.set_mjcb_control(None)

  m = mujoco.MjModel.from_xml_path(xml_path)
  d = mujoco.MjData(m)

  if m is not None:
    mujoco.set_mjcb_control(lambda m, d: control(m, d, req_q, res_q))

  return m, d

if __name__ == '__main__':
  xml_path = os.path.abspath('mouse.xml')
  m = mujoco.MjModel.from_xml_path(xml_path)
  req_q, res_q, p = create_vision_process(m, 'mouse', (640, 480))

  viewer.launch(loader=lambda m=None, d=None: load_callback(m, d, xml_path, req_q, res_q))
