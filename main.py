import mujoco
from mujoco import viewer

model_path = r"D:\Users\mimim\Documents\FYP\mujoco_menagerie\boston_dynamics_spot\scene.xml"  # or humanoid.xml
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    
    while v.is_running():   # keep running until you close window
        mujoco.mj_step(model, data)
        v.sync()
