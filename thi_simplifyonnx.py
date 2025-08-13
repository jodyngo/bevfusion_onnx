import onnx
import onnx_graphsurgeon as gs
import numpy as np

model = onnx.load('/var/local/home/thungo/mmdetection3d/projects/BEVFusion/demo/bevfusion_lidar_cam_s17.onnx')
graph = gs.import_onnx(model)

for node in graph.nodes:
    if node.op == "TopK":
        print(f"Processing TopK node: {node.name}")
        print(f"Input 0 (data): {node.inputs[0]}")
        print(f"Input 1 (K): {node.inputs[1]}")

        # Replace K input with a Constant if it's a Variable or Constant
        if isinstance(node.inputs[1], (gs.Variable, gs.Constant)):
            print(f"Replacing K input with Constant K=3840")
            node.inputs[1] = gs.Constant(name=f"{node.name}_k", values=np.array([3840], dtype=np.int64))
        else:
            print(f"Unexpected input type for K: {type(node.inputs[1])}. Skipping.")

graph.cleanup().toposort()

# Export  model
modified_model = gs.export_onnx(graph)
onnx.save(modified_model, '/var/local/home/thungo/mmdetection3d/projects/BEVFusion/demo/bevfusion_lidar_cam_s17_mod.onnx')


