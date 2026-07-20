# YOLOv5 (kuna-systems) on Blackwell GPU (A6000) — Full Setup

Tested on NVIDIA Blackwell (sm_120) with TensorRT 11 and PyTorch 2.11 (cu128).

---

## 1. Clone the repo

```bash
git clone https://github.com/kuna-systems/yolov5 /root/yolov5
cd /root/yolov5
```

---

## 2. Create Python 3.10 venv

> Python 3.8 is capped at PyTorch 2.4.x which has no Blackwell (sm_120) support.
> Python 3.10 + PyTorch 2.11 (cu128) is required.

```bash
python3.10 -m venv /root/venv310 --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | /root/venv310/bin/python
```

---

## 3. Install dependencies

```bash
/root/venv310/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
/root/venv310/bin/pip install -r /root/yolov5/requirements.txt
/root/venv310/bin/pip install onnx onnxsim onnxscript
```

---

## 4. Install TensorRT 11

```bash
/root/venv310/bin/pip install tensorrt==11.0.0.114 --extra-index-url https://pypi.nvidia.com
/root/venv310/bin/pip install tensorrt-cu12-bindings==10.9.0.post1 tensorrt-cu12-libs==10.9.0.post1 --extra-index-url https://pypi.nvidia.com
```

If `import tensorrt` fails (package installed as `tensorrt_bindings` only):

```bash
TRT_SITE=$(/root/venv310/bin/python -c "import tensorrt_bindings, os; print(os.path.dirname(tensorrt_bindings.__file__))")
cp -r $TRT_SITE /root/venv310/lib/python3.10/site-packages/tensorrt
```

---

## 5. Patch experimental.py — fix torch.load for PyTorch 2.6+

PyTorch 2.6+ changed `weights_only` default to `True`, which breaks YOLOv5 checkpoint loading.

```bash
sed -i 's/ckpt = torch.load(attempt_download(w), map_location=.cpu.)/ckpt = torch.load(attempt_download(w), map_location="cpu", weights_only=False)/' /root/yolov5/models/experimental.py
```

---

## 6. Patch export.py — TRT 11 API

Three breaking changes in TRT 10+:
- `config.max_workspace_size` removed → use `set_memory_pool_limit`
- `EXPLICIT_BATCH` flag removed → now the default (use `flag = 0`)
- `build_engine` removed → use `build_serialized_network`

```bash
/root/venv310/bin/python - << 'EOF'
path = "/root/yolov5/export.py"
with open(path) as f:
    src = f.read()

src = src.replace(
    "config.max_workspace_size = workspace * 1 << 30",
    "config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * 1 << 30)"
)
src = src.replace(
    "flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))",
    "flag = 0  # EXPLICIT_BATCH removed in TRT 10, now the default"
)
src = src.replace(
    "if builder.platform_has_fast_fp16 and half:\n            config.set_flag(trt.BuilderFlag.FP16)\n        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:\n            t.write(engine.serialize())",
    "if half:\n            pass  # FP16 inferred from network dtype in TRT 10+\n        serialized = builder.build_serialized_network(network, config)\n        with open(f, 'wb') as t:\n            t.write(serialized)"
)
with open(path, "w") as f:
    f.write(src)
print("export.py patched OK")
EOF
```

---

## 7. Patch models/common.py — TRT 11 inference API

Two breaking changes in TRT 10+:
- `num_bindings` / `get_binding_*` / `binding_is_input` removed → use tensor API
- `execute_v2` removed → use `set_tensor_address` + `execute_async_v3`

```bash
/root/venv310/bin/python - << 'EOF'
path = "/root/yolov5/models/common.py"
with open(path) as f:
    src = f.read()

# Replace deprecated num_bindings / get_binding_* with tensor API (TRT 10+)
src = src.replace(
    "for index in range(model.num_bindings):\n                name = model.get_binding_name(index)\n                dtype = trt.nptype(model.get_binding_dtype(index))\n                shape = tuple(model.get_binding_shape(index))",
    "for index in range(model.num_io_tensors):  # TRT 10+\n                name = model.get_tensor_name(index)\n                dtype = trt.nptype(model.get_tensor_dtype(name))\n                shape = tuple(model.get_tensor_shape(name))"
)
src = src.replace(
    "if model.binding_is_input(index) and dtype == np.float16:",
    "if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT and dtype == np.float16:"
)

# Replace execute_v2 with set_tensor_address + execute_async_v3 (TRT 10+)
src = src.replace(
    "self.binding_addrs['images'] = int(im.data_ptr())\n            self.context.execute_v2(list(self.binding_addrs.values()))",
    "self.context.set_tensor_address('images', int(im.data_ptr()))  # TRT 10+\n            for name, ptr in self.binding_addrs.items():\n                if name != 'images':\n                    self.context.set_tensor_address(name, ptr)\n            self.context.execute_async_v3(0)"
)

with open(path, "w") as f:
    f.write(src)
print("models/common.py patched OK")
EOF
```

---

## 8. Copy weights and convert to engine

Copy weights from your local machine:

```bash
scp yolov5x_baseline_lower_lr_100ep_lower_lr_30.05.2024.pt server:/root/yolov5/
```

Convert `.pt` → `.engine`:

```bash
cd /root/yolov5
/root/venv310/bin/python export.py \
  --weights yolov5x_baseline_lower_lr_100ep_lower_lr_30.05.2024.pt \
  --include engine --img-size 640 --device 0 --half
```

---

## 9. Run PyTorch inference

```bash
cd /root/yolov5
/root/venv310/bin/python detect.py \
  --weights yolov5x_baseline_lower_lr_100ep_lower_lr_30.05.2024.pt \
  --source data/images/bus.jpg --device 0
```

---

## 10. Run TensorRT inference

```bash
cd /root/yolov5
/root/venv310/bin/python detect.py \
  --weights yolov5x_baseline_lower_lr_100ep_lower_lr_30.05.2024.engine \
  --source data/images/bus.jpg --device 0
```

---

## Expected results

| Backend    | Inference time | Notes                        |
|------------|---------------|------------------------------|
| PyTorch    | ~24 ms/image  | FP32                         |
| TensorRT   | ~2 ms/image   | FP16, ~12× faster            |

Both backends detect the same objects (e.g. on `bus.jpg`: 4 persons, 1 bus).
