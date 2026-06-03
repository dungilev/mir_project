Local configuration notes for `src/mir_robot/tm` (auto-generated)

1. Usage

   Source the project local environment before running demo scripts:

   ```bash
   source /home/dung/mir_project/local_setup.sh
   ```

2. Recommended environment variables (already set in `local_setup.sh`):
   - `KHOANGCACH_DEVICE`: set to `0` to use GPU, or `cpu` to force CPU.
   - `KHOANGCACH_USE_REALSENSE`: `0` (off) unless you have an Intel RealSense camera.
   - `KHOANGCACH_USE_ONNX`: `0` by default.
   - `KHOANGCACH_INFER_EVERY`: process every N frames (default 2).

3. Notes from system check on this laptop:
   - CPU: AMD Ryzen 9 5900HX (16 threads)
   - RAM: 15 GiB (approx half free)
   - GPU: NVIDIA GeForce RTX 3050 Mobile (4 GiB) detected — GPU enabled by default.
   - Disk: root partition ~37 GiB total, ~5.4 GiB free (low disk space; consider cleaning large files).

4. If you want CPU-only mode, set `KHOANGCACH_DEVICE=cpu` before running scripts.
