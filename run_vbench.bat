@echo off
call .venv\Scripts\activate.bat

python run_vbench.py tencent/HY-World-2.0 ^
    --output_dir results_vbench/videos ^
    --num_samples 1 ^
    --image_types "indoor,scenery" ^
    --target_size 952 ^
    --render_interp_per_pair 15 ^
    --enable_bf16 True
