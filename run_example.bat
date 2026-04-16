@echo off
call .venv\Scripts\activate.bat

python -m hyworld2.worldrecon.pipeline ^
    --input_path examples\worldrecon\realistic\Park ^
    --output_path output\park ^
    --save_rendered ^
    --render_interp_per_pair 15 ^
    --enable_bf16
