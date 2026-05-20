@echo off
:: Run every HY-World 2.0 example end-to-end.
::
:: Modules:
::   --module worldrecon   WorldMirror 2.0 reconstruction (Park example). Default.
::   --module panogen      HY-Pano-2.0 panorama generation.
::   --module worldgen     WorldStereo + WorldNav 5-stage pipeline.
::   --module all          Run worldrecon, then panogen, then worldgen (if prereqs set).
::
:: panogen prereqs:
::   set HY_PANO_MODEL_DIR=C:\models\HY-Pano-2.0
::   set HY_PANO_INPUT_PNG=C:\some\image.png
::
:: worldgen prereqs:
::   - vLLM server reachable on LLM_ADDR:LLM_PORT (default 0.0.0.0:8000)
::   - custom CUDA builds compiled (gsplat_maskgaussian, navmesh — see worldgen README)
::   set HY_WG_TARGET_PATH=examples\worldgen\case000   (default)
::   set HY_WG_RESULT_DIR=output\worldgen              (default)
::   set HY_WG_GPUS=1                                  (default 1)
::   set LLM_ADDR=0.0.0.0 LLM_PORT=8000 LLM_NAME=Qwen/Qwen3-VL-8B-Instruct

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat

set MODULE=all
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--module" ( set MODULE=%~2 & shift & shift & goto parse_args )
shift
goto parse_args
:args_done

if /I "%MODULE%"=="all"        goto run_all
if /I "%MODULE%"=="worldrecon" goto run_worldrecon
if /I "%MODULE%"=="panogen"    goto run_panogen
if /I "%MODULE%"=="worldgen"   goto run_worldgen
echo ERROR: unknown module %MODULE%. Use worldrecon ^| panogen ^| worldgen ^| all.
exit /b 2


:run_worldrecon
echo === WorldMirror 2.0 reconstruction (examples\worldrecon\realistic\Park) ===
python -m hyworld2.worldrecon.pipeline --input_path examples\worldrecon\realistic\Park --output_path output\park --save_rendered --render_interp_per_pair 15 --enable_bf16
if errorlevel 1 ( echo FAIL worldrecon rc=%ERRORLEVEL% & exit /b %ERRORLEVEL% )
if /I not "%MODULE%"=="all" exit /b 0


:run_panogen
echo.
echo === HY-Pano 2.0 panorama generation ===
if not defined HY_PANO_MODEL_DIR (
    echo SKIP panogen: HY_PANO_MODEL_DIR not set ^(point at HY-Pano-2.0 weights^).
    if /I "%MODULE%"=="all" goto :run_worldgen_if_all
    exit /b 2
)
if not defined HY_PANO_INPUT_PNG (
    echo SKIP panogen: HY_PANO_INPUT_PNG not set ^(point at an input image^).
    if /I "%MODULE%"=="all" goto :run_worldgen_if_all
    exit /b 2
)
if not defined HY_PANO_PROMPT set HY_PANO_PROMPT=Expand this image to a 360-degree equirectangular panorama. Maintain realistic style.
pushd hyworld2\panogen
python pipeline.py --pretrained-model-name-or-path "%HY_PANO_MODEL_DIR%" --subfolder "" --image "%HY_PANO_INPUT_PNG%" --prompt "%HY_PANO_PROMPT%" --save "%~dp0output\panorama.png"
set RC=%ERRORLEVEL%
popd
if not %RC%==0 ( echo FAIL panogen rc=%RC% & exit /b %RC% )
if /I not "%MODULE%"=="all" exit /b 0
goto :run_worldgen_if_all


:run_worldgen
echo.
echo === HY-World 2.0 world generation (5-stage pipeline) ===
if not defined HY_WG_TARGET_PATH set HY_WG_TARGET_PATH=%~dp0examples\worldgen\case000
if not defined HY_WG_RESULT_DIR  set HY_WG_RESULT_DIR=%~dp0output\worldgen
if not defined HY_WG_GPUS        set HY_WG_GPUS=1
if not defined LLM_ADDR          set LLM_ADDR=0.0.0.0
if not defined LLM_PORT          set LLM_PORT=8000
if not defined LLM_NAME          set LLM_NAME=Qwen/Qwen3-VL-8B-Instruct

if not exist "%HY_WG_TARGET_PATH%" (
    echo ERROR: HY_WG_TARGET_PATH does not exist: %HY_WG_TARGET_PATH%
    exit /b 2
)
if not exist "%HY_WG_RESULT_DIR%" mkdir "%HY_WG_RESULT_DIR%" 2>nul

:: stage-5 max_steps auto-scale by GPU count when user didn't pin it.
if not defined HY_WG_MAX_STEPS (
    if "%HY_WG_GPUS%"=="1" set HY_WG_MAX_STEPS=8000
    if "%HY_WG_GPUS%"=="2" set HY_WG_MAX_STEPS=4000
    if "%HY_WG_GPUS%"=="4" set HY_WG_MAX_STEPS=2000
    if "%HY_WG_GPUS%"=="8" set HY_WG_MAX_STEPS=1500
    if not defined HY_WG_MAX_STEPS set HY_WG_MAX_STEPS=2000
)

:: Visible GPU mask "0,1,2,...,N-1".
set HY_WG_CUDA=0
for /L %%i in (1,1,7) do ( if %%i lss %HY_WG_GPUS% set HY_WG_CUDA=!HY_WG_CUDA!,%%i )

echo   target_path : %HY_WG_TARGET_PATH%
echo   result_dir  : %HY_WG_RESULT_DIR%
echo   gpus        : %HY_WG_GPUS%  (CUDA_VISIBLE_DEVICES=%HY_WG_CUDA%)
echo   max_steps   : %HY_WG_MAX_STEPS%
echo   vLLM        : http://%LLM_ADDR%:%LLM_PORT%  (%LLM_NAME%)
echo.

pushd hyworld2\worldgen

echo --- Stage 1/5: Trajectory Planning (traj_generate.py) ---
python traj_generate.py --target_path "%HY_WG_TARGET_PATH%" --llm_addr %LLM_ADDR% --llm_port %LLM_PORT% --llm_name "%LLM_NAME%" --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm
if errorlevel 1 ( set RC=%ERRORLEVEL% & echo FAIL stage1 rc=!RC! & popd & exit /b !RC! )

echo --- Stage 2/5: Trajectory Rendering (traj_render.py) ---
set CUDA_VISIBLE_DEVICES=%HY_WG_CUDA%
torchrun --nproc_per_node %HY_WG_GPUS% traj_render.py --target_path "%HY_WG_TARGET_PATH%" --llm_addr %LLM_ADDR% --llm_port %LLM_PORT% --llm_name "%LLM_NAME%"
if errorlevel 1 ( set RC=%ERRORLEVEL% & echo FAIL stage2 rc=!RC! & popd & exit /b !RC! )

echo --- Stage 3/5: World Expansion - Keyframe Generation (video_gen.py) ---
torchrun --nproc_per_node %HY_WG_GPUS% video_gen.py --target_path "%HY_WG_TARGET_PATH%" --fsdp
if errorlevel 1 ( set RC=%ERRORLEVEL% & echo FAIL stage3 rc=!RC! & popd & exit /b !RC! )

echo --- Stage 4/5: Build GS Training Data (gen_gs_data.py) ---
torchrun --nproc_per_node %HY_WG_GPUS% gen_gs_data.py --root_path "%HY_WG_TARGET_PATH%" --save_normal --split_sky
if errorlevel 1 ( set RC=%ERRORLEVEL% & echo FAIL stage4 rc=!RC! & popd & exit /b !RC! )

echo --- Stage 5/5: 3DGS Training (world_gs_trainer) ---
python -m world_gs_trainer default --data_dir "%HY_WG_TARGET_PATH%\gs_data" --result_dir "%HY_WG_RESULT_DIR%" --max_steps %HY_WG_MAX_STEPS% --save_steps %HY_WG_MAX_STEPS% --eval_steps %HY_WG_MAX_STEPS% --ply_steps %HY_WG_MAX_STEPS% --save_ply --convert_to_spz --disable_video --use_scale_regularization --antialiased --depth_loss --normal_loss --sky_depth_from_pcd --use_mask_gaussian --mask_export_stochastic --no-mask-export-anchor-protection --use_anchor_protection --export_mesh --strategy.refine-start-iter 150 --strategy.refine-stop-iter 750 --strategy.refine-every 100 --strategy.refine-scale2d-stop-iter 750 --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1
set RC=%ERRORLEVEL%
popd
if not %RC%==0 ( echo FAIL stage5 rc=%RC% & exit /b %RC% )

echo.
echo ============================================================
echo worldgen done. Inspect %HY_WG_RESULT_DIR% (.ply / .spz / mesh).
echo Viewer:  pushd hyworld2\worldgen ^&^& python show_gs.py --port 8081 --gpu_id 0 --ckpt "%HY_WG_RESULT_DIR%\ckpts\ckpt_*_rank*.pt"
echo ============================================================
exit /b 0


:run_worldgen_if_all
if /I "%MODULE%"=="all" goto run_worldgen
exit /b 0


:run_all
call :run_worldrecon
call :run_panogen
call :run_worldgen
exit /b 0

endlocal
