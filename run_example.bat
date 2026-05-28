@echo off
:: Run every HY-World 2.0 example end-to-end.
::
:: Modules:
::   --module worldrecon   WorldMirror 2.0 reconstruction (Park example). Default.
::   --module panogen      HY-Pano-2.0 panorama generation.
::   --module worldgen     WorldStereo + WorldNav 5-stage pipeline.
::   --module viewer       Open the latest gaussians.ply in show_gs.py web viewer.
::                         Override file with HY_VIEW_PLY, port with HY_VIEW_PORT (default 8081).
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

:: --- MSVC + CUDA env for gsplat JIT compile ---
:: gsplat_maskgaussian doesn't ship a precompiled .pyd — it JIT-compiles the
:: CUDA backend on first import via torch.utils.cpp_extension. That subprocess
:: runs `where cl` to find MSVC, and fails if cl.exe isn't on PATH. We source
:: vcvars64.bat once here so every python subprocess this bat spawns can
:: compile if needed. Set HY_NO_VCVARS=1 to skip (faster startup if you know
:: gsplat is already built and cached).
if not defined HY_NO_VCVARS (
    if "!VCINSTALLDIR!"=="" (
        set "INCLUDE="
        set "LIB="
        set "LIBPATH="
        set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        if exist "!VCVARS!" (
            call "!VCVARS!" >nul 2>&1
        ) else (
            echo WARN: vcvars64.bat not found at !VCVARS!. gsplat JIT compile will fail if not cached.
        )
    )
)
:: Compile for Blackwell (RTX 5090 sm_120) + common older arches. Without this
:: the JIT-compiled gsplat kernels fail at runtime with "no kernel image is
:: available for execution on the device" on a 5090.
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=8.6;8.9;9.0;12.0+PTX"
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
:: nvcc on Windows uses 6-8 GB RAM per CUDA kernel compile. gsplat defaults
:: MAX_JOBS=10 — on a 64 GB box that can OOM-kill mid-compile, leaving a
:: corrupt ninja cache. Cap at 2 for safety; bump if you have lots of RAM.
if not defined MAX_JOBS set "MAX_JOBS=2"

:: --- Windows env tweaks ---
:: PyTorch Win wheels are built without libuv. Without this, torch.distributed
:: TCPStore init crashes Stage 2 with "use_libuv was requested but PyTorch was
:: built without libuv support".
set "USE_LIBUV=0"
set "TORCH_TCPSTORE_USE_LIBUV=0"
:: gloo can't auto-pick a Windows NIC; nudge it at a real adapter.
if not defined GLOO_SOCKET_IFNAME set "GLOO_SOCKET_IFNAME=Wi-Fi"
:: transformers' async shard loader segfaults on Win + sm_120 mid-shard.
set "HF_DEACTIVATE_ASYNC_LOAD=1"
:: hf_transfer's mmap buffers compete with the DiT mmap for Win address space.
set "HF_HUB_ENABLE_HF_TRANSFER=0"
:: UTF-8 stdio so any unicode print doesn't crash cp1252.
set "PYTHONIOENCODING=utf-8"

set MODULE=all
:parse_args
if "%~1"=="" goto args_done
:: Quote the assignment so cmd doesn't fold the space before `&` into the value
:: (otherwise MODULE becomes "worldgen " and the dispatch comparison below fails).
if /I "%~1"=="--module" ( set "MODULE=%~2" & shift & shift & goto parse_args )
shift
goto parse_args
:args_done

if /I "%MODULE%"=="all"        goto run_all
if /I "%MODULE%"=="worldrecon" goto run_worldrecon
if /I "%MODULE%"=="panogen"    goto run_panogen
if /I "%MODULE%"=="worldgen"   goto run_worldgen
if /I "%MODULE%"=="viewer"     goto run_viewer
echo ERROR: unknown module %MODULE%. Use worldrecon ^| panogen ^| worldgen ^| viewer ^| all.
exit /b 2


:run_worldrecon
echo === WorldMirror 2.0 reconstruction (examples\worldrecon\realistic\Park) ===
:: Point at the local checkpoint dir from setup step 7 (download_models.py).
:: Without this, pipeline.py falls through to HuggingFace snapshot_download
:: which re-pulls the 5 GB HY-WorldMirror-2.0 weights into ~/.cache (filling C:).
:: NOTE: download_models.py wrote the checkpoint nested as
:: checkpoint/HY-WorldMirror-2.0/HY-WorldMirror-2.0/{config.json,model.safetensors}.
:: Point at the outer HY-WorldMirror-2.0 dir so --subfolder=HY-WorldMirror-2.0
:: resolves into the inner one (where the actual model files live).
python -m hyworld2.worldrecon.pipeline --pretrained_model_name_or_path "%~dp0checkpoint\HY-WorldMirror-2.0" --subfolder HY-WorldMirror-2.0 --input_path examples\worldrecon\realistic\Park --output_path output\park --save_rendered --render_interp_per_pair 15 --enable_bf16 --no_interactive
if errorlevel 1 ( echo FAIL worldrecon rc=%ERRORLEVEL% & exit /b %ERRORLEVEL% )
if /I not "%MODULE%"=="all" (
    if not defined HY_NO_AUTO_VIEW goto run_viewer
    exit /b 0
)


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
:: 0.0.0.0 is a server BIND wildcard. Using it as a CLIENT target raises
:: WinError 10049 on Windows. Force-rewrite the legacy default so a stale
:: LLM_ADDR=0.0.0.0 left in the shell environment gets coerced to loopback.
if not defined LLM_ADDR set LLM_ADDR=127.0.0.1
if "%LLM_ADDR%"=="0.0.0.0" set LLM_ADDR=127.0.0.1
if not defined LLM_PORT          set LLM_PORT=8000
if not defined LLM_NAME          set LLM_NAME=Qwen/Qwen3-VL-8B-Instruct

if not exist "%HY_WG_TARGET_PATH%" (
    echo ERROR: HY_WG_TARGET_PATH does not exist: %HY_WG_TARGET_PATH%
    exit /b 2
)
if not exist "%HY_WG_RESULT_DIR%" mkdir "%HY_WG_RESULT_DIR%" 2>nul

:: Pre-flight: WorldStereo must be in the HF cache before Stage 3 starts.
:: Without this, Stages 1-2 burn 3-5 min before Stage 3 crashes on a missing
:: model. Skip with HY_NO_WORLDSTEREO_CHECK=1.
if not defined HY_NO_WORLDSTEREO_CHECK (
    if not defined HY_WORLDSTEREO_VARIANT set HY_WORLDSTEREO_VARIANT=worldstereo-memory-dmd
    python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('hanshanxue/WorldStereo', allow_patterns=['%HY_WORLDSTEREO_VARIANT%/*.json'], local_files_only=True)" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: hanshanxue/WorldStereo ^(%HY_WORLDSTEREO_VARIANT%^) not in HF cache.
        echo Stage 3 would fail mid-pipeline. Pre-fetch first:
        echo   python download_models.py --worldstereo
        echo Or run the full setup again:
        echo   setup.bat
        echo Skip this check with: set HY_NO_WORLDSTEREO_CHECK=1
        exit /b 2
    )
)

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

:: --- VLM auto-start ---
:: traj_generate / video_gen call out to the Qwen3-VL server at LLM_ADDR:LLM_PORT.
:: Check if something's listening; if not, spawn start_vlm.bat in a new window
:: and poll until ready. Set HY_NO_AUTO_VLM=1 to skip (e.g. if you started it
:: manually or are running against a remote LLM).
if not defined HY_NO_AUTO_VLM (
    powershell -NoProfile -Command "$c=New-Object System.Net.Sockets.TcpClient; try { $c.Connect('%LLM_ADDR%', %LLM_PORT%); exit 0 } catch { exit 1 } finally { $c.Close() }" >nul 2>&1
    if errorlevel 1 (
        echo [VLM] Not running on %LLM_ADDR%:%LLM_PORT% -- starting via start_vlm.bat in new window...
        start "HY-VLM :%LLM_PORT%" cmd /k "%~dp0start_vlm.bat"
        echo [VLM] Polling for readiness ^(up to 5 min^)...
        set VLM_READY=0
        for /L %%i in (1,1,150) do (
            if !VLM_READY!==0 (
                powershell -NoProfile -Command "$c=New-Object System.Net.Sockets.TcpClient; try { $c.Connect('%LLM_ADDR%', %LLM_PORT%); exit 0 } catch { exit 1 } finally { $c.Close() }" >nul 2>&1
                if !errorlevel!==0 ( set VLM_READY=1 & echo [VLM] Ready after %%i x2s )
                if !VLM_READY!==0 timeout /t 2 /nobreak >nul 2>&1
            )
        )
        if !VLM_READY!==0 (
            echo [VLM] FAIL: server did not come up within 5 min. Check the "HY-VLM :%LLM_PORT%" window for errors.
            echo [VLM] To skip auto-start next time: set HY_NO_AUTO_VLM=1
            exit /b 1
        )
    ) else (
        echo [VLM] Already running on %LLM_ADDR%:%LLM_PORT% -- skipping start_vlm.bat
    )
) else (
    echo [VLM] HY_NO_AUTO_VLM=1 set; skipping auto-start ^(assuming external VLM^)
)

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
echo ============================================================
if not defined HY_NO_AUTO_VIEW goto run_viewer
exit /b 0


:run_worldgen_if_all
if /I "%MODULE%"=="all" goto run_worldgen
exit /b 0


:run_viewer
echo.
echo === Splat viewer (show_gs.py) ===
:: Find the most recent gaussians.ply in output/. cmd's dir /o-d sorts only
:: within each subdir of a recursive walk, not globally — use PowerShell to
:: get a real mtime-sorted result. Override HY_VIEW_PLY to pin a specific file.
if not defined HY_VIEW_PORT set HY_VIEW_PORT=8081
if defined HY_VIEW_PLY (
    set "PLY=%HY_VIEW_PLY%"
) else (
    for /f "delims=" %%P in ('powershell -NoProfile -Command "Get-ChildItem -Path '%~dp0output' -Recurse -Filter gaussians.ply 2>$null | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set "PLY=%%P"
)
if not defined PLY (
    echo ERROR: no gaussians.ply found under %~dp0output\.
    echo   Run worldrecon or worldgen first, or set HY_VIEW_PLY=^<path^> manually.
    exit /b 2
)
if not exist "%PLY%" (
    echo ERROR: PLY not found: %PLY%
    exit /b 2
)

:: show_gs.py requires position_meta_info.json next to the .ply (camera
:: position, look-at target, up-vector). worldgen Stage 5 writes it;
:: worldrecon does not. Synthesize one by computing the actual centroid +
:: bbox diagonal from the .ply — a literal default (camera at origin) places
:: the viewer INSIDE the gaussian cloud and renders black.
for %%I in ("%PLY%") do set "PLY_DIR=%%~dpI"
if not exist "%PLY_DIR%position_meta_info.json" (
    echo [viewer] position_meta_info.json missing -- computing from %PLY%
    python hyworld2\worldgen\write_position_meta.py "%PLY%"
    if errorlevel 1 (
        echo WARN: write_position_meta.py failed; using safe default
        > "%PLY_DIR%position_meta_info.json" echo {"up_direction":[0,1,0],"facing_direction":[0,0,0],"center_point":[0,0,5]}
    )
)

echo   ply  : %PLY%
echo   port : %HY_VIEW_PORT%
echo.
echo Open in your browser:  http://localhost:%HY_VIEW_PORT%/
echo Ctrl+C to stop.
echo.

:: Open the browser tab a couple seconds after the server starts.
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:%HY_VIEW_PORT%/"
:: Run viewer in foreground so Ctrl+C kills it.
pushd hyworld2\worldgen
python show_gs.py --ckpt "%PLY%" --port %HY_VIEW_PORT% --gpu_id 0
set RC=%ERRORLEVEL%
popd
exit /b %RC%


:run_all
:: Fail on first error — don't run later modules if an earlier one died.
:: Set HY_NO_AUTO_VIEW=1 to skip the splat viewer at the end.
:: We suppress auto-view for individual modules inside :run_all because we
:: want it to fire ONCE at the very end (after worldgen).
set _SAVED_HY_NO_AUTO_VIEW=%HY_NO_AUTO_VIEW%
set HY_NO_AUTO_VIEW=1
call :run_worldrecon
if errorlevel 1 ( echo ABORT: worldrecon failed; skipping panogen + worldgen & exit /b %ERRORLEVEL% )
call :run_panogen
if errorlevel 1 ( echo ABORT: panogen failed; skipping worldgen & exit /b %ERRORLEVEL% )
call :run_worldgen
set RC=%ERRORLEVEL%
:: Restore user's setting before final auto-view.
if defined _SAVED_HY_NO_AUTO_VIEW (
    set "HY_NO_AUTO_VIEW=%_SAVED_HY_NO_AUTO_VIEW%"
) else (
    set "HY_NO_AUTO_VIEW="
)
if not %RC%==0 exit /b %RC%
if not defined HY_NO_AUTO_VIEW goto run_viewer
exit /b 0

endlocal
