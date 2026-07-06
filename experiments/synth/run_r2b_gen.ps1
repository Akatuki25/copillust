# R2b 16K generation — detached runner (survives Claude Code session exit).
# Relaunch anytime: skips existing outputs. Log: r2b\gen_run.log
Set-Location $PSScriptRoot
$env:RENDERS_DIR = "$PSScriptRoot\r2b\renders"
$env:GEN_OUT = "$PSScriptRoot\r2b\gen"
& "$PSScriptRoot\genv\Scripts\python.exe" generate_sketch.py `
    --styles rough,pencil,lineart --scales 0.55,0.9 `
    --variants 2 --steps 16 --batch 1 *>> "$PSScriptRoot\r2b\gen_run.log"
