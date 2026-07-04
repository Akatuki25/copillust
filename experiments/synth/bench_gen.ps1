# Throughput/quality benchmark for scaled generation (R2b prep)
$env:RENDERS_DIR = "C:\Users\akatu\develop\copillust\experiments\synth\r2a\renders_v3"
$scenes = "fem_vroid_normal_walk_full,fem_vroid_chibi_armup_bust,masc_vroid_normal_wave_high,masc_vroid_chibi_stand_full,seed_san_normal_sit_full,seed_san_chibi_walk_high,fem_vroid_normal_lie_full,masc_vroid_chibi_crouch_bust"
$py = "C:\Users\akatu\develop\copillust\experiments\synth\genv\Scripts\python.exe"
Set-Location C:\Users\akatu\develop\copillust\experiments\synth

foreach ($cfg in @(@(28,1), @(16,4), @(14,4))) {
    $steps = $cfg[0]; $batch = $cfg[1]
    $env:GEN_OUT = "C:\Users\akatu\develop\copillust\experiments\synth\r2a\bench_gen\s${steps}_b${batch}"
    Write-Output "=== steps=$steps batch=$batch ==="
    & $py generate_sketch.py --scenes $scenes --styles rough,lineart --scales 0.55,0.9 --steps $steps --batch $batch
}
Write-Output "BENCH DONE"
