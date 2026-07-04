# R2b scale render: 64-seed sweep, line pass only, 4 parallel Blender workers
$b = "$env:ProgramFiles\Blender Foundation\Blender 5.1\blender.exe"
$root = "C:\Users\akatu\develop\copillust\experiments\synth"
New-Item -ItemType Directory -Force "$root\r2b\renders" | Out-Null
$ranges = @(@(100,115), @(116,131), @(132,147), @(148,163))
$jobs = foreach ($r in $ranges) {
    Start-Job -ScriptBlock {
        param($b, $root, $s0, $s1)
        for ($s = $s0; $s -le $s1; $s++) {
            foreach ($m in @('fem_vroid','masc_vroid','seed_san')) {
                & $b -b -P "$root\blender_render.py" -- --vrm "$root\assets\vrm\$m.vrm" --out "$root\r2b\renders" --poses stand,armup,sit,walk,wave,crouch,lie --builds normal,chibi --cams full,bust,high --seed $s --jscale 2 --passes line --seed-in-name 2>&1 | Out-Null
            }
            Write-Output "seed $s done"
        }
    } -ArgumentList $b, $root, $r[0], $r[1]
}
Wait-Job $jobs | Receive-Job
Write-Output "RENDER SWEEP DONE: $((Get-ChildItem $root\r2b\renders -Directory).Count) scenes"
