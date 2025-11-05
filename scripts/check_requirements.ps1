# Basic check for required packages (PowerShell)
$packages = @("numpy","PIL","opencv-python","matplotlib","torch")
Write-Host "This script only checks if 'pip show' finds the packages (not verifying binaries or CUDA)."
foreach ($p in $packages) {
    $res = pip show $p 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "$p: installed" -ForegroundColor Green } else { Write-Host "$p: NOT found" -ForegroundColor Yellow }
}
