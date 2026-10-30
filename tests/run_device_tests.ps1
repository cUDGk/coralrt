# All checks that need the stick, in order. Each step is killed after its timeout (a stuck transfer must not hang
# the run). Run from anywhere:  powershell -ExecutionPolicy Bypass -File tests\run_device_tests.ps1
#   -Data      the test_data folder (models and images, ..\fetch_test_data.py)
#   -Official  Python with pycoral 2.0 (the reference), -Py the Python with coralrt's packages
param(
  [string]$Data = "$PSScriptRoot\..\..\test_data",
  [string]$Official = "$PSScriptRoot\..\..\.venv\Scripts\python.exe",
  [string]$Py = "$PSScriptRoot\..\..\.venv-rt315\Scripts\python.exe"
)
$root = Resolve-Path "$PSScriptRoot\.."
$oracle = "$root\target\oracle"
$log = "$root\target\device_tests.log"
"" | Out-File -Encoding ascii $log

function Step($name, $exe, $arguments, $seconds) {
  "== $name" | Tee-Object -Append -FilePath $log
  $out = "$root\target\step.out"; $err = "$root\target\step.err"
  $p = Start-Process -FilePath $exe -ArgumentList $arguments -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  if (-not $p.WaitForExit($seconds * 1000)) { Stop-Process -Id $p.Id -Force; "TIMEOUT after $seconds s" | Tee-Object -Append -FilePath $log }
  Get-Content $out, $err | Tee-Object -Append -FilePath $log
  "exit $($p.ExitCode)" | Tee-Object -Append -FilePath $log
}

if (-not (Test-Path "$oracle\manifest.txt")) {
  Step "record the official outputs (pycoral 2.0)" $Official @("-I", "$root\tests\oracle\dump_pycoral.py", $Data, $oracle) 300
}
Step "Rust: every output byte vs pycoral, and speed" "$root\target\release\examples\oracle_check.exe" @($Data, $oracle) 300
Step "C: classify the parrot" "$root\target\release\classify_c.exe" @("$Data\mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite", "$oracle\classify_photo.in.bin", "$Data\inat_bird_labels.txt") 120
Step "Python: every output byte vs pycoral, and speed" $Py @("-I", "$root\tests\oracle\check_python.py", $Data, $oracle) 300
Step "pycoral's example scripts, official vs coralrt" $Py @("-I", "$root\tests\oracle\compare_examples.py", (Resolve-Path $Official), (Resolve-Path $Py), $Data, "$root\target\examples") 600
