# All checks that need the stick, in order. Each step is killed after its timeout (a stuck transfer must not hang
# the run). Run from anywhere:  powershell -ExecutionPolicy Bypass -File tests\run_device_tests.ps1
#   -Data      the test_data folder (models and images, ..\fetch_test_data.py)
#   -Official  Python with pycoral 2.0 (the reference), -Py the Python with coralrt's packages
#   -Tf        Python with TensorFlow, to write the model with a SignatureDef (oracle\make_signature_model.py)
param(
  [string]$Data = "$PSScriptRoot\..\..\test_data",
  [string]$Official = "$PSScriptRoot\..\..\.venv\Scripts\python.exe",
  [string]$Py = "$PSScriptRoot\..\..\.venv-rt315\Scripts\python.exe",
  [string]$Tf = "$PSScriptRoot\..\..\.venv-tf\Scripts\python.exe"
)
$root = Resolve-Path "$PSScriptRoot\.."
$oracle = "$root\target\oracle"
$log = "$root\target\device_tests.log"
"" | Out-File -Encoding utf8 $log

# Tee-Object writes UTF-16 in Windows PowerShell; the log stays UTF-8
function Log($lines) { $lines | Out-File -Append -Encoding utf8 $log; $lines }

function Step($name, $exe, $arguments, $seconds) {
  Log "== $name"
  $out = "$root\target\step.out"; $err = "$root\target\step.err"
  $p = Start-Process -FilePath $exe -ArgumentList $arguments -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  $null = $p.Handle  # without the handle cached now, ExitCode stays empty after the process ends
  if (-not $p.WaitForExit($seconds * 1000)) { Stop-Process -Id $p.Id -Force; Log "TIMEOUT after $seconds s" }
  Log (Get-Content $out, $err)
  Log "exit $($p.ExitCode)"
}

if (-not (Test-Path "$oracle\manifest.txt")) {
  Step "record the official outputs (pycoral 2.0)" $Official @("-I", "$root\tests\oracle\dump_pycoral.py", $Data, $oracle) 300
}
Step "Rust: every output byte vs pycoral, and speed" "$root\target\release\examples\oracle_check.exe" @($Data, $oracle) 300
Step "C: classify the parrot" "$root\target\release\classify_c.exe" @("$Data\mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite", "$oracle\classify_photo.in.bin", "$Data\inat_bird_labels.txt") 120
Step "Python: every output byte vs pycoral, and speed" $Py @("-I", "$root\tests\oracle\check_python.py", $Data, $oracle) 300
Step "pycoral's example scripts, official vs coralrt" $Py @("-I", "$root\tests\oracle\compare_examples.py", (Resolve-Path $Official), (Resolve-Path $Py), $Data, "$root\target\examples") 600

$sigModel = "$root\target\signature_model.tflite"
if (-not (Test-Path $sigModel)) {
  Step "write a model with a SignatureDef" $Tf @("-I", "$root\tests\oracle\make_signature_model.py", $sigModel) 300
}
$models = @($sigModel) + (Get-ChildItem "$Data\*_edgetpu.tflite" | ForEach-Object { $_.FullName })
$sig = @{}
foreach ($who in @(@("official", $Official), @("coralrt", $Py))) {
  Step "signature lists, $($who[0])" $who[1] (@("-I", "$root\tests\oracle\signatures.py") + $models) 300
  $sig[$who[0]] = (Get-Content "$root\target\step.out") -join "`n"
}
Log "signature lists: $(if ($sig['official'] -ceq $sig['coralrt']) { 'same' } else { 'DIFFERENT' })"
Step "speed, official and coralrt taking turns" $Py @("-I", "$root\tests\oracle\bench.py", (Resolve-Path $Official), (Resolve-Path $Py), $Data, "6", "200") 900
