# Elevated, for linux\device.sh: force-binds the Coral sticks on the bus with usbipd-win (the UsbDk filter that the
# official Edge TPU runtime installs makes a normal bind fail to attach), then runs only these fixed actions, each
# when its file appears in %TEMP%\coralrt_usbipd:
#   bind  force-bind again (after a replug, or when the stick came back under the other id)
#   done  unbind both ids, which gives the stick back to Windows' own driver, and quit (also after 3 hours)
#   Start-Process powershell -Verb RunAs -ArgumentList '-ExecutionPolicy','Bypass','-File',"<this file>"
$dir = Join-Path $env:TEMP 'coralrt_usbipd'
New-Item -ItemType Directory -Force $dir | Out-Null
Start-Transcript -Path (Join-Path $dir 'helper.log')
$u = 'C:\Program Files\usbipd-win\usbipd.exe'
$ids = @('18d1:9302', '1a6e:089a')

function Bind {
  $present = & $u list | Out-String
  foreach ($id in $ids) {
    if ($present -match $id) {
      & $u bind --force --hardware-id $id
      "bind $id exit $LASTEXITCODE"
    }
  }
}

Bind
$deadline = (Get-Date).AddHours(3)
while ((Get-Date) -lt $deadline -and -not (Test-Path (Join-Path $dir 'done'))) {
  $req = Join-Path $dir 'bind'
  if (Test-Path $req) {
    Remove-Item $req
    Bind
  }
  Start-Sleep -Seconds 2
}
foreach ($id in $ids) {
  & $u unbind --hardware-id $id
  "unbind $id exit $LASTEXITCODE"
}
Remove-Item (Join-Path $dir 'done') -ErrorAction SilentlyContinue
& $u list
Stop-Transcript
