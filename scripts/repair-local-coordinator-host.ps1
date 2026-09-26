param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$hostsPath = Join-Path $env:SystemRoot 'System32/drivers/etc/hosts'
$original = [IO.File]::ReadAllText($hostsPath)
$pattern = '(?m)^[\t ]*47\.201\.207\.37[\t ]+api\.joinhavn\.io[\t ]*(?:#[^\r\n]*)?\r?$'
$matchesFound = [regex]::Matches($original, $pattern).Count
Write-Output "Stale coordinator entries: $matchesFound"
if ($CheckOnly -or $matchesFound -eq 0) { exit 0 }
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in PowerShell as Administrator. No file was changed.'
}
$replacement = [regex]::Replace($original, $pattern, '# Removed stale HavnAI coordinator override; use public DNS.')
$backup = Join-Path $env:TEMP ('havnai-hosts-backup-' + [guid]::NewGuid().ToString('N') + '.txt')
[IO.File]::Copy($hostsPath, $backup, $false)
if ([IO.File]::ReadAllText($hostsPath) -cne $original) { throw 'Hosts file changed during inspection; retry after reviewing it.' }
[IO.File]::WriteAllText($hostsPath, $replacement, (New-Object Text.UTF8Encoding($false)))
Write-Output "Backup: $backup"
ipconfig /flushdns
if ($LASTEXITCODE -ne 0) { throw 'Hosts entry was removed, but DNS cache flush failed.' }
Write-Output 'Removed only the stale 47.201.207.37 api.joinhavn.io entry.'
