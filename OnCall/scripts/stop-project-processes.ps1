[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [int[]]$Ports = @(9900, 9105, 8003, 8004),

    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'
$normalizedRoot = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd(
    [char[]]@('\', '/')
)
$projectPrefix = $normalizedRoot + [IO.Path]::DirectorySeparatorChar
$seenProcessIds = @{}

function Test-ProjectProcess {
    param([string]$CommandLine)

    if ([string]::IsNullOrWhiteSpace($CommandLine)) {
        return $false
    }

    return $CommandLine.IndexOf(
        $projectPrefix,
        [StringComparison]::OrdinalIgnoreCase
    ) -ge 0
}

foreach ($servicePort in $Ports) {
    $listeners = @(
        Get-NetTCPConnection `
            -LocalPort $servicePort `
            -State Listen `
            -ErrorAction SilentlyContinue
    )

    foreach ($listener in $listeners) {
        $processId = [int]$listener.OwningProcess
        if ($seenProcessIds.ContainsKey($processId)) {
            continue
        }
        $seenProcessIds[$processId] = $true

        $process = Get-CimInstance Win32_Process `
            -Filter "ProcessId = $processId" `
            -ErrorAction SilentlyContinue
        $commandLine = [string]$process.CommandLine

        if (-not (Test-ProjectProcess -CommandLine $commandLine)) {
            Write-Warning (
                "Port {0} is owned by PID {1}, but the process does not belong " +
                "to this project. It was not stopped." -f $servicePort, $processId
            )
            continue
        }

        if ($WhatIf) {
            Write-Host (
                "[WHATIF] Would stop project process PID {0} on port {1}." -f
                $processId, $servicePort
            )
            continue
        }

        Write-Host (
            "[INFO] Stopping orphaned project process PID {0} on port {1}..." -f
            $processId, $servicePort
        )
        Stop-Process -Id $processId -Force -ErrorAction Stop
    }
}

if ($WhatIf) {
    exit 0
}

Start-Sleep -Milliseconds 500
$remainingPorts = @()
foreach ($servicePort in $Ports) {
    $listener = Get-NetTCPConnection `
        -LocalPort $servicePort `
        -State Listen `
        -ErrorAction SilentlyContinue
    if ($listener) {
        $remainingPorts += $servicePort
    }
}

if ($remainingPorts.Count -gt 0) {
    Write-Warning (
        "The following service ports are still occupied: {0}" -f
        ($remainingPorts -join ', ')
    )
    exit 1
}

exit 0
