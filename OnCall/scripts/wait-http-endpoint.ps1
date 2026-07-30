[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Uri,
    [string]$ServiceName = "HTTP service",
    [int]$TimeoutSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    try {
        $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
            Write-Host "[OK] $ServiceName is healthy: $Uri"
            exit 0
        }
    } catch {
        # The service may still be starting.
    }
    Start-Sleep -Seconds 2
} while ((Get-Date) -lt $deadline)

Write-Error "$ServiceName did not become healthy within $TimeoutSeconds seconds: $Uri"
exit 1
