[CmdletBinding()]
param(
    [string]$ComposeProject = "oncall",
    [int]$EtcdTimeoutSeconds = 75,
    [int]$MinioTimeoutSeconds = 90,
    [int]$MilvusTimeoutSeconds = 180,
    [int]$PrometheusTimeoutSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$expectedVolumes = @{
    "milvus-etcd" = @("/etcd", "oncall_milvus_etcd")
    "milvus-minio" = @("/minio_data", "oncall_milvus_minio")
    "milvus-standalone" = @("/var/lib/milvus", "oncall_milvus_data")
}

function Get-ContainerInfo {
    param([Parameter(Mandatory = $true)][string]$Name)

    $raw = docker inspect $Name 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Required container does not exist: $Name"
    }
    return @($raw | ConvertFrom-Json)[0]
}

function Assert-ComposeOwnership {
    param([Parameter(Mandatory = $true)][string]$Name)

    $info = Get-ContainerInfo -Name $Name
    $actualProject = $info.Config.Labels."com.docker.compose.project"
    if ($actualProject -ne $ComposeProject) {
        throw "$Name belongs to Compose project '$actualProject', expected '$ComposeProject'."
    }
    return $info
}

function Assert-NamedVolume {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$VolumeName
    )

    $info = Assert-ComposeOwnership -Name $Name
    $mount = @($info.Mounts | Where-Object { $_.Destination -eq $Destination })[0]
    if ($null -eq $mount) {
        throw "$Name does not mount $Destination."
    }
    if ($mount.Type -ne "volume" -or $mount.Name -ne $VolumeName) {
        throw "$Name uses '$($mount.Type):$($mount.Source)' for $Destination; expected Docker volume '$VolumeName'."
    }
}

function Wait-ContainerHealthy {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $info = Get-ContainerInfo -Name $Name
        $status = [string]$info.State.Status
        $health = if ($null -ne $info.State.Health) {
            [string]$info.State.Health.Status
        } else {
            "missing"
        }
        Write-Host "[INFO] $Name status=$status health=$health"
        if ($status -eq "running" -and $health -eq "healthy") {
            return
        }
        if ($status -in @("dead", "exited")) {
            throw "$Name exited before becoming healthy."
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    throw "$Name did not become healthy within $TimeoutSeconds seconds."
}

function Wait-HttpReady {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                return
            }
        } catch {
            # The service may still be starting.
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    throw "$Uri did not become ready within $TimeoutSeconds seconds."
}

try {
    foreach ($entry in $expectedVolumes.GetEnumerator()) {
        Assert-NamedVolume -Name $entry.Key -Destination $entry.Value[0] -VolumeName $entry.Value[1]
    }
    Assert-ComposeOwnership -Name "milvus-attu" | Out-Null
    Assert-ComposeOwnership -Name "oncall-prometheus" | Out-Null

    Wait-ContainerHealthy -Name "milvus-etcd" -TimeoutSeconds $EtcdTimeoutSeconds
    Wait-ContainerHealthy -Name "milvus-minio" -TimeoutSeconds $MinioTimeoutSeconds
    Wait-ContainerHealthy -Name "milvus-standalone" -TimeoutSeconds $MilvusTimeoutSeconds
    Wait-HttpReady -Uri "http://localhost:9090/-/ready" -TimeoutSeconds $PrometheusTimeoutSeconds

    Write-Host "[OK] Docker infrastructure is healthy and uses project-owned named volumes."
    exit 0
} catch {
    Write-Error $_
    exit 1
}
