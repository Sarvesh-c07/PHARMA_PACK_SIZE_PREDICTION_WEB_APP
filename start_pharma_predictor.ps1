$ErrorActionPreference = "Stop"

$workspace = $PSScriptRoot
$app = Join-Path $workspace "app.py"
$req = Join-Path $workspace "requirements.txt"
$log = Join-Path $workspace "work\app_log.txt"
$url = "http://127.0.0.1:8000"

New-Item -ItemType Directory -Force -Path (Join-Path $workspace "work") | Out-Null

function Show-Error($message) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show($message, "Pharma Predictor could not start", "OK", "Error") | Out-Null
}

# --- 1. Find a Python interpreter on this machine ---------------------------
$python = $null
$pythonArgs = @()
$candidates = @(
    @{ file = "py";      pre = @("-3") },
    @{ file = "python";  pre = @() },
    @{ file = "python3"; pre = @() }
)
foreach ($c in $candidates) {
    $cmd = Get-Command $c.file -ErrorAction SilentlyContinue
    if ($cmd) {
        try {
            & $c.file @($c.pre + @("--version")) *> $null
            if ($LASTEXITCODE -eq 0) { $python = $c.file; $pythonArgs = $c.pre; break }
        } catch { }
    }
}
if (-not $python) {
    Show-Error("No Python was found. Install Python 3.10 or newer from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run this file again.")
    exit 1
}

# --- 2. Make sure the required packages are installed -----------------------
function Test-Deps {
    & $python @($pythonArgs + @("-c", "import pandas, numpy, sklearn, openpyxl")) *> $null
    return ($LASTEXITCODE -eq 0)
}
if (-not (Test-Deps)) {
    Write-Host "Installing required packages (this can take a couple of minutes the first time)..."
    & $python @($pythonArgs + @("-m", "pip", "install", "--upgrade", "-r", "`"$req`"")) 2>&1 | Tee-Object -FilePath $log
    if (-not (Test-Deps)) {
        Show-Error("Could not install the required Python packages (scikit-learn is needed for the ML model). Open a Command Prompt in this folder and run:`n`n    $python -m pip install -r requirements.txt`n`nDetails were written to work\app_log.txt.")
        exit 1
    }
}

# --- 3. Start the app (if it is not already running) ------------------------
function Test-AppReady {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2
        return $r.StatusCode -eq 200
    } catch { return $false }
}

if (-not (Test-AppReady)) {
    $allArgs = $pythonArgs + @("`"$app`"")
    Start-Process -FilePath $python -ArgumentList $allArgs `
        -WorkingDirectory $workspace -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"

    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {   # up to ~30s
        Start-Sleep -Milliseconds 500
        if (Test-AppReady) { $ready = $true; break }
    }

    if (-not $ready) {
        $detail = ""
        if (Test-Path "$log.err") { $detail = "`n`n" + (Get-Content "$log.err" -Raw -ErrorAction SilentlyContinue) }
        Show-Error("The app did not respond on $url.$detail`n`nThe full log is in work\app_log.txt.")
        exit 1
    }
}

Start-Process $url
