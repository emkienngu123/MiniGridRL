param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$reportDir = $PSScriptRoot
$outputDir = Join-Path (Split-Path -Parent $reportDir) 'output\pdf'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$pythonCommand = (Get-Command $Python -ErrorAction Stop).Source

# Use an installed compiler, or the per-user TinyTeX installation.
$tinytexBin = Join-Path $env:APPDATA 'TinyTeX\bin\windows'
if (-not (Get-Command latexmk -ErrorAction SilentlyContinue) -and
    (Test-Path -LiteralPath (Join-Path $tinytexBin 'latexmk.exe'))) {
    $env:PATH = "$tinytexBin;$env:PATH"
}
if (-not (Get-Command latexmk -ErrorAction SilentlyContinue)) {
    throw 'latexmk was not found. Install TeX Live/TinyTeX and restart VS Code.'
}
Push-Location $reportDir
try {
    & $pythonCommand (Join-Path $reportDir 'prepare_assets.py')
    if ($LASTEXITCODE -ne 0) { throw 'Report asset verification failed' }
    # Native stderr can contain ordinary latexmk diagnostics in Windows PowerShell.
    $ErrorActionPreference = 'Continue'
    & latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error "-outdir=$outputDir" technical-report.tex
    $compileExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($compileExit -ne 0) { throw "LaTeX compilation failed ($compileExit)" }
    Write-Output "Report: $(Join-Path $outputDir 'technical-report.pdf')"
} finally {
    Pop-Location
}
