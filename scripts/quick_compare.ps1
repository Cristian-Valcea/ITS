# Quick File Comparison Script
# Usage: .\quick_compare.ps1 "path/to/their/file.py" "relative/path/in/project/file.py"

param(
    [Parameter(Mandatory=$true)]
    [string]$TheirFile,
    
    [Parameter(Mandatory=$true)]
    [string]$RelativePath
)

$ProjectRoot = "c:/Projects/IntradayJules"
$YourFile = Join-Path $ProjectRoot $RelativePath

Write-Host "=== Quick File Comparison ===" -ForegroundColor Green
Write-Host "Your file: $YourFile" -ForegroundColor Blue
Write-Host "Their file: $TheirFile" -ForegroundColor Yellow

if (-not (Test-Path $YourFile)) {
    Write-Host "Your file doesn't exist - this is a NEW file!" -ForegroundColor Red
    $choice = Read-Host "Copy their file to your project? (y/n)"
    if ($choice -eq 'y') {
        $targetDir = Split-Path $YourFile -Parent
        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }
        Copy-Item $TheirFile $YourFile
        Write-Host "File copied!" -ForegroundColor Green
    }
    exit
}

if (-not (Test-Path $TheirFile)) {
    Write-Host "Error: Their file doesn't exist!" -ForegroundColor Red
    exit 1
}

# Compare files
$yourContent = Get-Content $YourFile -Raw
$theirContent = Get-Content $TheirFile -Raw

if ($yourContent -eq $theirContent) {
    Write-Host "Files are identical - no changes needed!" -ForegroundColor Green
    exit
}

Write-Host "Files are DIFFERENT!" -ForegroundColor Yellow
Write-Host "Options:" -ForegroundColor White
Write-Host "1. View differences in VS Code (d)" -ForegroundColor White
Write-Host "2. Accept their version (a)" -ForegroundColor White
Write-Host "3. Keep your version (k)" -ForegroundColor White
Write-Host "4. Show text differences (t)" -ForegroundColor White

$choice = Read-Host "Choose option"

switch ($choice) {
    'd' {
        Write-Host "Opening VS Code diff view..." -ForegroundColor Cyan
        Start-Process "code" -ArgumentList "--diff", $YourFile, $TheirFile
    }
    'a' {
        Copy-Item $TheirFile $YourFile
        Write-Host "Accepted their version!" -ForegroundColor Green
    }
    'k' {
        Write-Host "Kept your version!" -ForegroundColor Blue
    }
    't' {
        Write-Host "=== TEXT DIFFERENCES ===" -ForegroundColor Magenta
        Compare-Object (Get-Content $YourFile) (Get-Content $TheirFile) -IncludeEqual | 
            ForEach-Object {
                $indicator = switch ($_.SideIndicator) {
                    '<=' { "[YOUR] " }
                    '=>' { "[THEIR] " }
                    '==' { "[SAME] " }
                }
                $color = switch ($_.SideIndicator) {
                    '<=' { "Blue" }
                    '=>' { "Yellow" }
                    '==' { "Gray" }
                }
                Write-Host "$indicator$($_.InputObject)" -ForegroundColor $color
            }
        Write-Host "========================" -ForegroundColor Magenta
    }
}