# Show Differences Script
param(
    [Parameter(Mandatory=$true)]
    [string]$File1,
    
    [Parameter(Mandatory=$true)]
    [string]$File2,
    
    [Parameter(Mandatory=$false)]
    [string]$Label1 = "YOUR VERSION",
    
    [Parameter(Mandatory=$false)]
    [string]$Label2 = "THEIR VERSION"
)

Write-Host "=== FILE COMPARISON ===" -ForegroundColor Green
Write-Host "${Label1}: $File1" -ForegroundColor Blue
Write-Host "${Label2}: $File2" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path $File1)) {
    Write-Host "ERROR: $Label1 file not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $File2)) {
    Write-Host "ERROR: $Label2 file not found!" -ForegroundColor Red
    exit 1
}

$content1 = Get-Content $File1
$content2 = Get-Content $File2

$differences = Compare-Object $content1 $content2 -IncludeEqual

if (-not $differences) {
    Write-Host "FILES ARE IDENTICAL!" -ForegroundColor Green
    exit 0
}

Write-Host "DIFFERENCES FOUND:" -ForegroundColor Yellow
Write-Host "=================" -ForegroundColor Yellow

$lineNum = 1
foreach ($diff in $differences) {
    $indicator = switch ($diff.SideIndicator) {
        '<=' { "[$Label1]" }
        '=>' { "[$Label2]" }
        '==' { "[SAME]" }
    }
    
    $color = switch ($diff.SideIndicator) {
        '<=' { "Blue" }
        '=>' { "Yellow" }
        '==' { "Gray" }
    }
    
    if ($diff.SideIndicator -ne '==') {
        Write-Host "$indicator $($diff.InputObject)" -ForegroundColor $color
    }
}

Write-Host ""
Write-Host "What would you like to do?" -ForegroundColor White
Write-Host "1. Open in VS Code diff view (d)" -ForegroundColor White
Write-Host "2. Accept their version (a)" -ForegroundColor White
Write-Host "3. Keep your version (k)" -ForegroundColor White
Write-Host "4. Exit without changes (e)" -ForegroundColor White

$choice = Read-Host "Choose option"

switch ($choice) {
    'd' {
        Write-Host "Opening VS Code diff..." -ForegroundColor Cyan
        Start-Process "code" -ArgumentList "--diff", $File1, $File2
    }
    'a' {
        Copy-Item $File2 $File1 -Force
        Write-Host "Accepted their version!" -ForegroundColor Green
    }
    'k' {
        Write-Host "Kept your version!" -ForegroundColor Blue
    }
    'e' {
        Write-Host "No changes made." -ForegroundColor Gray
    }
    default {
        Write-Host "Invalid choice. No changes made." -ForegroundColor Red
    }
}