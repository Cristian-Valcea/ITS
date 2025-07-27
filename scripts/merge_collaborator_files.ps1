# Collaborator File Merge Script
# This script helps you safely merge files from your collaborator

param(
    [Parameter(Mandatory=$true)]
    [string]$DownloadPath,
    
    [Parameter(Mandatory=$false)]
    [string]$ProjectRoot = "c:/Projects/IntradayJules"
)

Write-Host "=== Collaborator File Merge Tool ===" -ForegroundColor Green
Write-Host "Download Path: $DownloadPath" -ForegroundColor Yellow
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Yellow
Write-Host ""

# Create a temporary directory for comparison
$TempDir = Join-Path $ProjectRoot "temp_merge"
if (Test-Path $TempDir) {
    Remove-Item $TempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $TempDir | Out-Null

# Function to compare and merge a single file
function Compare-AndMergeFile {
    param($FilePath, $OriginalFile, $NewFile)
    
    Write-Host "Comparing: $FilePath" -ForegroundColor Cyan
    
    if (-not (Test-Path $OriginalFile)) {
        Write-Host "  -> NEW FILE: $FilePath" -ForegroundColor Green
        $choice = Read-Host "  Add this new file? (y/n)"
        if ($choice -eq 'y') {
            $targetDir = Split-Path (Join-Path $ProjectRoot $FilePath) -Parent
            if (-not (Test-Path $targetDir)) {
                New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            }
            Copy-Item $NewFile (Join-Path $ProjectRoot $FilePath)
            Write-Host "  -> File added!" -ForegroundColor Green
        }
        return
    }
    
    # Compare file contents
    $originalContent = Get-Content $OriginalFile -Raw
    $newContent = Get-Content $NewFile -Raw
    
    if ($originalContent -eq $newContent) {
        Write-Host "  -> No changes detected" -ForegroundColor Gray
        return
    }
    
    Write-Host "  -> CHANGES DETECTED!" -ForegroundColor Yellow
    Write-Host "  Options:" -ForegroundColor White
    Write-Host "    1. View differences (d)" -ForegroundColor White
    Write-Host "    2. Accept their version (a)" -ForegroundColor White
    Write-Host "    3. Keep your version (k)" -ForegroundColor White
    Write-Host "    4. Manual merge (m)" -ForegroundColor White
    Write-Host "    5. Skip for now (s)" -ForegroundColor White
    
    $choice = Read-Host "  Choose option"
    
    switch ($choice) {
        'd' {
            # Show differences using built-in PowerShell comparison
            Write-Host "=== DIFFERENCES ===" -ForegroundColor Magenta
            Compare-Object (Get-Content $OriginalFile) (Get-Content $NewFile) -IncludeEqual | 
                ForEach-Object {
                    $indicator = switch ($_.SideIndicator) {
                        '<=' { "YOUR VERSION: " }
                        '=>' { "THEIR VERSION: " }
                        '==' { "SAME: " }
                    }
                    Write-Host "$indicator$($_.InputObject)"
                }
            Write-Host "===================" -ForegroundColor Magenta
            
            # Ask again after showing differences
            $choice2 = Read-Host "Now choose: (a)ccept theirs, (k)eep yours, (m)anual merge, (s)kip"
            switch ($choice2) {
                'a' { Copy-Item $NewFile $OriginalFile; Write-Host "  -> Accepted their version" -ForegroundColor Green }
                'k' { Write-Host "  -> Kept your version" -ForegroundColor Blue }
                'm' { 
                    Write-Host "  -> Opening files for manual merge..." -ForegroundColor Yellow
                    Write-Host "     Original: $OriginalFile" -ForegroundColor Gray
                    Write-Host "     Their version: $NewFile" -ForegroundColor Gray
                    Start-Process "code" -ArgumentList "--diff", $OriginalFile, $NewFile
                    Read-Host "  Press Enter when you've finished manual merge"
                }
                's' { Write-Host "  -> Skipped" -ForegroundColor Gray }
            }
        }
        'a' { 
            Copy-Item $NewFile $OriginalFile
            Write-Host "  -> Accepted their version" -ForegroundColor Green 
        }
        'k' { Write-Host "  -> Kept your version" -ForegroundColor Blue }
        'm' { 
            Write-Host "  -> Opening files for manual merge..." -ForegroundColor Yellow
            Start-Process "code" -ArgumentList "--diff", $OriginalFile, $NewFile
            Read-Host "  Press Enter when you've finished manual merge"
        }
        's' { Write-Host "  -> Skipped" -ForegroundColor Gray }
    }
}

# Main processing
Write-Host "Scanning for files in: $DownloadPath" -ForegroundColor Cyan

if (-not (Test-Path $DownloadPath)) {
    Write-Host "Error: Download path does not exist!" -ForegroundColor Red
    exit 1
}

# Find all files in download directory
$downloadedFiles = Get-ChildItem -Path $DownloadPath -Recurse -File

foreach ($file in $downloadedFiles) {
    # Calculate relative path
    $relativePath = $file.FullName.Substring($DownloadPath.Length).TrimStart('\', '/')
    $originalFile = Join-Path $ProjectRoot $relativePath
    
    Compare-AndMergeFile $relativePath $originalFile $file.FullName
}

# Cleanup
Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== Merge process completed! ===" -ForegroundColor Green
Write-Host "Don't forget to:" -ForegroundColor Yellow
Write-Host "1. Test your changes" -ForegroundColor Yellow
Write-Host "2. Commit your changes: git add . && git commit -m 'Merged collaborator changes'" -ForegroundColor Yellow
Write-Host "3. Push to GitHub: git push" -ForegroundColor Yellow