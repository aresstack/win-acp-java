<#
.SYNOPSIS
    Releases a new version to Maven Central via GitHub Actions.
.PARAMETER Version
    The version to release, e.g. "0.1.0-beta.1", "0.2.0", "1.0.0"
.EXAMPLE
    .\release.ps1 0.1.0-beta.1
#>
param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidatePattern('^\d+\.\d+\.\d+')]
    [string]$Version
)

$ErrorActionPreference = 'Stop'
$tag = "v$Version"

if (-not (Test-Path 'build.gradle')) {
    Write-Error "build.gradle not found. Run this script from the repository root."
}

$existingTag = git tag -l $tag
if ($existingTag) {
    Write-Error "Tag $tag already exists. Choose a different version."
}

# --- Update version in build.gradle ---
Write-Host "[1/5] Updating build.gradle ..." -ForegroundColor Cyan
$gradleBytes = [System.IO.File]::ReadAllBytes("$PWD\build.gradle")
$gradleText  = [System.Text.Encoding]::UTF8.GetString($gradleBytes)
$gradleNew   = $gradleText -replace "(version\s*=\s*')[^']+(')", "`${1}$Version`${2}"
if ($gradleNew -ne $gradleText) {
    [System.IO.File]::WriteAllBytes("$PWD\build.gradle", [System.Text.Encoding]::UTF8.GetBytes($gradleNew))
    Write-Host "       build.gradle -> $Version" -ForegroundColor Green
} else {
    Write-Host "       build.gradle already at $Version" -ForegroundColor Yellow
}

# --- Update version in README.md ---
Write-Host "[2/5] Updating README.md ..." -ForegroundColor Cyan
if (Test-Path 'README.md') {
    $readmeBytes = [System.IO.File]::ReadAllBytes("$PWD\README.md")
    $readmeText  = [System.Text.Encoding]::UTF8.GetString($readmeBytes)
    $readmeNew   = $readmeText -replace '(<version>)[^<]+(</version>)', "`${1}$Version`${2}"
    $readmeNew   = $readmeNew  -replace "(implementation\s+'com\.aresstack:win-acp-java[^:]*:)[^']+'", "`${1}$Version'"
    if ($readmeNew -ne $readmeText) {
        [System.IO.File]::WriteAllBytes("$PWD\README.md", [System.Text.Encoding]::UTF8.GetBytes($readmeNew))
        Write-Host "       README.md -> $Version" -ForegroundColor Green
    } else {
        Write-Host "       README.md already at $Version" -ForegroundColor Yellow
    }
} else {
    Write-Host "       README.md not found — skipping" -ForegroundColor Yellow
}

# --- Commit ---
Write-Host "[3/5] Committing ..." -ForegroundColor Cyan
git add build.gradle README.md
$diff = git diff --cached --name-only
if ($diff) {
    git commit -m "release $Version"
} else {
    Write-Host "       No changes — skipping commit." -ForegroundColor Yellow
}

# --- Tag ---
Write-Host "[4/5] Creating tag $tag ..." -ForegroundColor Cyan
git tag $tag

# --- Push ---
Write-Host "[5/5] Pushing to origin ..." -ForegroundColor Cyan
git push origin HEAD --tags

Write-Host ""
Write-Host "Done! Tag $tag pushed." -ForegroundColor Green
Write-Host "GitHub Actions workflow will now build and publish to Maven Central."
Write-Host "Monitor: https://github.com/aresstack/win-acp-java/actions" -ForegroundColor Yellow
