# Downloads the ICML 2024 author kit and extracts icml2024.sty into this directory
$url = "https://icml.cc/media/icml-2024/Styles/icml2024.sty"
$dest = "$PSScriptRoot\icml2024.sty"

if (Test-Path $dest) {
    Write-Host "icml2024.sty already exists, skipping download."
} else {
    Write-Host "Downloading icml2024.sty..."
    Invoke-WebRequest -Uri $url -OutFile $dest
    Write-Host "Saved to $dest"
}
