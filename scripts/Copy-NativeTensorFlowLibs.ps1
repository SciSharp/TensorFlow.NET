$targetDirectory = [IO.Path]::Combine($PSScriptRoot, "..", "src", "runtime.win-x64.SciSharp.TensorFlow-Gpu.Redist")

$fileName = "libtensorflow-gpu-windows-x86_64-1.14.0.zip"
$zipfile = [IO.Path]::Combine($PSScriptRoot, "..", "packages", $fileName)
if (-not (Test-Path $zipfile -PathType Leaf)) {
    # Create the directory just in case it's actually needed...
    $path = [IO.Path]::Combine($PSScriptRoot, "..", "packages")
    New-Item -Path $path -Force -ItemType Directory
    Write-Host "Downloading libtensorflow gpu for Windows..."
    $wc = New-Object System.Net.WebClient
    $wc.DownloadFile("https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-1.14.0.zip", $zipfile)
}

$libraryName = "tensoflow.dll"
$libraryLocation = "lib\tensorflow.dll"
$windowsTensorFlow = Join-Path $targetDirectory $libraryName

if (-not (Test-Path $windowsTensorFlow))
{
    & 7z e $zipfile $libraryLocation -o"$targetDirectory"
}