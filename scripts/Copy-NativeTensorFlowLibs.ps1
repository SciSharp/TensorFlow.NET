<#
.SYNOPSIS
    Copy the native TensorFlow library to enable the packing a nuget to make
    them available to TensorFlow.NET

.DESCRIPTION
    The TensorFlow libraries are copied for Windows and Linux and it becomes
    possible to bundle a meta-package containing them.

.PARAMETER CpuLibraries
    Switch indicating if the script should download the CPU or GPU version of the
    TensorFlow libraries.
    By default the GPU version of the libraries is downloaded.

#>
param(
    [switch] $CpuLibraries = $false
)

function Expand-TarGzFiles {
    <#
    .SYNOPSIS
    Expands the given list of files from the given archive into the given
    target directory.

    .PARAMETER Archive
    Path to the archive that should be considered.

    .PARAMETER Files
    Files that should be extracted from the archive.

    .PARAMETER TargetDirectory
    Directory into which the files should be expanded.

    #>
    param
    (
        [Parameter(Mandatory = $true, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)] [string] $Archive,
        [Parameter(Mandatory = $true, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)] [string []] $Files,
        [Parameter(Mandatory = $true, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)] [string] $TargetDirectory
    )

    & 7z e $Archive -o"$TargetDirectory"
    $TarArchive = Join-Path $TargetDirectory "libtensorflow.tar"

    & 7z e  $TarArchive $Files -o"$TargetDirectory"
    Remove-Item $TarArchive
}

function Expand-ZipFiles {
    <#
    .SYNOPSIS
    Expands the given list of files from the given archive into the given target directory.

    .PARAMETER Archive
    Path to the archive that should be considered.

    .PARAMETER Files
    Files that should be extracted from the archive.

    .PARAMETER TargetDirectory
    Directory into which the files should be expanded.
    #>
    param(
        [Parameter(Mandatory = $true, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)] [string] $Archive,
        [Parameter(Mandatory = $true, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)] [string []] $Files,
        [Parameter(Mandatory = $true, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)] [string] $TargetDirectory
    )

    & 7z e $Archive $Files -o"$TargetDirectory"
}

function Split-ArchiveFromUrl {
    <#
    .SYNOPSIS
    Extracts the archive name out of the given Url.

    .PARAMETER ArchiveUrl
    Url of the archive that will be downloaded.

    #>
    param(
        [Parameter(Mandatory = $true, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)] [string] $ArchiveUrl
    )
    
    $uriParts = $ArchiveUrl.split("/")
    $ArchivePath = $uriParts[$uriParts.Count - 1]

    return $ArchivePath
}

function Copy-Archive {
    <#
    .SYNOPSIS
    This function copies the given binary file to the given target location.

    .PARAMETER ArchiveUrl
    Url where the archive should be downloaded from.

    .PARAMETER TargetDirectory
    Target directory where the archive should be downloaded.
#>
    param (
        [Parameter(Mandatory = $true, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
        [string] $ArchiveUrl,
        [Parameter(Mandatory = $true, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
        [string] $TargetDirectory
    )

    $ArchiveName = Split-ArchiveFromUrl $ArchiveUrl

    $TargetPath = [IO.Path]::Combine($PSScriptRoot, "..", "packages", $ArchiveName)

    if (Test-Path $TargetPath -PathType Leaf) {
        Write-Error "$TargetPath already exists, please remove to download againg."
        return $TargetPath
    }

    if (-not (Test-Path $TargetDirectory -PathType Container)) {
        Write-Host "Creating missing $TargetDirectory"
        New-Item -Path $TargetDirectory -ItemType Directory
    }
    Write-Host "Downloading $ArchiveUrl, this might take a while..."
    $wc = New-Object System.Net.WebClient
    $wc.DownloadFile($ArchiveUrl, $TargetPath)

    return $TargetPath
}

$LinuxGpuArchive = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.14.0.tar.gz"
$LinuxFiles = @(".\libtensorflow.tar", ".\lib\libtensorflow.so", ".\lib\libtensorflow.so.1", ".\lib\libtensorflow.so.1.14.0", `
        ".\lib\libtensorflow_framework.so", ".\lib\libtensorflow_framework.so.1", ".\lib\libtensorflow_framework.so.1.14.0")
$WindowsGpuArchive = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-1.14.0.zip"
$WindowsFiles = @("lib\tensorflow.dll")
$PackagesDirectory = [IO.Path]::Combine($PSScriptRoot, "..", "packages")


if (-not $CpuLibraries) {
    $WindowsArchive = $WindowsGpuArchive
    $LinuxArchive = $LinuxGpuArchive
}

$Archive = Copy-Archive -ArchiveUrl $WindowsArchive -TargetDirectory $PackagesDirectory
$TargetDirectory = [IO.Path]::Combine($PSScriptRoot, "..", "src", "runtime.win-x64.SciSharp.TensorFlow-Gpu.Redist")
Expand-ZipFiles $Archive $WindowsFiles $TargetDirectory

$Archive = Copy-Archive -ArchiveUrl $LinuxArchive -TargetDirectory $PackagesDirectory
$TargetDirectory = [IO.Path]::Combine($PSScriptRoot, "..", "src", "runtime.linux-x64.SciSharp.Tensorflow-Gpu.Redist")
Expand-TarGzFiles $Archive $LinuxFiles $TargetDirectory
