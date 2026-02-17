[CmdletBinding()]
param(
  [string]$Root = $PSScriptRoot
)

function Resolve-PdfLatex {
  $cmd = Get-Command pdflatex -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  $fallback = "C:\\Users\\tofik.ali\\texlive\\2026\\bin\\windows\\pdflatex.exe"
  if (Test-Path $fallback) { return $fallback }

  throw "pdflatex not found. Install TeX Live/MiKTeX or add pdflatex to PATH."
}

$pdflatex = Resolve-PdfLatex
Write-Host "Using pdflatex: $pdflatex"

$auxExts = @(
  ".aux", ".log", ".nav", ".snm", ".toc", ".out",
  ".fls", ".fdb_latexmk", ".vrb", ".lof", ".lot"
)

function Move-AuxArtifacts {
  param(
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][string]$LatexDir
  )

  foreach ($ext in $auxExts) {
    Get-ChildItem -Path $OutDir -File -Filter "*$ext" -ErrorAction SilentlyContinue |
      ForEach-Object {
        try { Move-Item -Force -ErrorAction Stop -Path $_.FullName -Destination $LatexDir }
        catch { }
      }
  }

  Get-ChildItem -Path $OutDir -File -Filter "*.synctex.gz" -ErrorAction SilentlyContinue |
    ForEach-Object {
      try { Move-Item -Force -ErrorAction Stop -Path $_.FullName -Destination $LatexDir }
      catch { }
    }
}

$texFiles =
  Get-ChildItem -Path $Root -Recurse -File -Filter "*.tex" |
  Where-Object { (Split-Path $_.DirectoryName -Leaf) -eq "latex" } |
  Sort-Object FullName

if (-not $texFiles) {
  Write-Host "No .tex files found under: $Root"
  exit 0
}

$failed = New-Object System.Collections.Generic.List[string]

foreach ($file in $texFiles) {
  $latexDir = $file.DirectoryName
  $outDir = Split-Path $latexDir -Parent

  # Keep LaTeX sources + all build artifacts together under latex/.
  Move-AuxArtifacts -OutDir $outDir -LatexDir $latexDir

  Write-Host ""
  Write-Host "Compiling: $($file.FullName)"

  & $pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory $latexDir $file.FullName
  if ($LASTEXITCODE -ne 0) { $failed.Add($file.FullName); continue }

  & $pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory $latexDir $file.FullName
  if ($LASTEXITCODE -ne 0) { $failed.Add($file.FullName); continue }

  # Place the final PDF next to the lecture/unit folder (not inside latex/).
  $pdfName = "$($file.BaseName).pdf"
  $pdfInLatex = Join-Path $latexDir $pdfName
  $pdfOut = Join-Path $outDir $pdfName

  if (Test-Path $pdfInLatex) {
    try {
      Move-Item -Force -ErrorAction Stop -Path $pdfInLatex -Destination $pdfOut
    } catch {
      $failed.Add("$($file.FullName) (PDF locked? couldn't update: $pdfOut)")
    }
  } else {
    $failed.Add("$($file.FullName) (no PDF produced: $pdfInLatex)")
  }
}

if ($failed.Count -gt 0) {
  Write-Error ("LaTeX compilation failed for:`n" + ($failed -join "`n"))
  exit 1
}

Write-Host ""
Write-Host "Done."

