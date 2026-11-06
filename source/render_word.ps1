$ErrorActionPreference='Stop'
$root=Join-Path (Split-Path -Parent $PSScriptRoot) 'IEEE_AQUALOC'
$word=New-Object -ComObject Word.Application
$word.Visible=$false
$word.DisplayAlerts=0
$word.AutomationSecurity=3
$audit=@()
try {
 foreach ($stem in @('ieee_manuscript','detailed_derivations')) {
  $path=Join-Path $root "$stem.docx"
  $pdf=Join-Path $root "$stem.pdf"
  $doc=$word.Documents.Open($path,$false,$false)
  try {
   $doc.Fields.Update() | Out-Null
   $doc.Repaginate()
   $pages=$doc.ComputeStatistics(2)
   $doc.Save()
   $doc.ExportAsFixedFormat($pdf,17)
   $audit+=@{document=$stem;pages=$pages;renderer='Microsoft Word';pdf=$pdf}
   Write-Output "$stem rendered pages $pages"
  } finally {$doc.Close(0)}
 }
} finally {$word.Quit()}
$audit | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'evidence\word_render_audit.json')
