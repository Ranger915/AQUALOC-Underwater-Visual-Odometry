$ErrorActionPreference='Stop'
$root=Join-Path (Split-Path -Parent $PSScriptRoot) 'IEEE_AQUALOC_Revision'
$word=New-Object -ComObject Word.Application
$word.Visible=$false
$word.DisplayAlerts=0
$word.AutomationSecurity=3
Write-Output 'Created Word renderer'
$audit=@()
try {
 foreach ($stem in @('ieee_manuscript','detailed_derivations')) {
  $path=Join-Path $root "$stem.docx"
  $pdf=Join-Path $root "$stem.pdf"
  $doc=$word.Documents.Open($path,$false,$false)
  Write-Output "Opened $stem"
  try {
   $doc.Fields.Update() | Out-Null
   Write-Output "Updated fields $stem"
   $doc.Repaginate()
   Write-Output "Repaginated $stem"
   $pages=$doc.ComputeStatistics(2)
   Write-Output "Counted pages $stem"
   $doc.Save()
   Write-Output "Saved $stem"
   $doc.SaveAs2($pdf,17)
   $audit+=@{document=$stem;pages=$pages;renderer='Microsoft Word';pdf=$pdf}
   Write-Output "$stem rendered pages $pages"
  } finally {$doc.Close(0)}
 }
} finally {$word.Quit()}
$audit | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $root 'word_render_audit.json')
