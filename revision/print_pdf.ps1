$ErrorActionPreference='Stop'
$root=Join-Path (Split-Path -Parent $PSScriptRoot) 'IEEE_AQUALOC_Revision'
$taskWord=[Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
if($taskWord.Documents.Count -ne 1 -or $taskWord.Documents.Item(1).Name -ne 'ieee_manuscript.docx'){throw 'Unexpected renderer document'}
$doc=$taskWord.Documents.Item(1)
$pdf=Join-Path $root 'ieee_manuscript_print.pdf'
Write-Output 'Starting native print-to-PDF'
$missing=[Type]::Missing
$doc.PrintOut($false,$false,0,$pdf,$missing,$missing,0,1,$missing,0,$true)
Write-Output 'Completed native print-to-PDF'
