$ErrorActionPreference='Stop'
try {
 $taskWord=[Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
 if($taskWord.Documents.Count -eq 1 -and $taskWord.Documents.Item(1).Name -eq 'ieee_manuscript.docx') {
  $taskWord.Documents.Item(1).Close(0)
  $taskWord.Quit()
  Write-Output 'Closed only the task renderer document'
 } else {throw 'Renderer documents differ from expected task document'}
} catch {Write-Output $_;throw}
