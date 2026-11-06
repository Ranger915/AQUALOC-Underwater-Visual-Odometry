$ErrorActionPreference='Stop'
$taskWord=[Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
Write-Output ('Documents: '+$taskWord.Documents.Count)
foreach($taskDoc in $taskWord.Documents){Write-Output $taskDoc.Name}
Write-Output ('Background saving: '+$taskWord.BackgroundSavingStatus)
Write-Output ('Window count: '+$taskWord.Windows.Count)
Write-Output ('Current document saved: '+$taskWord.ActiveDocument.Saved)
Write-Output ('Background printing: '+$taskWord.BackgroundPrintingStatus)
Write-Output ('Active printer: '+$taskWord.ActivePrinter)
Write-Output ('Background pagination: '+$taskWord.Options.Pagination)
