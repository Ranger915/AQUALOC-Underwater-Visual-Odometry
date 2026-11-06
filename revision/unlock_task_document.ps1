$ErrorActionPreference='Stop'
$expected=(Join-Path (Split-Path -Parent $PSScriptRoot) 'IEEE_AQUALOC_Revision\ieee_manuscript.docx')
$taskWord=[Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
foreach($taskDoc in @($taskWord.Documents)) {
 if($taskDoc.FullName -eq $expected){$taskDoc.Close(0);Write-Output 'Closed only the saved task manuscript to release its file lock'}
}
