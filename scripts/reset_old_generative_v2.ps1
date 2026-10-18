$ErrorActionPreference = 'Stop'
$tacRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$manifestPath = Join-Path $tacRoot 'reports\reconstrucao_zero\execucao_v2\deletion_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$prepared = Join-Path $tacRoot 'data\tac_v2\encoded_ready.json'
$newProgress = Get-Content -LiteralPath (Join-Path $tacRoot 'data\tac_v2\progress.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if (!(Test-Path -LiteralPath $prepared) -or $newProgress.generation -ne 'TAC-v2' -or $newProgress.global_step -ne 0) { throw 'Corpus/tokenização/pesos novos não estão prontos.' }
if ($manifest.root -ne $tacRoot) { throw 'Raiz do manifesto difere da TAC.' }
$targets = @()
foreach ($entry in $manifest.files) {
    $target = [IO.Path]::GetFullPath((Join-Path $tacRoot $entry.file))
    if (!$target.StartsWith($tacRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Fora da TAC: $target" }
    $item = Get-Item -LiteralPath $target -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Alvo não é arquivo regular: $target" }
    if ($entry.file -match '(^|/)(books|inbox|dialogue_imports|backups|tac_v2)/' -or $entry.file -match 'reservad|reserved|sealed|selado') { throw "Alvo protegido: $target" }
    if ($item.Length -ne $entry.bytes -or (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Arquivo mudou desde a identificação: $target" }
    $targets += $target
}
# Todos os caminhos absolutos e hashes são conferidos ANTES da primeira exclusão.
$deleted = @()
foreach ($target in $targets) {
    Remove-Item -LiteralPath $target -Force -ErrorAction Stop
    $deleted += $target.Substring($tacRoot.Length + 1).Replace('\','/')
}
$result = [ordered]@{ status='OK'; files_deleted=$deleted.Count; bytes_deleted=$manifest.bytes; backups_created=0; files=$deleted }
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $tacRoot 'reports\reconstrucao_zero\execucao_v2\deletion_result.json') -Encoding UTF8
Write-Output ("Excluídos somente {0} artefatos identificados; nenhum backup criado." -f $deleted.Count)
