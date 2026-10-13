# Creates real spoken test clips with Windows' built-in text-to-speech (no internet, no key).
# Used by run_check.bat to test the real voice path: speech -> Whisper -> LLM -> tools.
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = -1
$dir = Join-Path $PSScriptRoot "..\samples"
function Say([string]$file, [string]$text) {
    $s.SetOutputToWaveFile((Join-Path $dir $file))
    $s.Speak($text)
    $s.SetOutputToNull()
    Write-Output "wrote samples\$file : $text"
}
Say "speech_open_priya.wav" "Can you pull up the latest O P G for Priya Sharma?"
Say "speech_summarize.wav" "Summarize her report."
Say "speech_dictation.wav" "Lower right third molar is horizontally impacted. Possible caries on tooth four six. No periapical radiolucency in relation to tooth three six."
$s.Dispose()
