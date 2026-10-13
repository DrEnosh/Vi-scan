# Sample audio (mock STT)

These WAV files are half a second of silence. With `STT_PROVIDER=mock` the server ignores the audio and looks up the transcript by file name in `app/data/audio_fixtures.json`. That lets the voice endpoint be tested end to end without a paid API.

| File | Transcript the mock returns |
|---|---|
| `open_priya_latest_opg.wav` | Open Priya's latest OPG report |
| `open_priya_sharma_latest_opg.wav` | Open Priya Sharma's latest OPG report |
| `summarize_her_report.wav` | Summarize her report |
| `dictation_48_impacted.wav` | Lower right third molar is horizontally impacted |
| `dictation_multi.wav` | Lower right third molar is horizontally impacted with roots close to the mandibular canal. Possible caries on 46. No periapical radiolucency in relation to 36. |

For real speech, set `STT_PROVIDER=groq` with a free `GROQ_API_KEY` and use the mic button in the web UI, or upload your own recording.

```bash
curl -s -X POST localhost:8000/api/assistant/voice -F 'audio=@samples/dictation_multi.wav;type=audio/wav'
```
