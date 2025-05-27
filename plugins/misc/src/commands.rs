use tauri::Manager;
use tauri_plugin_opener::OpenerExt;

use crate::MiscPluginExt;

#[tauri::command]
#[specta::specta]
pub async fn get_git_hash<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> Result<String, String> {
    Ok(app.get_git_hash())
}

#[tauri::command]
#[specta::specta]
pub async fn get_fingerprint<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<String, String> {
    Ok(app.get_fingerprint())
}

#[tauri::command]
#[specta::specta]
pub async fn opinionated_md_to_html<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    text: String,
) -> Result<String, String> {
    app.opinionated_md_to_html(&text)
}

#[tauri::command]
#[specta::specta]
pub async fn audio_exist<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    session_id: String,
) -> Result<bool, String> {
    let data_dir = app.path().app_data_dir().unwrap();
    let audio_path = data_dir.join(session_id).join("audio.wav");

    let v = std::fs::exists(audio_path).map_err(|e| e.to_string())?;
    Ok(v)
}

#[tauri::command]
#[specta::specta]
pub async fn audio_open<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    session_id: String,
) -> Result<(), String> {
    let data_dir = app.path().app_data_dir().unwrap();
    let audio_path = data_dir.join(session_id).join("audio.wav");

    app.opener()
        .reveal_item_in_dir(&audio_path)
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn delete_session_folder<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    session_id: String,
) -> Result<(), String> {
    let data_dir = app.path().app_data_dir().unwrap();
    let session_dir = data_dir.join(session_id);

    if session_dir.exists() {
        std::fs::remove_dir_all(session_dir).map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn parse_meeting_link<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    text: String,
) -> Option<String> {
    app.parse_meeting_link(&text)
}

#[tauri::command]
pub async fn retranscribe_audio(
    session_id: String,
    use_local: bool,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let data_dir = state.data_dir.clone();
    let audio_path = data_dir.join(&session_id).join("audio.wav");
    let transcription_path = data_dir.join(&session_id).join("transcription.json");
    let diarization_path = data_dir.join(&session_id).join("diarization.json");

    if !audio_path.exists() {
        return Err("Audio file does not exist".to_string());
    }

    let session = state
        .db
        .session_get(&session_id)
        .map_err(|e| e.to_string())?;

    let language = session.language.clone();

    if use_local {
        // Use local models for transcription and diarization
        let whisper_model = data_dir.join("models/whisper/base.en.ggml");
        let diarization_model = data_dir.join("models/pyannote/speaker_diarization.onnx");

        let mut client = stt::local::client::LocalSTTClient::new(
            &whisper_model,
            Some(&diarization_model),
        )
        .map_err(|e| e.to_string())?;

        // Read audio file
        let audio_data = std::fs::read(&audio_path).map_err(|e| e.to_string())?;
        let (audio_samples, sample_rate) = wav::read_wav(&audio_data).map_err(|e| e.to_string())?;

        // Transcribe with diarization
        let segments = client
            .transcribe(&audio_samples, sample_rate)
            .map_err(|e| e.to_string())?;

        // Convert segments to transcription format
        let transcription: Vec<stt::types::Word> = segments
            .into_iter()
            .flat_map(|segment| {
                segment
                    .words()
                    .into_iter()
                    .map(|word| stt::types::Word {
                        word: word.to_string(),
                        start: segment.start,
                        end: segment.end,
                        confidence: segment.confidence,
                        speaker: segment.speaker().cloned(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Save transcription
        std::fs::write(
            &transcription_path,
            serde_json::to_string_pretty(&transcription).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        // Extract and save diarization data
        let diarization: Vec<stt::types::DiarizationSegment> = segments
            .into_iter()
            .filter_map(|segment| {
                segment.speaker().and_then(|speaker| {
                    Some(match speaker {
                        whisper::local::SpeakerIdentity::Unassigned { index } => stt::types::DiarizationSegment {
                            start: segment.start,
                            end: segment.end,
                            speaker: format!("speaker{}", index),
                        },
                        whisper::local::SpeakerIdentity::Assigned { id, label } => stt::types::DiarizationSegment {
                            start: segment.start,
                            end: segment.end,
                            speaker: id,
                        },
                    })
                })
            })
            .collect();

        std::fs::write(
            &diarization_path,
            serde_json::to_string_pretty(&diarization).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    } else {
        // Use Deepgram for transcription and diarization
        let client = stt::deepgram::client::DeepgramClient::new(&language)
            .map_err(|e| e.to_string())?;

        let result = client
            .transcribe_file(&audio_path)
            .await
            .map_err(|e| e.to_string())?;

        std::fs::write(
            &transcription_path,
            serde_json::to_string_pretty(&result.transcription).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        std::fs::write(
            &diarization_path,
            serde_json::to_string_pretty(&result.diarization).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }

    // Update session in database
    state
        .db
        .session_update_transcription(&session_id, &transcription_path, &diarization_path)
        .map_err(|e| e.to_string())?;

    Ok(())
}
