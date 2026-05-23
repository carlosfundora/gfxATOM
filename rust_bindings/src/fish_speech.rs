use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use regex::Regex;
use once_cell::sync::Lazy;

static LEGACY_SPEAKER_TAG_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"<speaker:(\d+)>").unwrap());
static CANONICAL_SPEAKER_TAG_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"<\|speaker:\d+\|>").unwrap());
static CONTROL_TOKEN_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"<\|[^>]+\|>").unwrap());

#[pyfunction]
#[pyo3(signature = (text, *, add_default_speaker = false))]
pub fn normalize_fish_speech_text(text: &str, add_default_speaker: bool) -> PyResult<String> {
    let normalized = LEGACY_SPEAKER_TAG_PATTERN.replace_all(text, "<|speaker:$1|>");

    let mut disallowed_tokens = Vec::new();
    for token in CONTROL_TOKEN_PATTERN.find_iter(&normalized) {
        let token_str = token.as_str();
        if !CANONICAL_SPEAKER_TAG_PATTERN.is_match(token_str) {
            disallowed_tokens.push(token_str.to_string());
        }
    }

    if !disallowed_tokens.is_empty() {
        disallowed_tokens.sort();
        disallowed_tokens.dedup();
        let disallowed_list = disallowed_tokens.join(", ");
        return Err(PyValueError::new_err(format!("Fish Speech input contains unsupported control token(s): {}", disallowed_list)));
    }

    if add_default_speaker && !CANONICAL_SPEAKER_TAG_PATTERN.is_match(&normalized) {
        return Ok(format!("<|speaker:0|>{}", normalized));
    }

    Ok(normalized.into_owned())
}
