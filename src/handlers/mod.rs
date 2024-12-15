use crate::types;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::model;

pub(crate) async fn handle_inference(
    State(models): State<model::AppState>,
    Json(data): Json<types::InferenceRequest>,
) -> impl IntoResponse {
    StatusCode::OK
}

pub(crate) async fn root() -> &'static str {
    "Hello World"
}
