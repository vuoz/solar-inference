use crate::model;
use crate::types;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::sync::Arc;

pub(crate) async fn handle_inference(
    State(models): State<Arc<model::AppState>>,
    Json(data): Json<types::InferenceRequest>,
) -> impl IntoResponse {
    let date = data.date;
    StatusCode::OK
}

pub(crate) async fn root() -> &'static str {
    "Hello World"
}
