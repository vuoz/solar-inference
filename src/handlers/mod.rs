use axum::{extract::State, http::StatusCode, response::IntoResponse};

use crate::model;

pub(crate) async fn handle_inference(State(models): State<model::AppState>) -> impl IntoResponse {
    StatusCode::OK
}

pub(crate) async fn root() -> &'static str {
    "Hello World"
}
