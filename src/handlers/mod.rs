use crate::api;
use crate::model;
use crate::types;
use crate::types::AppError;
use anyhow::Result;
use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use std::sync::Arc;

pub(crate) async fn handle_inference(
    State(models): State<Arc<model::AppState>>,
    Json(data): Json<types::InferenceRequest>,
) -> Result<impl IntoResponse, AppError> {
    let date = data.date;
    let weather_data = api::get_data(&date, data.cooridinates).await?;
    let res = models.inference(&date, weather_data)?;
    Ok(StatusCode::OK)
}

pub(crate) async fn root() -> &'static str {
    "Hello World"
}
