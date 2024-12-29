use crate::api;
use crate::model;
use crate::types;
use crate::types::AppError;
use anyhow::Result;
use axum::{extract::State, response::IntoResponse, Json};
use std::sync::Arc;

pub(crate) async fn handle_inference(
    State(models): State<Arc<model::AppState>>,
    Json(data): Json<types::InferenceRequest>,
) -> Result<impl IntoResponse, AppError> {
    let date = data.date;
    let weather_data = api::get_data(&date, data.coordinates).await?;
    let res = models.inference(&date, weather_data)?;
    let len = res.numel();
    let mut des = vec![0.0f32; len];
    res.copy_data(&mut des, len);
    Ok(Json(types::InferenceResponse { data: des }).into_response())
}

pub(crate) async fn root() -> &'static str {
    "Hello World"
}
