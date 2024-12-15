mod handlers;
mod model;
use axum::{http::header::DNT, routing::get, Router};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = tch::Device::Cpu;

    let models = model::Models {
        winter: model::load_model("./models/model_winter.pt", device)?,
        summer: model::load_model("./models/model_summer.pt", device)?,
        spring: model::load_model("./models/model_spring.pt", device)?,
        autumn: model::load_model("./models/model_autumn.pt", device)?,
    };
    let state = Arc::new(model::AppState { device, models });
    let app = Router::new()
        .route("/", get(handlers::root))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("localhost:4000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
