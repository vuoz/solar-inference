mod api;
mod handlers;
mod model;
mod types;
use axum::{routing::get, Router};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let device = tch::Device::Cpu;

    let models = model::Models {
        winter: model::load_model("./models/winter.pt", device)?,
        summer: model::load_model("./models/summer.pt", device)?,
        spring: model::load_model("./models/spring.pt", device)?,
        autumn: model::load_model("./models/autumn.pt", device)?,
    };
    let state = Arc::new(model::AppState { device, models });
    let app = Router::new()
        .route("/inference", get(handlers::handle_inference))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:4444").await?;
    println!("Listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}
