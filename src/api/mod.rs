use std::str;

use crate::types;
use crate::types::ForecastResponse;

pub(crate) async fn get_data(
    day: &str,
    cords: types::Coords,
) -> anyhow::Result<types::ForecastResponse> {
    let params = types::Params {
        latitude: cords.lat.to_string(),
        longitude: cords.long.to_string(),
        start_date: String::from(day),
        end_date: String::from(day),
        hourly: vec![
            String::from("temperature_2m"),
            String::from("relative_humidity_2m"),
            String::from("precipitation"),
            String::from("cloud_cover"),
            String::from("wind_speed_10m"),
            String::from("sunshine_duration"),
            String::from("diffuse_radiation"),
            String::from("direct_normal_irradiance"),
            String::from("global_tilted_irradiance"),
            String::from("diffuse_radiation_instant"),
            String::from("direct_normal_irradiance_instant"),
            String::from("global_tilted_irradiance_instant"),
        ]
        .join(","),
        daily: [
            String::from("sunrise"),
            String::from("sunset"),
            String::from("sunshine_duration"),
        ]
        .join(","),
        timezone: String::from("Europe/Berlin"),
    };

    let client = reqwest::Client::new();
    let response: ForecastResponse = client
        .get("https://api.open-meteo.com/v1/forecast")
        .query(&params)
        .send()
        .await?
        .json()
        .await?;
    Ok(response)
}
