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
            "temperature_2m".to_string(),
            "relative_humidity_2m".to_string(),
            "precipitation".to_string(),
            "cloud_cover".to_string(),
            "wind_speed_10m".to_string(),
            "sunshine_duration".to_string(),
            "diffuse_radiation".to_string(),
            "direct_normal_irradiance".to_string(),
            "global_tilted_irradiance".to_string(),
            "diffuse_radiation_instant".to_string(),
            "direct_normal_irradiance_instant".to_string(),
            "global_tilted_irradiance_instant".to_string(),
        ]
        .join(","),
        daily: vec![
            "sunrise".to_string(),
            "sunset".to_string(),
            "sunshine_duration".to_string(),
        ]
        .join(","),
        timezone: "Europe/Berlin".to_string(),
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
