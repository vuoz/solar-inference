use serde::Deserialize;
use serde::Serialize;

#[derive(Serialize, Deserialize)]
pub(crate) struct InferenceRequest {
    pub date: String,
}

pub(crate) struct Coords {
    pub long: f64,
    pub lat: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Params {
    pub latitude: String,
    pub longitude: String,
    pub start_date: String,
    pub end_date: String,
    pub hourly: String,
    pub daily: String,
    pub timezone: String,
}

#[derive(Debug, Deserialize)]
pub struct ForecastResponse {
    pub latitude: f64,
    pub longitude: f64,
    pub generationtime_ms: f64,
    pub utc_offset_seconds: i32,
    pub timezone: String,
    pub timezone_abbreviation: String,
    pub elevation: f64,
    pub hourly_units: HourlyUnits,
    pub hourly: HourlyData,
    pub daily_units: DailyUnits,
    pub daily: DailyData,
}

#[derive(Debug, Deserialize)]
pub struct HourlyUnits {
    pub time: String,
    pub temperature_2m: String,
    pub relative_humidity_2m: String,
    pub precipitation: String,
    pub cloud_cover: String,
    pub wind_speed_10m: String,
    pub sunshine_duration: String,
    pub diffuse_radiation: String,
    pub direct_normal_irradiance: String,
    pub global_tilted_irradiance: String,
    pub diffuse_radiation_instant: String,
    pub direct_normal_irradiance_instant: String,
    pub global_tilted_irradiance_instant: String,
}

#[derive(Debug, Deserialize)]
pub struct HourlyData {
    pub time: Vec<String>,
    pub temperature_2m: Vec<f64>,
    pub relative_humidity_2m: Vec<u32>,
    pub precipitation: Vec<f64>,
    pub cloud_cover: Vec<u32>,
    pub wind_speed_10m: Vec<f64>,
    pub sunshine_duration: Vec<f64>,
    pub diffuse_radiation: Vec<f64>,
    pub direct_normal_irradiance: Vec<f64>,
    pub global_tilted_irradiance: Vec<f64>,
    pub diffuse_radiation_instant: Vec<f64>,
    pub direct_normal_irradiance_instant: Vec<f64>,
    pub global_tilted_irradiance_instant: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct DailyUnits {
    pub time: String,
    pub sunrise: String,
    pub sunset: String,
    pub sunshine_duration: String,
}

#[derive(Debug, Deserialize)]
pub struct DailyData {
    pub time: Vec<String>,
    pub sunrise: Vec<String>,
    pub sunset: Vec<String>,
    pub sunshine_duration: Vec<f64>,
}
