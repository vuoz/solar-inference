use anyhow::anyhow;
use axum::response::IntoResponse;
use chrono::Datelike;
use chrono::NaiveDate;
use reqwest::StatusCode;
use serde::Deserialize;
use serde::Serialize;
use tch::Tensor;

#[derive(Serialize, Deserialize)]
pub(crate) struct InferenceRequest {
    pub date: String,
    pub coordinates: Coords,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Coords {
    pub long: f64,
    pub lat: f64,
}

#[derive(Serialize, Deserialize)]
pub struct Params {
    pub latitude: String,
    pub longitude: String,
    pub start_date: String,
    pub end_date: String,
    pub hourly: String,
    pub daily: String,
    pub timezone: String,
}

#[derive(Deserialize)]
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
impl ForecastResponse {
    pub(crate) fn to_feature_vec(&self) -> anyhow::Result<Tensor> {
        let mut weather_dict: Vec<Vec<f64>> = Vec::with_capacity(24);

        let first_date = &self.daily.time[0];
        let day_of_season = self.calc_day_of_season(first_date)?;

        for i in 0..24 {
            let temp_2m = self
                .hourly
                .temperature_2m
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let precipitation = self
                .hourly
                .precipitation
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let cloud_cover = self
                .hourly
                .cloud_cover
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let sunshine_duration = self
                .hourly
                .sunshine_duration
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let irradiance = self
                .hourly
                .global_tilted_irradiance
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let wind_speed = self
                .hourly
                .wind_speed_10m
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let humidity = self
                .hourly
                .relative_humidity_2m
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let diffuse_radiation = self
                .hourly
                .diffuse_radiation
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let direct_normal_irradiance = self
                .hourly
                .direct_normal_irradiance
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let diffuse_radiation_instant = self
                .hourly
                .diffuse_radiation_instant
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let direct_normal_irradiance_instant = self
                .hourly
                .direct_normal_irradiance_instant
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);
            let global_tilted_instant = self
                .hourly
                .global_tilted_irradiance_instant
                .get(i)
                .and_then(|v| Some(*v))
                .unwrap_or(0.0);

            // Append the feature vector for this hour
            weather_dict.push(vec![
                day_of_season as f64,
                temp_2m,
                precipitation,
                cloud_cover,
                sunshine_duration,
                irradiance,
                wind_speed,
                humidity,
                diffuse_radiation,
                direct_normal_irradiance,
                diffuse_radiation_instant,
                direct_normal_irradiance_instant,
                global_tilted_instant,
            ]);
        }

        // Flatten the 2D vector into a 1D vector
        let flattened: Vec<f64> = weather_dict.iter().flatten().copied().collect();

        // Create a tensor with shape [24, 13]
        Ok(Tensor::from_slice(&flattened)
            .view([24, 13])
            .to_kind(tch::Kind::Float))
    }
    fn calc_day_of_season(&self, inpt: &str) -> anyhow::Result<u32> {
        let format = "%Y-%m-%d";
        let val = match NaiveDate::parse_from_str(inpt, format) {
            Ok(val) => val,
            Err(e) => return Err(anyhow!(e)),
        };
        let day = val.day();
        match val.month() {
            12 => Ok(day),
            1 => Ok(day + 31),
            2 => Ok(day + 31 + 31),
            3 => Ok(day),
            4 => Ok(day + 31),
            5 => Ok(day + 31 + 30),
            6 => Ok(day),
            7 => Ok(day + 30),
            8 => Ok(day + 30 + 31),
            9 => Ok(day),
            10 => Ok(day + 30),
            11 => Ok(day + 31),
            month => Err(anyhow!("invalid month: {}", month)),
        }
    }
}

#[derive(Deserialize)]
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

#[derive(Deserialize)]
pub struct HourlyData {
    pub time: Vec<String>,
    pub temperature_2m: Vec<f64>,
    pub relative_humidity_2m: Vec<f64>,
    pub precipitation: Vec<f64>,
    pub cloud_cover: Vec<f64>,
    pub wind_speed_10m: Vec<f64>,
    pub sunshine_duration: Vec<f64>,
    pub diffuse_radiation: Vec<f64>,
    pub direct_normal_irradiance: Vec<f64>,
    pub global_tilted_irradiance: Vec<f64>,
    pub diffuse_radiation_instant: Vec<f64>,
    pub direct_normal_irradiance_instant: Vec<f64>,
    pub global_tilted_irradiance_instant: Vec<f64>,
}

#[derive(Deserialize)]
pub struct DailyUnits {
    pub time: String,
    pub sunrise: String,
    pub sunset: String,
    pub sunshine_duration: String,
}

#[derive(Deserialize)]
pub struct DailyData {
    pub time: Vec<String>,
    pub sunrise: Vec<String>,
    pub sunset: Vec<String>,
    pub sunshine_duration: Vec<f64>,
}
pub enum Season {
    Winter,
    Summer,
    Spring,
    Autumn,
}
#[derive(Serialize, Deserialize)]
pub struct ErrResponse<'a> {
    message: &'a str,
}
#[derive(Debug)]
pub(crate) struct AppError(pub anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        eprintln!("An error occured: \n--------\n{:?}\n--------", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            match serde_json::to_string(&ErrResponse {
                message: "internal server error",
            }) {
                Ok(v) => v,
                Err(_) => String::from("error"),
            },
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
#[derive(Serialize, Deserialize)]
pub struct InferenceResponse {
    pub data: Vec<f32>,
}
