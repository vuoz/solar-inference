use anyhow::anyhow;
use chrono::{Datelike, NaiveDate};

use crate::types::{self};
use tch::Kind;

use super::AppState;
use crate::types::Season;

impl AppState {
    pub(crate) fn inference(
        &self,
        date: &str,
        data: types::ForecastResponse,
    ) -> anyhow::Result<tch::Tensor> {
        let format = "%Y-%m-%d";
        let val = match NaiveDate::parse_from_str(date, format) {
            Ok(val) => val,
            Err(e) => return Err(anyhow!(e)),
        };

        let season = match val.month() {
            12 | 1 | 2 => Season::Winter,
            3..=5 => Season::Spring,
            6..=8 => Season::Summer,
            9..=11 => Season::Autumn,
            _ => return Err(anyhow!("Invalid Month")),
        };
        let input = data.to_feature_vec()?.unsqueeze(0).to_device(self.device);
        let hours = input.split(1, 1);
        self.forward(season, hours)
    }
    fn forward(
        &self,
        season: types::Season,
        hours: Vec<tch::Tensor>,
    ) -> anyhow::Result<tch::Tensor> {
        match season {
            Season::Spring => {
                let model = &self.models.spring;
                forward_inner(model, hours, self.device)
            }
            Season::Autumn => {
                let model = &self.models.autumn;
                forward_inner(model, hours, self.device)
            }
            Season::Winter => {
                let model = &self.models.winter;
                forward_inner(model, hours, self.device)
            }
            Season::Summer => {
                let model = &self.models.summer;
                forward_inner(model, hours, self.device)
            }
        }
    }
}
fn create_window_input(
    device: tch::Device,
    hours: &[tch::Tensor],
    i: usize,
) -> anyhow::Result<tch::Tensor> {
    let hour_0 = hours.first().ok_or(anyhow!("hour 0 does not exist"))?;
    let mut window = Vec::with_capacity(6);

    // first we get the prev 4 hours
    for x in (0..5).rev() {
        if (i as i32 - x as i32) < 0 {
            window.push(tch::Tensor::zeros_like(hour_0).to(device));
            continue;
        }
        match hours.get(i - x) {
            // i dont like hte tensor copy here need to find another solution in the future
            Some(v) => window.push(v.copy()),
            None => {
                window.push(tch::Tensor::zeros_like(hour_0).to_device(device));
            }
        };
    }
    // then we add the next hour in the future
    match hours.get(i + 1) {
        Some(v) => window.push(v.copy()),
        None => {
            window.push(tch::Tensor::zeros_like(hour_0).to_device(device));
        }
    }
    let tensor = tch::Tensor::cat(&window, 2);

    Ok(tensor.view([tensor.size()[0], -1]).to_device(device))
}
fn create_window_prev_out(
    device: tch::Device,
    prev_out: &[tch::Tensor],
    i: usize,
) -> anyhow::Result<tch::Tensor> {
    let mut window = Vec::with_capacity(5);

    for x in (0..4).rev() {
        if (i as i32 - x as i32) < 0 {
            window.push(tch::Tensor::zeros([1, 12], (Kind::Float, device)));
            continue;
        }
        match prev_out.get(i - x) {
            // i dont like the tensor copy here, will have to find another solution in the
            // future
            Some(v) => window.push(v.copy()),
            None => {
                window.push(tch::Tensor::zeros([1, 12], (Kind::Float, device)));
            }
        }
    }
    let tensor = tch::Tensor::cat(&window, 1).to(device);
    Ok(tensor.to_device(device))
}

fn forward_inner(
    model: &tch::CModule,
    hours: Vec<tch::Tensor>,
    device: tch::Device,
) -> anyhow::Result<tch::Tensor> {
    let mut past_out = Vec::with_capacity(24);

    for (i, _) in hours.iter().enumerate() {
        let input_window = create_window_input(device, &hours, i)?.to_device(device);
        let out_window = create_window_prev_out(device, &past_out, i)?.to_device(device);

        let model_out = model
            .forward_ts(&[input_window, out_window])?
            .to_device(device);

        past_out.push(model_out.copy());
    }
    let combined_tensor = tch::Tensor::concat(&past_out, 0);

    Ok(combined_tensor)
}
