use anyhow::anyhow;
use chrono::{Datelike, NaiveDate};

use crate::types;
use tch::Kind;

use super::AppState;
use crate::types::Season;

impl AppState {
    pub(crate) fn inference(
        &self,
        date: &str,
        data: types::ForecastResponse,
    ) -> anyhow::Result<tch::Tensor> {
        println!("device {:?}", self.device);
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
                let mut past_out = Vec::with_capacity(24);

                //let mut output = tch::Tensor::new().to_device(self.device);
                let mut output = tch::Tensor::empty(288, (Kind::Float, self.device));
                for (i, _) in hours.iter().enumerate() {
                    let input_window = self.create_window_input(&hours, i)?.to_device(self.device);
                    let out_window = self
                        .create_window_prev_out(&past_out, i)?
                        .to_device(self.device);

                    let model_out = model
                        .forward_ts(&[input_window, out_window])?
                        .to_device(self.device);

                    past_out.push(model_out.copy());
                    output = tch::Tensor::cat(&[output, model_out], 1).to_device(self.device);
                }

                Ok(output)
            }
            Season::Autumn => {
                let mut past_out = Vec::with_capacity(24);
                let model = &self.models.autumn;

                let mut output = tch::Tensor::empty(288, (Kind::Float, self.device));
                for (i, _) in hours.iter().enumerate() {
                    let input_window = self.create_window_input(&hours, i)?.to_device(self.device);
                    let out_window = self
                        .create_window_prev_out(&past_out, i)?
                        .to_device(self.device);

                    let model_out = model
                        .forward_ts(&[input_window, out_window])?
                        .to_device(self.device);

                    past_out.push(model_out.copy());
                    output = tch::Tensor::cat(&[output, model_out], 1).to_device(self.device);
                }

                Ok(output)
            }
            Season::Winter => {
                let mut past_out = Vec::with_capacity(24);
                let model = &self.models.winter;

                let mut output = tch::Tensor::empty(288, (Kind::Float, self.device));
                for (i, _) in hours.iter().enumerate() {
                    let input_window = self.create_window_input(&hours, i)?.to_device(self.device);
                    let out_window = self
                        .create_window_prev_out(&past_out, i)?
                        .to_device(self.device);

                    let model_out = model
                        .forward_ts(&[input_window, out_window])?
                        .to_device(self.device);

                    past_out.push(model_out.copy());
                    output = tch::Tensor::cat(&[output, model_out], 1).to_device(self.device);
                }

                Ok(output)
            }
            Season::Summer => {
                let mut past_out = Vec::with_capacity(24);
                let model = &self.models.summer;

                let mut output = tch::Tensor::empty(288, (Kind::Float, self.device));
                for (i, _) in hours.iter().enumerate() {
                    let input_window = self.create_window_input(&hours, i)?.to_device(self.device);
                    let out_window = self
                        .create_window_prev_out(&past_out, i)?
                        .to_device(self.device);

                    let model_out = model
                        .forward_ts(&[input_window, out_window])?
                        .to_device(self.device);

                    past_out.push(model_out.copy());
                    output = tch::Tensor::cat(&[output, model_out], 1).to_device(self.device);
                }
                Ok(output)
            }
        }
    }
    fn create_window_input(&self, hours: &[tch::Tensor], i: usize) -> anyhow::Result<tch::Tensor> {
        let hour_0 = hours.first().ok_or(anyhow!("hour 0 does not exist"))?;
        let mut window = tch::Tensor::empty([1, 5, 13], (Kind::Float, self.device));

        // first we get the prev 4 hours
        for x in (0..4).rev() {
            if (i as i32 - x as i32) < 0 {
                window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(hour_0)], 1);
                continue;
            }
            match hours.get(i - x) {
                // i dont like hte tensor copy here
                Some(v) => window = tch::Tensor::cat(&[window, v.copy()], 1).to_device(self.device),
                None => {
                    window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(hour_0)], 1)
                        .to_device(self.device)
                }
            };
        }
        // then we add the next hour in the future
        match hours.get(i + 1) {
            Some(v) => window = tch::Tensor::cat(&[window, v.copy()], 1).to_device(self.device),
            None => {
                window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(hour_0)], 1)
                    .to_device(self.device)
            }
        }

        Ok(window.view([window.size()[0], -1]).to_device(self.device))
    }
    fn create_window_prev_out(
        &self,
        prev_out: &[tch::Tensor],
        i: usize,
    ) -> anyhow::Result<tch::Tensor> {
        let mut window = tch::Tensor::empty([1, 4, 12], (Kind::Float, self.device));
        let prev_0 = match prev_out.first() {
            None => &tch::Tensor::zeros([1, 1, 12], (Kind::Float, self.device)),
            Some(prev_0) => prev_0,
        };

        for x in (0..4).rev() {
            if (i as i32 - x as i32) < 0 {
                window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(prev_0)], 1)
                    .to_device(self.device);
                continue;
            }
            match prev_out.get(i - x) {
                // i dont like the tensor copy here
                Some(v) => window = tch::Tensor::cat(&[window, v.copy()], 1).to_device(self.device),
                None => {
                    window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(prev_0)], 1)
                        .to_device(self.device)
                }
            }
        }
        Ok(window)
    }
}
