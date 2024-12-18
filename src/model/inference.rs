use anyhow::anyhow;
use chrono::{Datelike, NaiveDate};

use crate::types;

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
        let input = data.to_feature_vec()?.unsqueeze(0).to(self.device);
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

                let mut output = tch::Tensor::new();
                for (i, _) in hours.iter().enumerate() {
                    let input_window = self.create_window_input(&hours, i)?;
                    let out_window = self.create_window_prev_out(&past_out, i)?;

                    let model_out = model.forward_ts(&[input_window, out_window])?;

                    past_out.push(model_out.copy());
                    output = tch::Tensor::cat(&[output, model_out], 1);
                }

                Ok(output)
            }
            Season::Autumn => {
                let mut past_out = Vec::with_capacity(24);
                let model = &self.models.autumn;

                let mut output = tch::Tensor::new();
                for (i, _) in hours.iter().enumerate() {
                    let input_window = self.create_window_input(&hours, i)?;
                    let out_window = self.create_window_prev_out(&past_out, i)?;

                    let model_out = model.forward_ts(&[input_window, out_window])?;

                    past_out.push(model_out.copy());
                    output = tch::Tensor::cat(&[output, model_out], 1);
                }

                Ok(output)
            }
            Season::Winter => {
                let mut past_out = Vec::with_capacity(24);
                let model = &self.models.winter;

                let mut output = tch::Tensor::new();
                for (i, _) in hours.iter().enumerate() {
                    let input_window = self.create_window_input(&hours, i)?;
                    let out_window = self.create_window_prev_out(&past_out, i)?;

                    let model_out = model.forward_ts(&[input_window, out_window])?;

                    past_out.push(model_out.copy());
                    output = tch::Tensor::cat(&[output, model_out], 1);
                }

                Ok(output)
            }
            Season::Summer => {
                let mut past_out = Vec::with_capacity(24);
                let model = &self.models.summer;

                let mut output = tch::Tensor::new();
                for (i, _) in hours.iter().enumerate() {
                    let input_window = self.create_window_input(&hours, i)?;
                    let out_window = self.create_window_prev_out(&past_out, i)?;

                    let model_out = model.forward_ts(&[input_window, out_window])?;

                    past_out.push(model_out.copy());
                    output = tch::Tensor::cat(&[output, model_out], 1);
                }
                Ok(output)
            }
        }
    }
    fn create_window_input(&self, hours: &[tch::Tensor], i: usize) -> anyhow::Result<tch::Tensor> {
        let hour_0 = hours.first().ok_or(anyhow!("hour 0 does not exist"))?;
        let mut window = tch::Tensor::new();

        // first we get the prev 4 hours
        for x in (0..4).rev() {
            if (i as i32 - x as i32) < 0 {
                window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(hour_0)], 1);
                continue;
            }
            match hours.get(i - x) {
                // i dont like hte tensor copy here
                Some(v) => window = tch::Tensor::cat(&[window, v.copy()], 1),
                None => window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(hour_0)], 1),
            };
        }
        // then we add the next hour in the future
        match hours.get(i + 1) {
            Some(v) => window = tch::Tensor::cat(&[window, v.copy()], 1),
            None => window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(hour_0)], 1),
        }

        Ok(window.view([window.size()[0], -1]))
    }
    fn create_window_prev_out(
        &self,
        prev_out: &[tch::Tensor],
        i: usize,
    ) -> anyhow::Result<tch::Tensor> {
        let prev_0 = prev_out.get(0).ok_or(anyhow!(" prev 0 is None"))?;

        let mut window = tch::Tensor::new();
        for x in (0..4).rev() {
            if (i as i32 - x as i32) < 0 {
                window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(prev_0)], 1);
                continue;
            }
            match prev_out.get(i - x) {
                // i dont like hte tensor copy here
                Some(v) => window = tch::Tensor::cat(&[window, v.copy()], 1),
                None => window = tch::Tensor::cat(&[window, tch::Tensor::zeros_like(prev_0)], 1),
            }
        }
        Ok(window)
    }
}
