use crate::model;

use super::AppState;

impl AppState {
    pub(crate) fn inference(self, inpt: tch::Tensor) -> anyhow::Result<tch::Tensor> {
        todo!("missing impl")
    }
}
