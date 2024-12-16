mod inference;

use tch::{CModule, Device};
#[derive(Debug)]
pub(crate) struct AppState {
    pub device: Device,
    pub models: Models,
}
#[derive(Debug)]
pub(crate) struct Models {
    pub winter: CModule,
    pub summer: CModule,
    pub spring: CModule,
    pub autumn: CModule,
}

pub(crate) fn load_model(path: &str, device: tch::Device) -> anyhow::Result<tch::CModule> {
    Ok(CModule::load_on_device(path, device)?)
}
