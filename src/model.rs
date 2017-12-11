use tomorrow_core::Result;

use std::fs::File;
use std::io::Read;

pub struct Model {
    bytes: Vec<u8>
}

impl Model {

    pub fn new(bytes: Vec<u8>) -> Self {
        Model {
            bytes: bytes
        }
    }

    pub fn from_path(path: &str) -> Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;

        Ok(Model::new(bytes))
    }
}

impl Into<Vec<u8>> for Model {
    fn into(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {

    use super::Model;

    #[test]
    fn into_should_return_model_as_bytes() {
        let expected = vec![1u8];
        let bytes = expected.clone();

        let model = Model::new(bytes);
        let result: Vec<u8> = model.into();

        assert_eq!(expected, result);
    }

    #[test]
    fn from_path_should_init_model_with_file_bytes() {
        let path = "LICENSE";
        let file_size = 496;

        let model = Model::from_path(path);
        assert!(model.is_ok());

        let result: Vec<u8> = model.unwrap().into();
        assert_eq!(file_size, result.len());
    }
}