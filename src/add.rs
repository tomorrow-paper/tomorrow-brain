use ::prelude::*;

fn add(model: &Vec<u8>, a: i32, b: i32) -> Result<i32> {
    let mut x = Tensor::new(&[1]);
    let mut y = Tensor::new(&[1]);

    x[0] = a;
    y[0] = b;

    let mut graph = Graph::new();
    graph.import_graph_def(model, &ImportGraphDefOptions::new()).map_err(|e| e.to_string())?;

    let mut session = Session::new(&SessionOptions::new(), &graph).map_err(|e| e.to_string())?;

    let mut step = StepWithGraph::new();
    step.add_input(&graph.operation_by_name_required("x").map_err(|e| e.to_string())?, 0, &x);
    step.add_input(&graph.operation_by_name_required("y").map_err(|e| e.to_string())?, 0, &y);

    let output = step.request_output(&graph.operation_by_name_required("z").map_err(|e| e.to_string())?, 0);
    session.run(&mut step).map_err(|e| e.to_string())?;

    let result = step.take_output(output).map_err(|e| e.to_string())?[0];
    
    Ok(result)
}

#[cfg(test)]
mod tests {

    use ::prelude::*;
    use super::*;

    #[test]
    fn add_should_return_the_expected_result() {
        let a = 2;
        let b = 3;
        let expected = 5;

        let model = Model::from_path("models/addition.pb").unwrap();
        let result = add(&model.into(), a, b);

        println!("{:?}", result);

        assert!(result.is_ok());
        assert_eq!(expected, result.unwrap());
    }
}