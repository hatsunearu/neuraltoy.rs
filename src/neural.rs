pub struct BNN {
    pub layers: Vec<Layer>,

    pub inputs: usize,
    pub outputs: usize,
    pub hiddens: Vec<usize>,

    pub learningrate: f32
}

pub enum BNNCreationErr {
    IllegalInputSize,
    IllegalOutputSize,
    IllegalHiddenLayer,
    IllegalLearningRate
}

impl BNN {
    pub fn new(num_i: usize, num_o: usize, nums_h: Vec<usize>, learningrate: f32) -> Result<BNN, BNNCreationErr> {
        
        if num_i == 0 {
            return Err(BNNCreationErr::IllegalInputSize);
        }
        else if num_o == 0 {
            return Err(BNNCreationErr::IllegalOutputSize);
        }
        else if nums_h.len() == 0 {
            return Err(BNNCreationErr::IllegalHiddenLayer);
        }
        else if learningrate <= 0f32 {
            return Err(BNNCreationErr::IllegalLearningRate);
        }


        let mut layers = Vec::new();
        let mut last_num = num_i;
        for num_h in &nums_h {
            if *num_h == 0 {
                return Err(BNNCreationErr::IllegalHiddenLayer);
            }
            layers.push(Layer::new(*num_h, last_num));
            last_num = *num_h;
        }
        layers.push(Layer::new(num_o, last_num));

        Ok(BNN {
            layers: layers,
            
            inputs: num_i,
            outputs: num_o,
            hiddens: nums_h,
            
            learningrate: learningrate
        })
    }
    pub fn forward<'a>(&'a mut self, input: &'a Vec<f32>) -> &'a Vec<f32> {
        let mut last_output = input;
        for mut l in &mut self.layers {
            l.forward_prop(last_output);
            last_output = &l.output;
        }
        last_output
    }
}

pub struct Layer {
    pub weights: Vec<Vec<f32>>,
    pub bias: f32,
    pub output: Vec<f32>,
    pub delta: Vec<f32>,
}

impl Layer {
    fn new(num_nodes: usize, prev_num_nodes: usize) -> Layer {
        Layer {
            weights: vec![vec![0f32; prev_num_nodes]; num_nodes],
            bias: 0f32,
            output: vec![0f32; num_nodes],
            delta: vec![0f32; num_nodes]
        }
    }
    fn forward_prop(&mut self, last_output: &Vec<f32>) {
        for (o, w) in self.output.iter_mut().zip(self.weights.iter()) {
            let mut net_input = self.bias;
            // dot prod between input vector and weight for this node
            for e_prod in w.iter().zip(last_output).map(|(x, y)| x*y) {
                net_input += e_prod;
            }
            *o = activation(&net_input);
        }
    }
}

fn activation(x: &f32) -> f32 {
    1f32/(1f32 + (-x).exp())
}

#[test]
fn test_fwdprop() {
    let mut bnn = BNN::new(2, 2, vec![2], 0.01).ok().unwrap();

    bnn.layers[0].weights = vec![vec![0.15, 0.20], vec![0.25, 0.30]];
    bnn.layers[0].bias = 0.35;

    bnn.layers[1].weights = vec![vec![0.40, 0.45], vec![0.50, 0.55]];
    bnn.layers[1].bias = 0.60;

    let input = vec![0.05f32, 0.10f32];
    
    let out = bnn.forward(&input);

    let o1 = 0.75136507;
    let o2 = 0.772928465;
    assert!((o1-out[0]).abs() < o1*1e-6);
    assert!((o2-out[1]).abs() < o2*1e-6);

}
