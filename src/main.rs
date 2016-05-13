mod neural;

use neural::*;

fn main() {

    let mut bnn = BNN::new(2, 2, vec![2], 0.01).ok().unwrap();
    
    bnn.layers[0].weights = vec![vec![0.15, 0.20], vec![0.25, 0.30]];
    bnn.layers[0].bias = 0.35;

    bnn.layers[1].weights = vec![vec![0.40, 0.45], vec![0.50, 0.55]];
    bnn.layers[1].bias = 0.60;
    println!("{:?}", bnn.forward(&vec![0.05f32,0.10f32]));

}
