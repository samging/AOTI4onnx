use aoti_ffi::AotiModel;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: infer <model.pt2>");

    let shape = [1i64, 3, 8, 8];
    let input: Vec<f32> = (0..shape.iter().product::<i64>() as usize)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let model = AotiModel::load(&path, 1).expect("load failed");
    let outputs = model.run(&input, &shape).expect("inference failed");

    for (i, out) in outputs.iter().enumerate() {
        println!("output {i}: shape {:?}, values {:?}", out.shape, out.data);
    }
}
