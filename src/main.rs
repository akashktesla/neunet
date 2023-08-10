#![allow(warnings)]
use ndarray_rand::RandomExt;
use polars::prelude::*;
use std::f64::consts::E;
use ndarray::prelude::*;
use ndarray::Array2;
use rand::*;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;

fn main() {
    let df = CsvReader::from_path("train.csv")
        .unwrap()
        .infer_schema(None)
        .has_header(true)
        .finish()
        .unwrap();
    println!("file read");
    let mut rng = rand::thread_rng();
    let mut data:Array2<f64> = df.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();
    data.as_slice_mut().unwrap().shuffle(&mut rng);
    // println!("data: {:?}",data);
    println!("data generated");

    let a = data.shape();
    let m = a[0];
    let n = a[1];
    drop(a);

    let data_dev:Array2<f64> = data.slice(s![0..1000,..]).to_owned();
    let y_dev = data_dev.column(0);
    let x_dev = data_dev.slice(s![..,1..n]).to_owned();
    // println!("x_dev: {:?}",x_dev);
    // println!("y_dev: {:?}",y_dev);

    let data_train:Array2<f64> = data.slice(s![1000..m,..]).to_owned();
    let y_train = data_train.column(0);
    let x_train = data_train.slice(s![..,1..n]).to_owned();
    println!("data splitted");
    // println!("x_train: {:?}",x_train);
    // println!("y_train: {:?}",y_train);
    let w1 = Array2::random((10,784), Uniform::new(-0.5, 0.5));
    let b1 = Array2::random((10,1), Uniform::new(-0.5, 0.5));
    let w2 = Array2::random((10,10), Uniform::new(-0.5, 0.5));
    let b2 = Array2::random((10,1), Uniform::new(-0.5, 0.5));
    let x = x_train.row(0).to_owned();
    println!("x: {:?}",x);
    forward_prop(w1, b1, w2, b2, x);
    // println!("x forward: {:?}", forward_prop(w1, b1, w2, b2,x));

}

fn init_params()->(Array2<f64>,Array2<f64>,Array2<f64>,Array2<f64>){
    let w1 = Array2::random((10,784), Uniform::new(-0.5, 0.5));
    let b1 = Array2::random((10,1), Uniform::new(-0.5, 0.5));
    let w2 = Array2::random((10,10), Uniform::new(-0.5, 0.5));
    let b2 = Array2::random((10,1), Uniform::new(-0.5, 0.5));
    return (w1,b1,w2,b2)
}

fn relu(x:Array2<f64>)->Array2<f64>{
    let vec =  x.map(|s|{ 
        if *s>0.0{
            return *s;
        }
        else{
            return 0.0;
        }
    });
    return vec;
}

fn softmax(x:Array2<f64>)->Array2<f64>{
    let sum = x.map(|s|E.powf(*s)).sum();
    return x.map(|s|E.powf(*s)/sum);
}

fn forward_prop(w1:Array2<f64>,b1:Array2<f64>,w2:Array2<f64>,b2:Array2<f64>,x:Array1<f64>)->Array2<f64>{
    let z1 = w1.dot(&x).into_shape((10,1)).unwrap()+b1;
    println!("z1: {:?}",z1);
    let a1 = relu(z1);
    println!("a1: {:?}",a1);
    let z2 = w2.dot(&a1) + b2;
    println!("z2: {:?}",z2);
    let a2 = softmax(relu(z2));
    println!("a2: {:?}",a2);
    let mut sum = 0.0;
    for i in &a2{
        sum+=*i;
    }
    println!("sum: {}",sum);
    return a2;
}

fn back_prop(){
    
}







