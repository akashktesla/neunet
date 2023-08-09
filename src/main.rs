#![allow(warnings)]
use polars::prelude::*;
use ndarray::prelude::*;
use ndarray::Array2;
use rand::*;

fn main() {
    let df = CsvReader::from_path("train.csv")
        .unwrap()
        .infer_schema(None)
        .has_header(true)
        .finish()
        .unwrap();
    println!("{:?}",df.head(Some(5)));
    let rng = rand::thread_rng();
    let mut data:Array2<i64> = df.to_ndarray::<Int64Type>(IndexOrder::C).unwrap();
    let slice = data.as_slice_mut();

    let a = data.shape();
    let m = a[0];
    let n = a[1];
    drop(a);

    let data_dev:Array2<i64> = data.slice(s![0..1000,..]).to_owned();
    let y_dev = data_dev.column(0);
    let x_dev = data_dev.slice(s![..,1..n]).to_owned();
    println!("x_dev: {:?}",x_dev);
    println!("y_dev: {:?}",y_dev);

    let data_train:Array2<i64> = data.slice(s![1000..m,..]).to_owned();
    let y_train = data_train.column(0);
    let x_train = data_train.slice(s![..,1..n]).to_owned();
    println!("x_train: {:?}",x_train);
    println!("y_train: {:?}",y_train);

}
