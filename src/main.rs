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
    println!("data generated");

    let a = data.shape();
    let m = a[0];
    let n = a[1];
    drop(a);

    let data_dev:Array2<f64> = data.slice(s![0..1000,..]).to_owned();
    let y_dev = data_dev.column(0).to_owned();
    let x_dev = data_dev.slice(s![..,1..n]).to_owned();
    println!("x_dev: {:?}",x_dev);
    println!("y_dev: {:?}",y_dev);

    let data_train:Array2<f64> = data.slice(s![1000..m,..]).to_owned();
    let y_train = data_train.column(0).to_owned();
    let x_train = data_train.slice(s![..,1..n]).to_owned();
    println!("data splitted");
    println!("x_train: {:?}",x_train);
    println!("y_train: {:?}",y_train);
    let x = x_train.row(0).to_owned();
    println!("x: {:?}",x);
    let (a1,z1,a2,z2) = forward_prop(x_train.clone()); 
    // println!("x forward: {:?}", forward_prop(w1, b1, w2, b2,x));
    let w2 = Array2::random((10,10), Uniform::new(-0.1, 0.11));
    back_prop(z1, a1, z2, a2, w2,x_train,y_train)

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
    // println!("softmax");
    return Array::from_iter(x.axis_iter(Axis(0)).map(|s|{
        let es = s.map(|t|{
            return E.powf(*t);
        });
        let sum = es.sum().min(f64::MAX);
        // println!("sum: {:?}",sum);
        let es = es.map(|t|{
            // println!("t,sum,t/sum {:?},{:?},{:?}",t,sum,t/sum);
            return t.min(f64::MAX)/sum;
        });
        return es;})
        .flatten())
        .into_shape((x.nrows(),x.ncols())).unwrap();
}

fn forward_prop(x:Array2<f64>)->(Array2<f64>,Array2<f64>,Array2<f64>,Array2<f64>){
    let w1 = Array2::random((784,10), Uniform::new(-0.1, 0.1));
    let b1 = Array1::random((10), Uniform::new(-0.1, 0.11));
    let w2 = Array2::random((10,10), Uniform::new(-0.1, 0.11));
    let b2 = Array1::random((10), Uniform::new(-0.1, 0.1));
    // let z1 = w1.dot(&x).into_shape((10,1)).unwrap()+b1;
    let z1 = x.dot(&w1)+b1;
    println!("z1: {:?}",z1);
    let a1 = relu(z1.clone());
    println!("a1: {:?}",a1);
    let z2 = a1.dot(&w2) + b2;
    println!("z2: {:?}",z2);
    let a2 = softmax(z2.clone());
    println!("a2: {:?}",a2);
    let mut sum = 0.0;
    for i in &a2{
        sum+=*i;
    }
    println!("sum: {}",sum);
    return (z1,a1,z2,a2);
}

fn one_hot(y:Array1<f64>)->Array2<f64>{
    println!("y: {:?}",y);
    println!("y: {:?}",y.shape());
    let r = y.shape()[0];
    let mut one_hot_y:Array2<f64> = Array2::<f64>::zeros((r,10)); //10 is no of labels (0 to 9) la
    for i in 0..r{
        let val = y[i] as usize;
        one_hot_y[(i,val)] = 1.0;
    }
    println!("one_hot_y: {:?}",one_hot_y);
    return one_hot_y;
}

fn deriv_relu(z:f64)->f64{
    if z>0.0{
        return 1.0
    }
    return 0.0
}

fn back_prop(z1:Array2<f64>,a1:Array2<f64>,z2:Array2<f64>,a2:Array2<f64>,w2:Array2<f64>,x:Array2<f64>,y:Array1<f64>){
    let m = y.ndim();
    let one_hot_y  = one_hot(y);
    println!("one_hot_y: {:?}",one_hot_y);
    let dz2 = a2- one_hot_y;
    println!("dz2 {:?}",dz2);
    //is it not working or taking too much time (vanthu check pannu)
    //inaikulla itha mudichitu cs50 course start paniranum... this shit is complex AF(fr now)
    let dw2 = (1/m) as f64 * a1.dot(&dz2.t());
    println!("dw2 {:?}",dw2);
    // let db2 = (1/m) as f64 * dz2.sum_axis(Axis(1));
    // println!("db2 {:?}",db2);
    // let dz1 = dz2.dot(&w2.t());
    // println!("dz1 {:?}",dz1);
     
}



// fn main(){
//     let x: Array2<f64> = Array2::from_shape_vec((2, 3), vec![
//         100.0, 200.0, 300.0,
//         400.0, 500.0, 1000.0,
//     ]).unwrap();
//     let a = softmax(x);
//     println!("{:?}",a);

// }


