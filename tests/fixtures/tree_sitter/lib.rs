use std::collections::{HashMap, HashSet as Set};
use std::io;

/// Classify a number.
fn classify(x: i32) -> i32 {
    if x > 0 && x < 5 {
        return 1;
    }
    match x {
        1 => 2,
        _ => 3,
    }
}

struct Store;

impl Store {
    fn get(&self) -> Option<i32> {
        return None;
        let _y = HashMap::<i32, i32>::new();
    }
}
