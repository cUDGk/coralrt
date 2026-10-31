// From TensorFlow Lite 2.5 (Copyright The TensorFlow Authors, Apache License 2.0), translated and changed; see NOTICE.
//! Hash tables kept as resources of the interpreter (hashtable*.cc, experimental/resource/static_hashtable.cc):
//! int64 -> string or string -> int64, filled once by HASHTABLE_IMPORT.

use std::collections::HashMap;

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{TType, Tensor};
use crate::tflite::OpSpec;
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    Ok(Some(match s.code {
        136 => Box::new(Hashtable { id: opt(s, 0, 0), key: TType::from_code(opt(s, 1, 0i8)), value: TType::from_code(opt(s, 2, 0i8)) }),
        137 => Box::new(Find),
        138 => Box::new(Import),
        139 => Box::new(Size),
        _ => return Ok(None),
    }))
}

/// StaticHashtable: keys are int64 or byte strings
#[derive(Default)]
struct Table {
    key: Option<TType>,
    map: HashMap<Vec<u8>, Vec<u8>>,
    initialized: bool,
}

/// the table a resource tensor names
fn table<'a>(env: &'a mut Env, handle: &Tensor) -> Result<&'a mut Table> {
    let id = handle.as_slice::<i32>().first().copied().unwrap_or(-1);
    env.resources
        .get_mut(&id)
        .and_then(|r| r.downcast_mut::<Table>())
        .ok_or_else(|| crate::Error(format!("no hash table {id}")))
}

/// the elements of an int64 or string tensor as byte keys
fn items(t: &Tensor) -> Result<Vec<Vec<u8>>> {
    Ok(match t.dtype {
        TType::I64 => t.as_slice::<i64>().iter().map(|v| v.to_le_bytes().to_vec()).collect(),
        TType::Str => t.strings()?.into_iter().map(|s| s.to_vec()).collect(),
        d => return err(format!("{d:?} keys or values are not supported")),
    })
}

fn check_handle(t: &Tensor) -> Result<()> {
    if t.dtype != TType::Resource || t.shape != [1] {
        return err("the table handle must be a resource tensor of shape [1]");
    }
    Ok(())
}

fn types_ok(key: TType, value: TType) -> bool {
    matches!((key, value), (TType::I64, TType::Str) | (TType::Str, TType::I64))
}

/// HASHTABLE: the handle of table `id`, made empty on its first run
struct Hashtable {
    id: i32,
    key: TType,
    value: TType,
}

impl Kernel for Hashtable {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if io.n_in() != 0 || !types_ok(self.key, self.value) || io.outs[0].dtype != TType::Resource {
            return err("HASHTABLE maps int64 to string or string to int64 into a resource output");
        }
        io.outs[0].resize(vec![1]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        io.outs[0].as_mut_slice::<i32>()[0] = self.id;
        env.resources.entry(self.id).or_insert_with(|| Box::new(Table { key: Some(self.key), ..Default::default() }));
        Ok(())
    }
}

/// HASHTABLE_FIND (handle, keys, default): each key's value, else the default's first element
struct Find;

impl Kernel for Find {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (h, k, d) = (io.input(0)?, io.input(1)?, io.input(2)?);
        check_handle(h)?;
        if d.dtype != io.outs[0].dtype || !types_ok(k.dtype, io.outs[0].dtype) {
            return err("HASHTABLE_FIND looks up int64 or string keys into the other type");
        }
        io.outs[0].resize(k.shape.clone());
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        let (h, k, d) = (io.input(0)?, io.input(1)?, io.input(2)?);
        let t = table(env, h)?;
        if !t.initialized {
            return err("hashtable need to be initialized before using");
        }
        if t.key != Some(k.dtype) {
            return err("the keys do not have the table's key type");
        }
        let default = items(d)?.into_iter().next().unwrap_or_default();
        let found: Vec<Vec<u8>> = items(k)?.iter().map(|key| t.map.get(key).cloned().unwrap_or_else(|| default.clone())).collect();
        let o = &mut io.outs[0];
        if o.dtype == TType::Str {
            o.set_strings(&found, k.shape.clone());
        } else {
            o.as_mut_slice::<i64>().iter_mut().zip(&found).for_each(|(x, v)| *x = i64::from_le_bytes(v[..8].try_into().unwrap()));
        }
        Ok(())
    }
}

/// HASHTABLE_IMPORT (handle, keys, values): fills the table on the first call only; of equal keys the first stays
struct Import;

impl Kernel for Import {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        let (h, k, v) = (io.input(0)?, io.input(1)?, io.input(2)?);
        check_handle(h)?;
        if !io.outs.is_empty() || !types_ok(k.dtype, v.dtype) || k.shape != v.shape {
            return err("HASHTABLE_IMPORT takes int64 / string keys and values of the same shape, and has no output");
        }
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        let (h, k, v) = (io.input(0)?, io.input(1)?, io.input(2)?);
        let t = table(env, h)?;
        if t.key != Some(k.dtype) {
            return err("the keys do not have the table's key type");
        }
        if !t.initialized {
            for (key, value) in items(k)?.into_iter().zip(items(v)?) {
                t.map.entry(key).or_insert(value);
            }
            t.initialized = true;
        }
        Ok(())
    }
}

/// HASHTABLE_SIZE (handle): the number of keys, as int64 [1]
struct Size;

impl Kernel for Size {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        check_handle(io.input(0)?)?;
        if io.outs[0].dtype != TType::I64 {
            return err("HASHTABLE_SIZE gives int64");
        }
        io.outs[0].resize(vec![1]);
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        let n = table(env, io.input(0)?)?.map.len() as i64;
        io.outs[0].as_mut_slice::<i64>()[0] = n;
        Ok(())
    }
}
