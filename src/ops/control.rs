//! Control flow (if.cc, while.cc, call_once.cc): other subgraphs of the model, run through Env.

use super::util::*;
use super::{Io, Kernel};
use crate::interp::Env;
use crate::tensor::{Alloc, TType, Tensor};
use crate::tflite::OpSpec;
use crate::Result;

pub fn create(s: &OpSpec) -> Result<Option<Box<dyn Kernel>>> {
    let index = |i: usize| -> Result<usize> {
        let v: i32 = opt(s, i, -1);
        usize::try_from(v).or_else(|_| err(format!("subgraph index {v} is not valid")))
    };
    Ok(Some(match s.code {
        118 => Box::new(If { then: index(0)?, otherwise: index(1)? }),
        119 => Box::new(While { cond: index(0)?, body: index(1)? }),
        129 => Box::new(CallOnce { init: index(0)?, done: false }),
        _ => return Ok(None),
    }))
}

/// every input of the op, which must all be there
fn inputs<'a>(io: &Io<'a>, from: usize) -> Result<Vec<&'a Tensor>> {
    (from..io.n_in()).map(|k| io.input(k)).collect()
}

/// a subgraph's result into the op's output (2.5 copies the bytes, resizing a dynamic output first)
fn put(o: &mut Tensor, r: &Tensor) -> Result<()> {
    if o.dtype != r.dtype {
        return err(format!("output {:?}, subgraph result {:?}", o.dtype, r.dtype));
    }
    o.shape = r.shape.clone();
    o.buf = r.buf.clone();
    Ok(())
}

fn check_cond(t: &Tensor) -> Result<bool> {
    if t.dtype != TType::Bool || t.elements() != 1 {
        return err("the condition must be a single bool");
    }
    Ok(t.data()[0] != 0)
}

/// IF (cond, inputs...): the then or else subgraph on the inputs
struct If {
    then: usize,
    otherwise: usize,
}

impl Kernel for If {
    fn prepare(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        if io.n_in() == 0 {
            return err("IF needs a condition");
        }
        let c = io.input(0)?;
        if c.dtype != TType::Bool || c.elements() != 1 {
            return err("the condition must be a single bool");
        }
        let ins = inputs(io, 1)?;
        let (a, b) = (env.prepare(self.then, &ins)?, env.prepare(self.otherwise, &ins)?);
        if a.len() != io.outs.len() || b.len() != io.outs.len() {
            return err("both branches must give one tensor per output");
        }
        let dynamic = a.iter().chain(&b).any(|t| t.alloc == Alloc::Dynamic) || a.iter().zip(&b).any(|(x, y)| x.shape != y.shape);
        for (o, t) in io.outs.iter_mut().zip(&a) {
            if dynamic {
                o.alloc = Alloc::Dynamic;
            } else {
                o.resize(t.shape.clone());
            }
        }
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        let branch = if check_cond(io.input(0)?)? { self.then } else { self.otherwise };
        let res = env.run(branch, &inputs(io, 1)?)?;
        for (o, r) in io.outs.iter_mut().zip(&res) {
            put(o, r)?;
        }
        Ok(())
    }
}

/// WHILE (values...): body on the values while cond on them is true; the outputs are the last values
struct While {
    cond: usize,
    body: usize,
}

impl Kernel for While {
    fn prepare(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        let ins = inputs(io, 0)?;
        if io.outs.len() != ins.len() {
            return err("WHILE has one output per input");
        }
        let c = env.prepare(self.cond, &ins)?;
        if c.len() != 1 {
            return err("the condition subgraph must give one tensor");
        }
        if c[0].alloc != Alloc::Dynamic && (c[0].dtype != TType::Bool || c[0].elements() != 1) {
            return err("the condition must be a single bool");
        }
        let b = env.prepare(self.body, &ins)?;
        if b.len() != ins.len() {
            return err("the body must give one tensor per input");
        }
        if b.iter().zip(&ins).any(|(o, i)| o.dtype != i.dtype) {
            return err("the body must keep the inputs' types");
        }
        let dynamic = b.iter().zip(&ins).any(|(o, i)| o.alloc == Alloc::Dynamic || o.shape != i.shape);
        for (o, t) in io.outs.iter_mut().zip(&b) {
            if dynamic {
                o.alloc = Alloc::Dynamic;
            } else {
                o.resize(t.shape.clone());
            }
        }
        Ok(())
    }

    fn eval(&mut self, io: &mut Io, env: &mut Env) -> Result<()> {
        let mut values: Vec<Tensor> = inputs(io, 0)?.into_iter().cloned().collect();
        loop {
            let refs: Vec<&Tensor> = values.iter().collect();
            let c = env.run(self.cond, &refs)?;
            if !check_cond(&c[0])? {
                break;
            }
            values = env.run(self.body, &refs)?;
        }
        for (o, r) in io.outs.iter_mut().zip(&values) {
            put(o, r)?;
        }
        Ok(())
    }
}

/// CALL_ONCE: the init subgraph (no inputs or outputs), on the first run only
struct CallOnce {
    init: usize,
    done: bool,
}

impl Kernel for CallOnce {
    fn prepare(&mut self, io: &mut Io, _: &mut Env) -> Result<()> {
        if !self.done && (io.n_in() != 0 || !io.outs.is_empty()) {
            return err("CALL_ONCE has no inputs or outputs");
        }
        Ok(())
    }

    fn eval(&mut self, _: &mut Io, env: &mut Env) -> Result<()> {
        if !self.done {
            env.run(self.init, &[])?;
            self.done = true;
        }
        Ok(())
    }
}
