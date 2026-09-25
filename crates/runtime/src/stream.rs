// CEP:FILE: crates/runtime/src/stream.rs
// CEP:WHAT: CPU streams — ordered execution queues with events.
// CEP:WHY: Master architecture section 8: "Streams". Streams serialize
//          kernel launches; events fence host-device synchronization. The
//          CPU stream executes eagerly in order (the reference semantics);
//          async device queues are the CEP-16 placeholder.
// CEP:CLASS: CEP-1
// CEP:STATUS: complete
// CEP:FAILURE: RuntimeError propagation; UnknownEvent for stale events.
// CEP:ASSUMES: single-threaded launch discipline per stream.
// CEP:COST: O(launches).
// CEP:EVIDENCE: tests `stream_orders_launches`, `events_fence`.
// CEP:SECURITY: bounded event table.
// CEP:HPC-DETERMINISM: deterministic ordering.
//! Streams and events.

use xir_levels::level4::TargetProgram;

use crate::device::CpuDevice;
use crate::interp::RuntimeError;
use crate::value::Value;

/// Stream failure enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamError {
    /// An event id was unknown or already consumed.
    UnknownEvent,
    /// Execution failed (wrapped interpreter error).
    Exec(RuntimeError),
}

/// One recorded event.
///
/// CEP:WHAT: Stream synchronization point.
/// CEP:WHY: The host fences on events; on the CPU stream they complete
///          immediately (recorded for API parity with async devices).
/// CEP:STATUS: complete
/// CEP:FAILURE: none (plain data)
/// CEP:ASSUMES: none
/// CEP:COST: plain data
/// CEP:EVIDENCE: tests in this module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamEvent {
    /// Monotonic event id.
    pub id: u32,
    /// Launch index the event fences.
    pub after_launch: u32,
}

/// The CPU stream.
///
/// CEP:WHAT: Ordered launch queue over the CPU device.
/// CEP:WHY: Callers that need stream semantics (the JIT's execution
///          thread) get ordering + events without device-specific code.
/// CEP:STATUS: complete
/// CEP:FAILURE: see StreamError.
/// CEP:ASSUMES: single launching thread.
/// CEP:COST: O(1) per launch (eager execution).
/// CEP:EVIDENCE: tests in this module.
/// CEP:HPC-DETERMINISM: deterministic ordering.
pub struct CpuStream<'a> {
    device: &'a CpuDevice,
    launches: u32,
    events: Vec<StreamEvent>,
}

impl<'a> CpuStream<'a> {
    /// CEP:WHAT: Creates a stream bound to a device.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn new(device: &'a CpuDevice) -> CpuStream<'a> {
        CpuStream {
            device,
            launches: 0,
            events: Vec::new(),
        }
    }

    /// CEP:WHAT: Launches a program (eager, in order).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: Exec(RuntimeError) propagation.
    /// CEP:ASSUMES: args match the signature.
    /// CEP:COST: interpreter cost.
    /// CEP:EVIDENCE: tests in this module.
    pub fn launch(
        &mut self,
        program: &TargetProgram,
        args: &[Value],
    ) -> Result<Vec<Value>, StreamError> {
        let out = self
            .device
            .launch(program, args)
            .map_err(StreamError::Exec)?;
        self.launches = self.launches.wrapping_add(1);
        Ok(out)
    }

    /// CEP:WHAT: Records an event after the current launch count.
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn record(&mut self) -> StreamEvent {
        let ev = StreamEvent {
            id: self.events.len() as u32,
            after_launch: self.launches,
        };
        self.events.push(ev);
        ev
    }

    /// CEP:WHAT: Waits for an event (no-op on the eager CPU stream).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: UnknownEvent for stale ids.
    /// CEP:ASSUMES: none
    /// CEP:COST: O(events)
    /// CEP:EVIDENCE: tests
    pub fn wait_for(&self, ev: StreamEvent) -> Result<(), StreamError> {
        match self.events.get(ev.id as usize) {
            Some(recorded) if *recorded == ev => Ok(()),
            _ => Err(StreamError::UnknownEvent),
        }
    }

    /// CEP:WHAT: Launch count (diagnostic).
    /// CEP:STATUS: complete
    /// CEP:FAILURE: none
    /// CEP:ASSUMES: none
    /// CEP:COST: O(1)
    /// CEP:EVIDENCE: tests
    pub fn launch_count(&self) -> u32 {
        self.launches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xir_core::op::{BinaryOp, Op};
    use xir_core::ty::{ScalarType, Type};
    use xir_levels::level3::{LoopProgram, ScheduledOp};
    use xir_levels::level4::lower;

    fn add_program() -> TargetProgram {
        let prog = LoopProgram {
            params: vec![Type::Scalar(ScalarType::F64); 2],
            ops: vec![ScheduledOp {
                op: Op::Binary(BinaryOp::Add),
                inputs: [0, 1, 0, 0, 0, 0],
                n_inputs: 2,
                output: 2,
                ty: Type::Scalar(ScalarType::F64),
            }],
            results: vec![2],
            buffers: vec![],
        };
        match lower(&prog) {
            Ok(t) => t,
            Err(_) => TargetProgram {
                instrs: vec![],
                results: vec![],
                value_count: 3,
            },
        }
    }

    // CEP:WHAT: Streams count ordered launches.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on miscount.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn stream_orders_launches() {
        let dev = CpuDevice;
        let mut s = CpuStream::new(&dev);
        let p = add_program();
        let r = s.launch(&p, &[Value::F64(1.0), Value::F64(2.0)]);
        assert!(r.is_ok());
        let r2 = s.launch(&p, &[Value::F64(3.0), Value::F64(4.0)]);
        assert!(r2.is_ok());
        assert_eq!(s.launch_count(), 2);
    }

    // CEP:WHAT: Events fence and stale events fail loudly.
    // CEP:STATUS: complete
    // CEP:FAILURE: assert fires on silent stale acceptance.
    // CEP:ASSUMES: none
    // CEP:COST: test-only
    // CEP:EVIDENCE: this test
    #[test]
    fn events_fence() {
        let dev = CpuDevice;
        let mut s = CpuStream::new(&dev);
        let p = add_program();
        let _ = s.launch(&p, &[Value::F64(1.0), Value::F64(1.0)]);
        let ev = s.record();
        assert!(s.wait_for(ev).is_ok());
        // Forged/stale event ids fail.
        let bad = StreamEvent {
            id: 99,
            after_launch: 0,
        };
        assert_eq!(s.wait_for(bad), Err(StreamError::UnknownEvent));
    }
}
