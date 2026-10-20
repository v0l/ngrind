#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Setup,
    Search,
}

pub trait Control: Sync {
    fn add(&self, n: u64);
    fn stopped(&self) -> bool;
    fn phase(&self, _: Phase) {}
    fn log(&self, _: &str) {}
}

pub type OnFound<'a> = &'a (dyn Fn([u8; 32], [u8; 32]) + Sync);
