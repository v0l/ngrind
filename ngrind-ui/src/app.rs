use crate::fmt;
use crate::job::{Backend, Kind, PowHit, Run, VanityFound};
use crate::work;
use egui::text::{LayoutJob, TextFormat};
use egui::{Color32, Margin, Ui, Vec2};
use egui_bench::prelude::*;
use ngrind::Phase;
use ngrind::backend::{Backend as Device, best};
use ngrind::event::{Template, parse_tags};
use ngrind::pattern::{HEX_MAX, NPUB_CHARS};
use ngrind::{Alphabet, Pattern, cpu, keys};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Duration;

const STORAGE_KEY: &str = "ngrind-settings";
const SIDE_W: f32 = 340.0;
const MAX_DIFFICULTY: u32 = 64;
const SLOW_AFTER: f64 = 3.0 * 86_400.0;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    kind: Kind,
    backend: Backend,
    alphabet: Alphabet,
    prefix: String,
    suffix: String,
    stop_after: usize,
    threads: usize,
    device: u32,
    pubkey: String,
    event_kind: u16,
    content: String,
    tags: String,
    difficulty: u32,
    fast: bool,
    save: bool,
    rates: [[f64; 3]; 2],
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            kind: Kind::Npub,
            backend: Backend::Cuda,
            alphabet: Alphabet::Npub,
            prefix: String::new(),
            suffix: String::new(),
            stop_after: 1,
            threads: cpu::cores(),
            device: 0,
            pubkey: String::new(),
            event_kind: 1,
            content: String::new(),
            tags: String::new(),
            difficulty: 24,
            fast: false,
            save: true,
            rates: [[0.0; 3]; 2],
        }
    }
}

enum Target {
    Vanity(Pattern),
    Event {
        template: Template,
        secret: Option<[u8; 32]>,
    },
}

impl Target {
    fn expected(&self) -> f64 {
        match self {
            Target::Vanity(p) => p.expected(),
            Target::Event { template, .. } => (template.difficulty as f64).exp2(),
        }
    }

    fn label(&self) -> String {
        match self {
            Target::Vanity(p) => p.describe(),
            Target::Event { template, .. } => {
                format!("{} bits on kind {}", template.difficulty, template.kind)
            }
        }
    }
}

pub struct App {
    s: Settings,
    nsec: String,
    devices: Vec<Device>,
    run: Option<Run>,
    revealed: HashSet<usize>,
    notice: Option<(String, bool)>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, devices: Vec<Device>) -> App {
        egui_bench::install(&cc.egui_ctx);
        let mut s: Settings = cc
            .storage
            .and_then(|st| st.get_string(STORAGE_KEY))
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        if !devices.iter().any(|d| d.kind == s.backend.kind()) {
            s.backend = Backend::of(best(&devices).kind);
        }
        s.threads = s.threads.clamp(1, cpu::cores());
        s.stop_after = s.stop_after.clamp(1, 100);
        App {
            s,
            nsec: String::new(),
            devices,
            run: None,
            revealed: HashSet::new(),
            notice: None,
        }
    }

    fn target(&self) -> Result<Target, String> {
        match self.s.kind {
            Kind::Npub => {
                Pattern::parse(self.s.alphabet, &self.s.prefix, &self.s.suffix).map(Target::Vanity)
            }
            Kind::Event => {
                let nsec = self.nsec.trim();
                let (public, secret) = if nsec.is_empty() {
                    (keys::parse_public(&self.s.pubkey)?, None)
                } else {
                    let secret = keys::parse_secret(nsec)?;
                    (keys::public_of(&secret)?, Some(secret))
                };
                let tags = parse_tags(&self.s.tags)?;
                let template = Template {
                    pubkey: keys::hex(&public),
                    kind: self.s.event_kind,
                    tags,
                    content: self.s.content.clone(),
                    difficulty: self.s.difficulty,
                    fast: self.s.fast,
                };
                Ok(Target::Event { template, secret })
            }
        }
    }

    fn rate_hint(&self, kind: Kind, backend: Backend) -> f64 {
        self.s.rates[kind_slot(kind)][backend.slot()]
    }

    fn begin(&mut self, ctx: &egui::Context, target: Target) {
        let threads = self.s.threads;
        let devices = self.devices.clone();
        let (kind, expected, label) = (self.s.kind, target.expected(), target.label());
        let spawn = match target {
            Target::Vanity(pattern) => work::vanity(pattern, threads, devices),
            Target::Event { template, secret } => work::pow(template, secret, threads, devices),
        };
        self.revealed.clear();
        self.notice = None;
        self.run = Some(Run::start(
            ctx,
            kind,
            self.s.backend,
            label,
            expected,
            self.s.stop_after,
            spawn,
        ));
    }

    fn set_backend(&mut self, backend: Backend) {
        self.s.backend = backend;
        if let Some(run) = self.run.as_mut() {
            run.switch(backend);
        }
    }

    fn poll(&mut self) {
        let Some(run) = self.run.as_mut() else { return };
        run.poll();
        if run.live() && run.rate > 0.0 {
            self.s.rates[kind_slot(run.kind)][run.backend.slot()] = run.rate;
        }
        if !self.s.save || cfg!(target_arch = "wasm32") {
            run.saved_vanity = run.vanity.len();
            run.saved_pow = run.pow.len();
            return;
        }
        let mut lines = Vec::new();
        for f in &run.vanity[run.saved_vanity..] {
            lines.push(vanity_record(f));
        }
        for f in &run.pow[run.saved_pow..] {
            lines.push(f.found.event.clone());
        }
        run.saved_vanity = run.vanity.len();
        run.saved_pow = run.pow.len();
        #[cfg(not(target_arch = "wasm32"))]
        for line in lines {
            self.notice = Some(match keys::record(&line) {
                Ok(path) => (format!("saved to {}", path.display()), false),
                Err(e) => (format!("could not save: {e}"), true),
            });
        }
        #[cfg(target_arch = "wasm32")]
        let _ = lines;
    }
}

fn kind_slot(kind: Kind) -> usize {
    match kind {
        Kind::Npub => 0,
        Kind::Event => 1,
    }
}

fn vanity_record(f: &VanityFound) -> String {
    format!(
        "{}\n{}\n{}",
        keys::npub(&f.public),
        keys::nsec(&f.secret),
        keys::hex(&f.secret)
    )
}

fn face(font: egui::FontId, color: Color32) -> TextFormat {
    TextFormat {
        font_id: font,
        color,
        ..Default::default()
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll();
        egui::Panel::top("header")
            .frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(Margin {
                left: 12,
                right: 12,
                top: 8,
                bottom: 6,
            }))
            .show(ui, |ui| self.header(ui));
        egui::Panel::bottom("footer")
            .frame(
                egui::Frame::NONE
                    .fill(CHASSIS)
                    .inner_margin(Margin::symmetric(12, 6)),
            )
            .show(ui, |ui| self.footer(ui));
        egui::Panel::left("setup")
            .exact_size(SIDE_W)
            .resizable(false)
            .frame(egui::Frame::NONE.fill(CHASSIS).inner_margin(8))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 8.0;
                        self.setup(ui, &ctx);
                    })
            });
        egui::CentralPanel::no_frame().show(ui, |ui| {
            let r = ui.max_rect();
            ui.painter().rect_filled(r, 0.0, CHASSIS);
            ui.painter()
                .line_segment([r.left_top(), r.right_top()], egui::Stroke::new(1.0, ETCH));
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    egui::Frame::NONE
                        .inner_margin(Margin::same(10))
                        .show(ui, |ui| {
                            ui.spacing_mut().item_spacing.y = 8.0;
                            self.body(ui);
                        });
                });
        });
        if self.run.as_ref().is_some_and(|r| r.live()) {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Ok(text) = serde_json::to_string(&self.s) {
            storage.set_string(STORAGE_KEY, text);
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(run) = self.run.as_mut() {
            run.shutdown();
        }
    }
}

impl App {
    fn header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let mut job = LayoutJob::default();
            job.append(
                "NGRIND",
                0.0,
                TextFormat {
                    font_id: legend_font(16.0),
                    color: VALUE,
                    extra_letter_spacing: 2.0,
                    ..Default::default()
                },
            );
            job.append(env!("CARGO_PKG_VERSION"), 10.0, face(figure(12.0), LEGEND));
            ui.label(job);
            ui.add_space(28.0);
            ui.push_id("kind", |ui| {
                segments(
                    ui,
                    &mut self.s.kind,
                    &[(Kind::Npub, "npub", None), (Kind::Event, "event", None)],
                )
            });
            ui.add_space(28.0);
            Line::new().legend("device").show(ui);
            let mut backend = self.s.backend;
            let options: Vec<(Backend, &str, Option<usize>)> =
                [Backend::Cuda, Backend::Wgpu, Backend::Cpu]
                    .into_iter()
                    .filter(|b| self.devices.iter().any(|d| d.kind == b.kind()))
                    .map(|b| (b, b.label(), None))
                    .collect();
            if ui
                .push_id("device", |ui| segments(ui, &mut backend, &options))
                .inner
            {
                self.set_backend(backend);
            }
        });
    }

    fn footer(&mut self, ui: &mut Ui) {
        let r = ui.max_rect();
        ui.painter().line_segment(
            [
                r.left_top() - Vec2::new(12.0, 6.0),
                r.right_top() + Vec2::new(12.0, -6.0),
            ],
            egui::Stroke::new(1.0, ETCH),
        );
        ui.horizontal(|ui| match (&self.notice, self.run.as_ref()) {
            (Some((text, fault)), _) => {
                status(ui, !*fault, text);
                if ui.add(Key::new("dismiss").outline(VALUE)).clicked() {
                    self.notice = None;
                }
            }
            (None, Some(run)) if run.error.is_some() => {
                status(ui, false, run.error.as_deref().unwrap_or_default());
            }
            (None, Some(run)) if run.live() => {
                let text = match run.switching() {
                    Some(b) => format!("moving to the {}", b.label()),
                    None => match run.phase {
                        Phase::Setup => format!("starting the {}", run.backend.label()),
                        Phase::Search => format!("searching on the {}", run.backend.label()),
                    },
                };
                status(ui, true, &text);
            }
            _ => {
                let text = self
                    .devices
                    .iter()
                    .find(|d| d.kind == self.s.backend.kind())
                    .map(|d| d.label())
                    .unwrap_or_default();
                Line::new().note(text).show(ui);
            }
        });
    }

    fn setup(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        let live = self.run.as_ref().is_some_and(|r| r.live());
        match self.s.kind {
            Kind::Npub => self.npub_form(ui),
            Kind::Event => self.event_form(ui),
        }
        self.device_form(ui, live);
        let target = self.target();
        section(ui, "run", "", |ui| {
            match &target {
                Ok(t) => self.estimate(ui, t),
                Err(e) => status(ui, false, e),
            }
            if cfg!(not(target_arch = "wasm32")) {
                switch(
                    ui,
                    "save",
                    &mut self.s.save,
                    "keep what is found",
                    "Appends to found.txt in the ngrind data folder, readable by you only.",
                );
            } else {
                hint(
                    ui,
                    "Nothing is saved or sent anywhere. Copy what you find before closing the tab.",
                );
            }
            row(ui, "stop after", |ui| {
                let one = self.s.stop_after == 1;
                ui.add(
                    egui::DragValue::new(&mut self.s.stop_after)
                        .range(1..=100)
                        .suffix(match (self.s.kind, one) {
                            (Kind::Npub, true) => " match",
                            (Kind::Npub, false) => " matches",
                            (Kind::Event, true) => " event",
                            (Kind::Event, false) => " events",
                        }),
                );
            });
            if live {
                if ui
                    .add(
                        Key::new("stop")
                            .solid(FAULT)
                            .min_width(ui.available_width()),
                    )
                    .clicked()
                    && let Some(run) = self.run.as_ref()
                {
                    run.stop();
                }
            } else if ui
                .add(
                    Key::new("start")
                        .solid(READOUT)
                        .enabled(target.is_ok())
                        .min_width(ui.available_width()),
                )
                .clicked()
                && let Ok(t) = target
            {
                self.begin(ctx, t);
            }
        });
    }

    fn estimate(&self, ui: &mut Ui, target: &Target) {
        let expected = target.expected();
        let (unit, what) = match target {
            Target::Vanity(p) => ("keys", format!("{:.1} bits", p.bits())),
            Target::Event { template, .. } => ("hashes", format!("{} bits", template.difficulty)),
        };
        reading(
            ui,
            "odds",
            format!("1 in {} {unit} ({what})", fmt::count(expected)),
        );
        let rate = self.rate_hint(self.s.kind, self.s.backend);
        if rate > 0.0 {
            let wait = expected / rate;
            reading(
                ui,
                "typical wait",
                format!("{} at {}{unit}/s", fmt::duration(wait), fmt::count(rate)),
            );
            if wait > SLOW_AFTER {
                note(
                    ui,
                    "That is a long search. Shorten the target if you want it today.",
                    WARN,
                );
            }
        } else {
            hint(ui, "The wait is shown once this device has run once.");
        }
    }

    fn npub_form(&mut self, ui: &mut Ui) {
        let live = self.run.as_ref().is_some_and(|r| r.live());
        section(ui, "vanity key", "", |ui| {
            ui.add_enabled_ui(!live, |ui| {
                row_help(
                    ui,
                    "match",
                    "npub patterns use the bech32 alphabet. Hex patterns match the raw public key.",
                    |ui| {
                        choice(
                            ui,
                            "alphabet",
                            &mut self.s.alphabet,
                            [
                                (Alphabet::Npub, "npub".to_string()),
                                (Alphabet::Hex, "hex".to_string()),
                            ],
                        );
                    },
                );
                row_help(
                    ui,
                    "prefix",
                    "Characters the key must start with, after npub1.",
                    |ui| {
                        field(ui, &mut self.s.prefix, "e.g. alice");
                    },
                );
                row_help(
                    ui,
                    "suffix",
                    "Characters the key must end with, before the checksum.",
                    |ui| {
                        field(ui, &mut self.s.suffix, "optional");
                    },
                );
            });
            let limit = match self.s.alphabet {
                Alphabet::Npub => NPUB_CHARS,
                Alphabet::Hex => HEX_MAX,
            };
            hint(
                ui,
                &format!("Up to {limit} characters in total. Letters are case insensitive."),
            );
        });
    }

    fn event_form(&mut self, ui: &mut Ui) {
        let live = self.run.as_ref().is_some_and(|r| r.live());
        section(ui, "event", "nip-13 proof of work", |ui| {
            ui.add_enabled_ui(!live, |ui| {
                row_help(ui, "sign with", "An nsec signs the finished event. It stays in this window and is never saved.", |ui| {
                    secret(ui, &mut self.nsec);
                });
                if self.nsec.trim().is_empty() {
                    row_help(ui, "public key", "An npub or 64 digit hex key. Without an nsec you get the tags to paste into your event.", |ui| {
                        field(ui, &mut self.s.pubkey, "npub1... or hex");
                    });
                }
                row(ui, "kind", |ui| {
                    ui.add(egui::DragValue::new(&mut self.s.event_kind).range(0..=65535));
                });
                row(ui, "difficulty", |ui| {
                    ui.add(egui::DragValue::new(&mut self.s.difficulty).range(1..=MAX_DIFFICULTY).suffix(" bits"));
                });
                switch(
                    ui,
                    "nonce",
                    &mut self.s.fast,
                    "fast mode",
                    "Puts the nonce at the end of the content so long notes hash as fast as short ones. The content gains a trailing line with the number.",
                );
                ui.add_space(4.0);
                Line::new().legend("content").show(ui);
                prose(ui, &mut self.s.content, "the note to publish", 4);
                Line::new().legend("tags").show(ui);
                prose(ui, &mut self.s.tags, r#"[["t","bitcoin"]]"#, 2);
            });
        });
    }

    fn device_form(&mut self, ui: &mut Ui, live: bool) {
        section(ui, "device", "", |ui| {
            let kind = self.s.backend.kind();
            let choices: Vec<&Device> = self.devices.iter().filter(|d| d.kind == kind).collect();
            if kind == ngrind::backend::Kind::Cpu {
                row(ui, "threads", |ui| {
                    ui.add_enabled(
                        !live,
                        egui::DragValue::new(&mut self.s.threads).range(1..=cpu::cores()),
                    );
                });
            } else if choices.len() > 1 {
                row(ui, "gpu", |ui| {
                    ui.add_enabled_ui(!live, |ui| {
                        let names: Vec<(u32, String)> = choices
                            .iter()
                            .enumerate()
                            .map(|(i, d)| (i as u32, d.name.clone()))
                            .collect();
                        choice(ui, "gpu", &mut self.s.device, names);
                    });
                });
            } else if let Some(d) = choices.first() {
                reading(ui, "gpu", d.name.clone());
            }
            hint(
                ui,
                match kind {
                    ngrind::backend::Kind::Cuda => {
                        "NVIDIA only. Switch to wgpu or the cpu above at any time and the search carries on there."
                    }
                    ngrind::backend::Kind::Wgpu => {
                        "Works on any GPU with Vulkan, Metal or DirectX 12. Slower than cuda on NVIDIA cards."
                    }
                    ngrind::backend::Kind::Cpu => {
                        "Switch to a gpu above at any time and the search carries on there."
                    }
                },
            );
        });
    }
}

impl App {
    fn body(&mut self, ui: &mut Ui) {
        let Some(run) = self.run.as_ref() else {
            card(
                ui,
                None,
                |ui| {
                    Line::new().legend("idle").show(ui);
                },
                |ui| {
                    note(
                        ui,
                        "Pick a target on the left and press start. Progress, speed and anything found show up here.",
                        LEGEND,
                    );
                },
            );
            return;
        };
        progress_card(ui, run);
        if run.found() > 0 {
            heading(ui, "found");
        }
        let mut toggled = None;
        for (i, f) in run.vanity.iter().enumerate().rev() {
            let shown = self.revealed.contains(&i);
            let pattern = Pattern::parse(self.s.alphabet, &self.s.prefix, &self.s.suffix).ok();
            if vanity_card(ui, i, f, pattern.as_ref(), shown) {
                toggled = Some(i);
            }
        }
        for (i, f) in run.pow.iter().enumerate().rev() {
            pow_card(ui, i, f);
        }
        if let Some(i) = toggled
            && !self.revealed.remove(&i)
        {
            self.revealed.insert(i);
        }
        log_card(ui, run);
    }
}

fn progress_card(ui: &mut Ui, run: &Run) {
    let (rail, stage) = match (&run.error, run.live(), run.found(), run.phase) {
        (Some(_), _, _, _) => (FAULT, Some(1)),
        (None, _, n, _) if n > 0 => (OK, Some(2)),
        (None, true, _, Phase::Setup) => (TRACE, Some(0)),
        (None, true, _, Phase::Search) => (TRACE, Some(1)),
        (None, false, _, _) => (ETCH, Some(1)),
    };
    let (unit, noun) = match run.kind {
        Kind::Npub => ("keys", "keys searched"),
        Kind::Event => ("hashes", "hashes tried"),
    };
    card(
        ui,
        Some(rail),
        |ui| {
            Line::new()
                .legend("search")
                .note(run.target.clone())
                .show(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                lamp(ui, run.backend.label(), run.live(), run.error.is_some());
            });
        },
        |ui| {
            let muted = !run.live() && run.found() == 0;
            stage_rail(ui, &["setup", "search", "found"], stage, rail, 22.0, muted);
            ui.add_space(6.0);

            let (n, p) = fmt::scaled(if run.live() { run.rate } else { run.average() });
            let caption = if run.live() { "speed" } else { "average speed" };
            hero(ui, caption, &n, &format!("{p}{unit}/s"), TRACE);
            ui.add_space(4.0);

            let done = run.done() as f64;
            let rate = if run.live() { run.rate } else { run.average() };
            let chance = fmt::chance(done, run.expected);
            readouts(
                ui,
                &[
                    (noun, fmt::count(done), TRACE),
                    ("running", fmt::clock(run.elapsed()), VALUE),
                    (
                        "average",
                        format!("{}{unit}/s", fmt::count(run.average())),
                        TRACE,
                    ),
                    (
                        "found",
                        run.found().to_string(),
                        if run.found() > 0 { OK } else { VALUE },
                    ),
                    ("odds so far", format!("{:.1} %", chance * 100.0), VALUE),
                    (
                        "50 % by",
                        fmt::duration(fmt::median_seconds(run.expected, rate)),
                        VALUE,
                    ),
                ],
            );
            progress(
                ui,
                "chance of a hit by now",
                chance as f32,
                Some(1.0),
                &format!(
                    "{:.2} % of {} expected {unit}",
                    chance * 100.0,
                    fmt::count(run.expected)
                ),
            );
            if run.history.len() > 1 {
                ui.add_space(4.0);
                Line::new().legend("speed history").show(ui);
                let top = run.history.iter().copied().fold(0.0f32, f32::max).max(1.0) * 1.15;
                ui.add(
                    Trace::new(&run.history)
                        .size(Vec2::new(ui.available_width(), 56.0))
                        .range(0.0, top)
                        .filled(true),
                );
            }
            if let Some(e) = &run.error {
                status(ui, false, e);
            }
        },
    );
}

fn highlighted(text: &str, lit: &[(usize, usize)], head: usize) -> LayoutJob {
    let mut job = LayoutJob::default();
    job.wrap.max_width = f32::INFINITY;
    let font = figure(12.5);
    let mut at = 0;
    let mut pieces: Vec<(usize, usize)> = lit.to_vec();
    pieces.sort();
    for (from, to) in pieces {
        let (from, to) = (from.min(text.len()), to.min(text.len()));
        if from < at || from >= to {
            continue;
        }
        let colour = |slice_start: usize| if slice_start < head { LEGEND } else { VALUE };
        if from > at {
            job.append(&text[at..from], 0.0, face(font.clone(), colour(at)));
        }
        job.append(&text[from..to], 0.0, face(font.clone(), READOUT));
        at = to;
    }
    if at < text.len() {
        job.append(&text[at..], 0.0, face(font, VALUE));
    }
    job
}

fn vanity_card(
    ui: &mut Ui,
    i: usize,
    f: &VanityFound,
    pattern: Option<&Pattern>,
    revealed: bool,
) -> bool {
    let npub = keys::npub(&f.public);
    let hex = keys::hex(&f.public);
    let nsec = keys::nsec(&f.secret);
    let mut toggle = false;
    card(
        ui,
        Some(OK),
        |ui| {
            Line::new()
                .legend("match")
                .value(format!("#{}", i + 1))
                .show(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                Line::new()
                    .note(format!("after {} keys", fmt::count(f.after as f64)))
                    .size(10.5)
                    .show(ui);
            });
        },
        |ui| {
            let (npub_lit, hex_lit) = match pattern {
                Some(p) => {
                    let n = p.prefix.len();
                    let s = p.suffix.len();
                    match p.alphabet {
                        Alphabet::Npub => (
                            vec![(5, 5 + n), (5 + NPUB_CHARS - s, 5 + NPUB_CHARS)],
                            Vec::new(),
                        ),
                        Alphabet::Hex => (Vec::new(), vec![(0, n), (64 - s, 64)]),
                    }
                }
                None => (Vec::new(), Vec::new()),
            };
            ui.add(egui::Label::new(highlighted(&npub, &npub_lit, 5)).wrap());
            ui.add(egui::Label::new(highlighted(&hex, &hex_lit, 0)).wrap());
            let secret_text = if revealed {
                nsec.clone()
            } else {
                "•".repeat(24)
            };
            Line::new()
                .legend("nsec")
                .column(ui, 50.0)
                .value(secret_text)
                .size(12.5)
                .elided(ui);
            ui.horizontal(|ui| {
                if ui.add(Key::new("copy npub").outline(VALUE)).clicked() {
                    ui.ctx().copy_text(npub.clone());
                }
                if ui.add(Key::new("copy nsec").outline(READOUT)).clicked() {
                    ui.ctx().copy_text(nsec.clone());
                }
                if ui
                    .add(Key::new(if revealed { "hide nsec" } else { "show nsec" }).outline(VALUE))
                    .clicked()
                {
                    toggle = true;
                }
            });
        },
    );
    toggle
}

fn pow_card(ui: &mut Ui, i: usize, hit: &PowHit) {
    let (f, after) = (&hit.found, hit.after);
    card(
        ui,
        Some(OK),
        |ui| {
            Line::new()
                .legend("event")
                .value(format!("#{}", i + 1))
                .show(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let text = if f.signed { "signed" } else { "unsigned" };
                Line::new()
                    .note(format!("{text}, after {} hashes", fmt::count(after as f64)))
                    .size(10.5)
                    .show(ui);
            });
        },
        |ui| {
            let zeros = f.id.chars().take_while(|c| *c == '0').count();
            ui.add(egui::Label::new(highlighted(&f.id, &[(0, zeros)], 0)).wrap());
            readouts(
                ui,
                &[
                    ("leading zero bits", f.bits.to_string(), TRACE),
                    ("nonce", f.nonce.to_string(), VALUE),
                    ("created at", f.created_at.to_string(), VALUE),
                ],
            );
            if !f.signed {
                hint(
                    ui,
                    "No nsec was given, so the event has no signature. Take created_at and the tags from it and sign it yourself.",
                );
            }
            let shown: String = f.event.chars().take(360).collect();
            let more = if f.event.chars().count() > 360 {
                "…"
            } else {
                ""
            };
            Line::new()
                .note(format!("{shown}{more}"))
                .size(10.5)
                .wrapped(ui);
            ui.horizontal(|ui| {
                if ui.add(Key::new("copy event").outline(READOUT)).clicked() {
                    ui.ctx().copy_text(f.event.clone());
                }
                if ui.add(Key::new("copy id").outline(VALUE)).clicked() {
                    ui.ctx().copy_text(f.id.clone());
                }
            });
        },
    );
}

fn log_card(ui: &mut Ui, run: &Run) {
    section(ui, "log", "", |ui| {
        for (t, text) in run.log.iter().rev().take(8) {
            Line::new()
                .measured(fmt::clock(*t))
                .size(11.0)
                .note(text.clone())
                .show(ui);
        }
    });
}
