#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1120.0, 780.0])
        .with_min_inner_size([760.0, 520.0])
        .with_title("ngrind")
        .with_app_id("ngrind");
    if let Ok(icon) = eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon.png")) {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "ngrind",
        options,
        Box::new(|cc| {
            Ok(Box::new(ngrind_ui::app::App::new(
                cc,
                ngrind::backend::available(),
            )))
        }),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {}
