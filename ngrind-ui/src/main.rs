#![cfg_attr(
    all(target_os = "windows", not(debug_assertions)),
    windows_subsystem = "windows"
)]

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    if std::env::args().any(|a| a == "--render-check") {
        let ok = ngrind_ui::render_check::report();
        std::process::exit(if ok { 0 } else { 1 });
    }
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1120.0, 780.0])
        .with_min_inner_size([760.0, 520.0])
        .with_title("ngrind")
        .with_app_id("ngrind");
    if let Ok(icon) = eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon.png")) {
        viewport = viewport.with_icon(icon);
    }
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = ngrind_ui::render_check::pick_backends();
    }
    let options = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
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
