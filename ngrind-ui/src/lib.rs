pub mod app;
mod fmt;
mod job;
mod work;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod renderer_tests {
    #[test]
    fn native_default_renderer_is_wgpu() {
        assert_eq!(eframe::Renderer::default(), eframe::Renderer::Wgpu);
    }
}
