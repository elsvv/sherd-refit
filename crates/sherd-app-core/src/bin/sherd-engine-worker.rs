//! The engine role of the app's binary, as a binary of its own (A §2.1): what the headless tests
//! spawn, and what the Tauri shell reproduces by running itself with `--engine-worker`. Both are
//! [`sherd_app_core::worker::serve_stdio`] and nothing else, so the two cannot drift apart.

fn main() {
    std::process::exit(sherd_app_core::worker::serve_stdio());
}
