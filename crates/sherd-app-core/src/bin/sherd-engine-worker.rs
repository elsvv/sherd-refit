//! The engine role of the app's binary, as a binary of its own (A §2.1): what the headless tests
//! spawn, and what the Tauri shell reproduces by running itself with `--engine-worker`.

fn main() {
    // The engine's log goes to stderr, which the host appends to the run's `engine.log`; stdout
    // is the protocol's and nothing else may write to it.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("sherd=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_target(false)
        .init();
    // `StdinLock<'static>` is `BufRead` but not `Send`, and `serve` reads the rest of the input on
    // a thread of its own; a `BufReader` over `Stdin` is both.
    let input = std::io::BufReader::new(std::io::stdin());
    let code = sherd_app_core::worker::serve(input, std::io::stdout());
    std::process::exit(code);
}
