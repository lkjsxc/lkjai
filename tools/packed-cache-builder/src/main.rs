use clap::{Parser, Subcommand};

mod build;
mod digest;
mod error;
mod metadata;
mod source;
mod validate;

use error::Result;

const FORMAT: &str = "lkjai-packed-cache-v2";

#[derive(Parser)]
#[command(name = "lkjai-packed-cache-builder")]
#[command(about = "Build and validate deterministic lkjai packed-token caches")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Build(build::BuildArgs),
    Validate(validate::ValidateArgs),
}

fn main() {
    if let Err(error) = run_cli() {
        eprintln!("packed cache builder failed: {error}");
        std::process::exit(2);
    }
}

fn run_cli() -> Result<()> {
    match Cli::parse().command {
        Command::Build(args) => {
            let metadata = build::build_cache(&args)?;
            println!(
                "{}",
                serde_json::to_string(&metadata).expect("metadata JSON")
            );
        }
        Command::Validate(args) => {
            let metadata = validate::validate_cache(&args)?;
            println!(
                "{{\"status\":\"pass\",\"format\":\"{}\",\"sequence_count\":{},\"token_count\":{}}}",
                metadata.format, metadata.sequence_count, metadata.token_count
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
