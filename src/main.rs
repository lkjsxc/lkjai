use clap::{Parser, Subcommand};
use lkjai::{cli, config::Config, web};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "lkjai")]
#[command(about = "Local agentic AI system for multi-turn tool use")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Inference,
    Docs {
        #[command(subcommand)]
        action: DocsCommand,
    },
    Quality {
        #[command(subcommand)]
        action: QualityCommand,
    },
}

#[derive(Subcommand)]
enum DocsCommand {
    ValidateTopology,
    ValidateLinks,
}

#[derive(Subcommand)]
enum QualityCommand {
    CheckLines,
    NoNode,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    match Args::parse().command {
        Some(Command::Inference) => inference().await?,
        Some(Command::Docs { action }) => match action {
            DocsCommand::ValidateTopology => cli::docs::validate_topology()?,
            DocsCommand::ValidateLinks => cli::docs::validate_links()?,
        },
        Some(Command::Quality { action }) => match action {
            QualityCommand::CheckLines => cli::quality::check_lines()?,
            QualityCommand::NoNode => cli::quality::no_node()?,
        },
        None => {
            let config = Config::from_env();
            info!("starting lkjai on {}:{}", config.host, config.port);
            web::serve(config).await?;
        }
    }
    Ok(())
}

async fn inference() -> Result<(), Box<dyn std::error::Error>> {
    info!("starting lkjai inference runtime");
    lkjai::inference::serve().await
}

fn init_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "lkjai=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}
