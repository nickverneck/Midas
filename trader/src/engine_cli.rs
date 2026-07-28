use crate::cli::Cli;
use crate::engine_control::{close_and_kill_engine, kill_engine_process};
use crate::engine_registry::{list_running_engines, resolve_engine};
use anyhow::{Result, bail};

pub(crate) fn list_engines() -> Result<()> {
    let engines = list_running_engines()?;
    if engines.is_empty() {
        println!("No running engines found.");
        return Ok(());
    }

    println!("ID\tSTATUS\tSOCKET\tCWD");
    for engine in engines {
        let status = if engine.socket_is_live {
            "live"
        } else {
            "stale"
        };
        println!(
            "{}\t{}\t{}\t{}",
            engine.id,
            status,
            engine.socket_path.display(),
            engine.cwd.display()
        );
    }
    Ok(())
}

pub(crate) async fn kill_engine(id: u32, close: bool) -> Result<()> {
    if close {
        close_and_kill_engine(id).await?;
        println!("Closed the selected market and killed engine {id}.");
    } else {
        kill_engine_process(id).await?;
        println!("Killed engine {id}.");
    }
    Ok(())
}

pub(crate) async fn kill_all_engines(close: bool) -> Result<()> {
    let engines = list_running_engines()?;
    if engines.is_empty() {
        println!("No running engines found.");
        return Ok(());
    }

    let mut failures = Vec::new();
    for engine in engines {
        let result = if close {
            close_and_kill_engine(engine.id).await
        } else {
            kill_engine_process(engine.id).await
        };
        match result {
            Ok(()) if close => println!(
                "Closed the selected market and killed engine {}.",
                engine.id
            ),
            Ok(()) => println!("Killed engine {}.", engine.id),
            Err(err) => failures.push(format!("{}: {err}", engine.id)),
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        bail!("failed to stop some engines: {}", failures.join("; "))
    }
}

pub(crate) fn configure_attach_mode(cli: &mut Cli, id: u32) -> Result<()> {
    let engine = resolve_engine(id)?;
    if !engine.socket_is_live {
        bail!(
            "engine {} is running but its socket {} is unavailable",
            engine.id,
            engine.socket_path.display()
        );
    }

    cli.engine_socket = engine.socket_path;
    cli.no_spawn_engine = true;
    Ok(())
}
