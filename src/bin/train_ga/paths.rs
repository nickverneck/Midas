use chrono::Local;
use std::path::{Path, PathBuf};

pub fn resolve_outdir(outdir: PathBuf, default_base: &str) -> PathBuf {
    if !is_default_outdir(&outdir, default_base) {
        return outdir;
    }
    let stamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    outdir.join(stamp)
}

fn is_default_outdir(outdir: &Path, default_base: &str) -> bool {
    outdir == Path::new(default_base) || outdir == Path::new(&format!("./{default_base}"))
}
