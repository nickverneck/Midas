//! Chronological window sampler for rolling train/val/test splits without lookahead.

/// Generate (start, end) index windows over `len` with given `window` size and `step`.
/// Windows are half-open [start, end), end <= len.
pub fn windows(len: usize, window: usize, step: usize) -> Vec<(usize, usize)> {
    if window == 0 || step == 0 || len < window {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start + window <= len {
        out.push((start, start + window));
        start += step;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_correct_windows() {
        let w = windows(10, 4, 3);
        assert_eq!(w, vec![(0, 4), (3, 7), (6, 10)]);
    }

    #[test]
    fn empty_when_window_too_big() {
        let w = windows(3, 5, 1);
        assert!(w.is_empty());
    }
}
