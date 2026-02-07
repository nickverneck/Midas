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

/// Shift windows so starts are at least `min_start`.
/// Windows with fewer than two bars after shifting are dropped.
pub fn enforce_min_start(windows: &[(usize, usize)], min_start: usize) -> Vec<(usize, usize)> {
    windows
        .iter()
        .filter_map(|&(start, end)| {
            let adj_start = start.max(min_start);
            if end > adj_start + 1 {
                Some((adj_start, end))
            } else {
                None
            }
        })
        .collect()
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

    #[test]
    fn enforce_min_start_shifts_and_drops_short_windows() {
        let input = vec![(0, 10), (8, 12), (12, 14)];
        let out = enforce_min_start(&input, 9);
        assert_eq!(out, vec![(9, 12), (12, 14)]);

        let out_drop = enforce_min_start(&[(0, 9)], 9);
        assert!(out_drop.is_empty());
    }
}
