use super::*;

const EPOCH_TICKS: i64 = 621_355_968_000_000_000;
const TICKS_PER_MINUTE: i64 = 600_000_000;

fn header(tick_size: f64, first_price: f64, t0: i64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(28);
    bytes.extend_from_slice(&1_i32.to_le_bytes());
    bytes.extend_from_slice(&tick_size.to_le_bytes());
    bytes.extend_from_slice(&first_price.to_le_bytes());
    bytes.extend_from_slice(&t0.to_le_bytes());
    bytes
}

fn push_be(out: &mut Vec<u8>, value: u64, width: usize) {
    for shift in (0..width).rev() {
        out.push((value >> (shift * 8)) as u8);
    }
}

fn minute_record(
    out: &mut Vec<u8>,
    time_code: u8,
    time_delta: u64,
    open_code: u8,
    open_delta: i64,
    high_code: u8,
    high: u64,
    low_code: u8,
    low: u64,
    close_code: u8,
    close_from_low: u64,
    volume_code: u8,
    volume: u64,
) {
    let b1 = time_code | (open_code << 2) | (volume_code << 5);
    let b2 = (high_code << 4) | (low_code << 6) | close_code;
    out.extend_from_slice(&[b1, b2]);
    match time_code {
        0 => {}
        1 => push_be(out, time_delta, 1),
        2 => push_be(out, time_delta, 2),
        _ => unreachable!(),
    }
    match open_code {
        0 => {}
        1 => out.push((open_delta + 0x80) as u8),
        2 => push_be(out, (open_delta + 0x8000) as u64, 2),
        _ => unreachable!(),
    }
    for (code, value) in [
        (high_code, high),
        (low_code, low),
        (close_code, close_from_low),
    ] {
        push_be(out, value, [0, 1, 2, 4][code as usize]);
    }
    push_be(
        out,
        volume,
        match volume_code {
            1..=4 => 1,
            5 => 2,
            6 => 4,
            _ => unreachable!(),
        },
    );
}

fn tick_record(
    out: &mut Vec<u8>,
    time_code: u8,
    time_delta: u64,
    price_code: u8,
    price_delta: i64,
    spread_code: u8,
    volume_code: u8,
    volume: u64,
) {
    let mut b2 = volume_code << 5;
    let mut price_bytes = Vec::new();
    match price_code {
        0 => {}
        1 => b2 |= ((price_delta + 16) as u8) & 0x1f,
        2 => price_bytes.push((price_delta + 0x80) as u8),
        3 => push_be(&mut price_bytes, (price_delta + 0x8000_0000) as u64, 4),
        _ => unreachable!(),
    }
    let b1 = time_code | (spread_code << 3) | (price_code << 6);
    out.extend_from_slice(&[b1, b2]);
    match time_code {
        0 => {}
        1 => push_be(out, time_delta, 1),
        2 => push_be(out, time_delta, 2),
        3 => push_be(out, time_delta, 4),
        4 => push_be(out, time_delta, 8),
        5 => push_be(out, time_delta, 1), // seconds
        6 => push_be(out, time_delta, 2), // seconds
        _ => unreachable!(),
    }
    out.extend_from_slice(&price_bytes);
    out.resize(out.len() + spread_code.saturating_sub(5) as usize, 0xA5);
    push_be(
        out,
        volume,
        match volume_code {
            1 | 2 => 1,
            5 => 2,
            _ => unreachable!(),
        },
    );
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

#[test]
fn minute_decodes_deltas_widths_and_scaled_volume() {
    let mut bytes = header(0.25, 100.0, EPOCH_TICKS);
    // First record advances zero minutes via the standard one-minute time code's first-row rule
    // in the file format's initialization convention: time delta code 1 with payload zero.
    minute_record(&mut bytes, 1, 0, 0, 0, 1, 4, 1, 2, 1, 5, 1, 7);
    // Subsequent records exercise u16 time/open deltas and the wider H/L/C encodings.
    minute_record(&mut bytes, 2, 3, 2, -2, 2, 300, 2, 10, 2, 40, 2, 2);
    minute_record(&mut bytes, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 4, 9);

    let rows = decode_bytes(&bytes, NcdKind::Minute).expect("valid minute fixture");
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].local_ticks, EPOCH_TICKS);
    assert_eq!(rows[1].local_ticks, EPOCH_TICKS + 3 * TICKS_PER_MINUTE);
    assert_eq!(rows[2].local_ticks, EPOCH_TICKS + 4 * TICKS_PER_MINUTE);
    for (got, expected) in [rows[0].open, rows[0].high, rows[0].low, rows[0].close]
        .into_iter()
        .zip([100.0, 101.0, 99.5, 100.75])
    {
        close(got, expected);
    }
    assert_eq!(rows[0].volume, 7);
    // open: 400 ticks - 2 ticks, then +1 tick; volumes are code 2 => x100 and code 4 => x1000.
    close(rows[1].open, 99.5);
    close(rows[1].high, 174.5);
    close(rows[1].low, 97.0);
    close(rows[1].close, 107.0);
    assert_eq!(rows[1].volume, 200);
    close(rows[2].open, 99.75);
    assert_eq!(rows[2].volume, 9_000);
}

#[test]
fn minute_volume_codes_cover_supported_multipliers_and_widths() {
    let mut bytes = header(1.0, 10.0, EPOCH_TICKS);
    for (code, amount, expected) in [
        (1, 3, 3),
        (2, 4, 400),
        (3, 5, 2_500),
        (4, 6, 6_000),
        (5, 257, 257),
        (6, 65_537, 65_537),
    ] {
        minute_record(&mut bytes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, code, amount);
        let decoded = decode_bytes(&bytes, NcdKind::Minute).expect("supported minute volume");
        assert_eq!(decoded.last().unwrap().volume, expected);
        // Keep only the header for the next one-record sample.
        bytes.truncate(28);
    }
}

#[test]
fn tick_decodes_time_price_spread_and_volume_fields() {
    let mut bytes = header(0.5, 20.0, EPOCH_TICKS);
    // Price code 1 is embedded in b2; then exercise u8/u16/u32 price deltas and spread skips.
    tick_record(&mut bytes, 0, 0, 1, 3, 0, 1, 2);
    tick_record(&mut bytes, 1, 7, 2, -2, 6, 2, 3);
    tick_record(&mut bytes, 2, 9, 3, 4, 7, 5, 258);
    let rows = decode_bytes(&bytes, NcdKind::Tick).expect("valid tick fixture");
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].local_ticks, EPOCH_TICKS);
    assert_eq!(rows[1].local_ticks, EPOCH_TICKS + 7);
    assert_eq!(rows[2].local_ticks, EPOCH_TICKS + 16);
    for row in &rows {
        close(row.open, row.close);
        close(row.open, row.high);
        close(row.open, row.low);
    }
    close(rows[0].close, 21.5);
    close(rows[1].close, 20.5);
    close(rows[2].close, 22.5);
    assert_eq!(
        [rows[0].volume, rows[1].volume, rows[2].volume],
        [2, 300, 258]
    );
}

#[test]
fn tick_time_codes_include_seconds_and_large_ticks() {
    let mut bytes = header(1.0, 1.0, EPOCH_TICKS);
    for (code, delta) in [(3, 0x0102_0304), (4, 123_456_789), (5, 2), (6, 3)] {
        tick_record(&mut bytes, code, delta, 0, 0, 0, 1, 1);
    }
    let rows = decode_bytes(&bytes, NcdKind::Tick).expect("all tick time codes");
    assert_eq!(rows[0].local_ticks, EPOCH_TICKS + 0x0102_0304);
    assert_eq!(rows[1].local_ticks, EPOCH_TICKS + 0x0102_0304 + 123_456_789);
    assert_eq!(rows[2].local_ticks, rows[1].local_ticks + 20_000_000);
    assert_eq!(rows[3].local_ticks, rows[2].local_ticks + 30_000_000);
}

#[test]
fn rejects_bad_headers_codes_and_truncated_records() {
    assert!(decode_bytes(&[0; 27], NcdKind::Minute).is_err());
    let mut bad_version = header(0.25, 100.0, EPOCH_TICKS);
    bad_version[0] = 2;
    assert!(decode_bytes(&bad_version, NcdKind::Minute).is_err());

    let mut bad_minute_time = header(1.0, 1.0, EPOCH_TICKS);
    bad_minute_time.extend_from_slice(&[3, 0]);
    assert!(decode_bytes(&bad_minute_time, NcdKind::Minute).is_err());
    let mut bad_minute_open = header(1.0, 1.0, EPOCH_TICKS);
    bad_minute_open.extend_from_slice(&[0x0c, 0]);
    assert!(decode_bytes(&bad_minute_open, NcdKind::Minute).is_err());
    let mut bad_minute_volume = header(1.0, 1.0, EPOCH_TICKS);
    bad_minute_volume.extend_from_slice(&[0, 0]); // volume code 0 is unsupported
    assert!(decode_bytes(&bad_minute_volume, NcdKind::Minute).is_err());
    let mut bad_minute_low = header(1.0, 1.0, EPOCH_TICKS);
    bad_minute_low.extend_from_slice(&[0, 0xc0]); // low width code 3 is unsupported
    assert!(decode_bytes(&bad_minute_low, NcdKind::Minute).is_err());

    let mut bad_tick_time = header(1.0, 1.0, EPOCH_TICKS);
    bad_tick_time.extend_from_slice(&[7, 0x20]);
    assert!(decode_bytes(&bad_tick_time, NcdKind::Tick).is_err());
    let mut bad_tick_volume = header(1.0, 1.0, EPOCH_TICKS);
    bad_tick_volume.extend_from_slice(&[0, 0]);
    assert!(decode_bytes(&bad_tick_volume, NcdKind::Tick).is_err());

    // Each sample advertises a variable-length payload but ends before that payload is complete.
    for record in [
        vec![1, 0],    // minute time u8 + payload
        vec![4, 0],    // minute open u8 + payload
        vec![0, 0x10], // minute high u8 + payload
        vec![0, 0x40], // minute low u8 + payload
        vec![0, 1],    // minute close u8 + payload
        vec![0x20, 0], // minute volume u8 + payload
    ] {
        let mut fixture = header(1.0, 1.0, EPOCH_TICKS);
        fixture.extend_from_slice(&record);
        assert!(
            decode_bytes(&fixture, NcdKind::Minute).is_err(),
            "accepted {record:?}"
        );
    }
    for record in [
        vec![1, 0x20],    // tick time u8 + payload
        vec![0x80, 0x20], // tick price u8 + payload
        vec![0x30, 0x20], // tick spread byte + volume byte
        vec![0x38, 0x20], // two tick spread bytes + volume byte
    ] {
        let mut fixture = header(1.0, 1.0, EPOCH_TICKS);
        fixture.extend_from_slice(&record);
        assert!(
            decode_bytes(&fixture, NcdKind::Tick).is_err(),
            "accepted {record:?}"
        );
    }
}
