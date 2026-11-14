//! NinjaTrader 8 NCD minute and tick reader.
//!
//! The format is undocumented. Record decoding is based on the MIT-licensed
//! ninjatrader-to-parquet and NinjaTraderNCDFiles projects; see
//! `tools/licenses/ninjatrader-ncd-MIT.txt` for their notices.
//! Timestamps are .NET ticks in the NinjaTrader installation's local timezone;
//! interpretation is deliberately left to the caller.
use anyhow::{Context, Result, bail, ensure};
use std::path::Path;

const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RECORDS: usize = 2_000_000;
const MINUTE_TICKS: i64 = 600_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NcdKind {
    Minute,
    Tick,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NcdRecord {
    pub local_ticks: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
}

pub(crate) fn decode_file(path: &Path, kind: NcdKind) -> Result<Vec<NcdRecord>> {
    use std::io::Read;
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut data = Vec::new();
    file.take(MAX_FILE_BYTES + 1).read_to_end(&mut data)?;
    ensure!(
        data.len() as u64 <= MAX_FILE_BYTES,
        "NCD file exceeds 64 MiB limit"
    );
    decode_bytes(&data, kind).with_context(|| format!("decode {}", path.display()))
}

struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}
impl Cursor<'_> {
    fn read(&mut self, n: usize) -> Result<u64> {
        let end = self.offset.checked_add(n).context("NCD offset overflow")?;
        let bytes = self
            .data
            .get(self.offset..end)
            .with_context(|| format!("truncated NCD record at byte {}", self.offset))?;
        self.offset = end;
        Ok(bytes.iter().fold(0u64, |v, b| (v << 8) | u64::from(*b)))
    }
}

fn decode_bytes(data: &[u8], kind: NcdKind) -> Result<Vec<NcdRecord>> {
    ensure!(data.len() >= 28, "NCD header is shorter than 28 bytes");
    let version = i32::from_le_bytes(data[0..4].try_into()?);
    ensure!(version == 1, "unsupported NCD version {version}");
    let tick_size = f64::from_le_bytes(data[4..12].try_into()?);
    let first_price = f64::from_le_bytes(data[12..20].try_into()?);
    ensure!(
        tick_size.is_finite() && tick_size > 0.0 && first_price.is_finite(),
        "invalid NCD price header"
    );
    let first_ticks = first_price / tick_size;
    ensure!(
        first_ticks.is_finite()
            && first_ticks.abs() < (i64::MAX as f64)
            && (first_ticks - first_ticks.round()).abs() < 1e-5,
        "NCD starting price is not a whole tick"
    );
    let mut price = first_ticks.round() as i64;
    let mut time = i64::from_le_bytes(data[20..28].try_into()?);
    let mut input = Cursor { data, offset: 28 };
    let mut rows = Vec::new();
    while input.offset < data.len() {
        ensure!(
            rows.len() < MAX_RECORDS,
            "NCD file exceeds {MAX_RECORDS} record limit"
        );
        let start = input.offset;
        let result = (|| -> Result<NcdRecord> {
            let flags = input.read(1)? as u8;
            let flags2 = input.read(1)? as u8;
            let (open, high, low, close, volume) = match kind {
                NcdKind::Minute => {
                    let advance = match flags & 3 {
                        0 => 1,
                        1 => input.read(1)?,
                        2 => input.read(2)?,
                        _ => bail!("unsupported minute time code"),
                    };
                    let delta = i64::try_from(advance)?
                        .checked_mul(MINUTE_TICKS)
                        .context("minute time overflow")?;
                    time = time.checked_add(delta).context("timestamp overflow")?;
                    let change = match (flags >> 2) & 3 {
                        0 => 0,
                        1 => input.read(1)? as i64 - 128,
                        2 => input.read(2)? as i64 - 32768,
                        _ => bail!("unsupported minute open code"),
                    };
                    price = price.checked_add(change).context("price overflow")?;
                    ensure!(
                        flags2 & 0x0c == 0 && flags2 >> 6 != 3,
                        "unsupported minute OHLC code"
                    );
                    let widths = [0, 1, 2, 4];
                    let high_delta =
                        i64::try_from(input.read(widths[((flags2 >> 4) & 3) as usize])?)?;
                    let low_delta = i64::try_from(input.read(widths[(flags2 >> 6) as usize])?)?;
                    let close_delta = i64::try_from(input.read(widths[(flags2 & 3) as usize])?)?;
                    let high = price.checked_add(high_delta).context("high overflow")?;
                    let low = price.checked_sub(low_delta).context("low overflow")?;
                    let close = low.checked_add(close_delta).context("close overflow")?;
                    ensure!(
                        close >= low && close <= high,
                        "close lies outside bar range"
                    );
                    let (width, scale) = match flags >> 5 {
                        1 => (1, 1),
                        2 => (1, 100),
                        3 => (1, 500),
                        4 => (1, 1000),
                        5 => (2, 1),
                        6 => (4, 1),
                        _ => bail!("unsupported minute volume code"),
                    };
                    (price, high, low, close, input.read(width)? * scale)
                }
                NcdKind::Tick => {
                    let delta = match flags & 7 {
                        0 => 0,
                        1 => input.read(1)?,
                        2 => input.read(2)?,
                        3 => input.read(4)?,
                        4 => input.read(8)?,
                        5 => input.read(1)? * 10_000_000,
                        6 => input.read(2)? * 10_000_000,
                        _ => bail!("unsupported tick time code"),
                    };
                    time = time
                        .checked_add(i64::try_from(delta)?)
                        .context("timestamp overflow")?;
                    let change = match flags >> 6 {
                        0 => 0,
                        1 => i64::from(flags2 & 31) - 16,
                        2 => input.read(1)? as i64 - 128,
                        3 => input.read(4)? as i64 - 2_147_483_648,
                        _ => unreachable!(),
                    };
                    price = price.checked_add(change).context("price overflow")?;
                    let spread = (flags >> 3) & 7;
                    if spread >= 6 {
                        input.read((spread - 5) as usize)?;
                    }
                    let (width, scale) = match flags2 >> 5 {
                        1 => (1, 1),
                        2 => (1, 100),
                        5 => (2, 1),
                        _ => bail!("unsupported tick volume code"),
                    };
                    (price, price, price, price, input.read(width)? * scale)
                }
            };
            let convert = |v: i64| v as f64 * tick_size;
            let row = NcdRecord {
                local_ticks: time,
                open: convert(open),
                high: convert(high),
                low: convert(low),
                close: convert(close),
                volume,
            };
            ensure!(
                [row.open, row.high, row.low, row.close]
                    .iter()
                    .all(|v| v.is_finite()),
                "price is not finite"
            );
            Ok(row)
        })();
        rows.push(result.with_context(|| format!("NCD record at byte {start}"))?);
    }
    Ok(rows)
}

#[cfg(test)]
#[path = "ninjatrader_ncd_tests.rs"]
mod tests;
