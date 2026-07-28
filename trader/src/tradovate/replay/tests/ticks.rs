use super::super::ticks::parse_tick_line;

#[test]
fn parse_tick_line_reads_ninjatrader_last_format() {
    let tick =
        parse_tick_line("20260324 040000 1800000;6603;6603;6603.25;1").expect("tick should parse");
    assert_eq!(tick.last, 6603.0);
    assert!(tick.ts_ns > 0);
}
