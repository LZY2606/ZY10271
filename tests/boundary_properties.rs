//! Bounded property and differential test suite.
//!
//! The corpus tests in `main.rs` cover a rich set of textual inputs; this
//! suite instead generates systematic boundary inputs algorithmically:
//!
//! * every valid date in 1600..=2400, with parse -> format -> parse roundtrips
//! * an independent civil-date algorithm (Howard Hinnant's `days_from_civil` /
//!   `civil_from_days`) as the oracle for unix timestamp <-> date conversion —
//!   it deliberately does not share any code with speedate's date helpers
//! * per-value enumeration around the second/millisecond inference watershed
//!   (`|t| == 20_000_000_000`), including negative values
//! * directed coverage of ±24h timezone offsets, month ends, leap centuries,
//!   fraction truncate/error policies and NUL bytes embedded in byte input
//!
//! All generators use a fixed seed; nothing here reads the timezone database,
//! the system locale or the wall clock. Every assertion reports the minimal
//! triggering input and the configuration profile in use.

use speedate::{
    Date, DateConfig, DateTime, DateTimeConfig, Duration, MicrosecondsPrecisionOverflowBehavior, ParseError, Time,
    TimeConfig, TimestampUnit,
};

/// Fixed seed: failures must be reproducible without any environment dependence.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// speedate infers milliseconds when `abs(timestamp) > MS_WATERSHED`
/// (mirrors `src/date.rs`, restated here so the suite pins the contract).
const MS_WATERSHED: i64 = 20_000_000_000;

/// splitmix64 — small deterministic PRNG, independent of any library.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// ---------------------------------------------------------------------------
// Independent reference algorithms (the differential oracle)
// ---------------------------------------------------------------------------

fn ref_is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn ref_days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if ref_is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => panic!("bad month {month}"),
    }
}

/// Howard Hinnant's `days_from_civil`: days from 1970-01-01 to y-m-d
/// in the proleptic Gregorian calendar.
fn ref_days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400; // [0, 399]
    let mp = (m as i64 + 9) % 12; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Howard Hinnant's `civil_from_days`, the exact inverse of `ref_days_from_civil`.
fn ref_civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Split unix seconds into a civil date and time-of-day, using floored
/// division so negative timestamps round towards the previous day.
fn ref_unix_to_civil_hms(secs: i64) -> ((i64, u32, u32), u32, u32, u32) {
    let days = secs.div_euclid(86400);
    let sod = secs.rem_euclid(86400);
    (
        ref_civil_from_days(days),
        (sod / 3600) as u32,
        ((sod % 3600) / 60) as u32,
        (sod % 60) as u32,
    )
}

/// Independent restatement of the second/millisecond inference contract.
fn ref_split_infer(v: i64) -> (i64, u32) {
    if v.abs() <= MS_WATERSHED {
        (v, 0)
    } else {
        (v.div_euclid(1_000), (v.rem_euclid(1_000) * 1_000) as u32)
    }
}

// ---------------------------------------------------------------------------
// Failure context: minimal input + profile on every assertion
// ---------------------------------------------------------------------------

struct Ctx {
    input: String,
    profile: &'static str,
}

impl Ctx {
    fn new(input: impl Into<String>, profile: &'static str) -> Self {
        Self {
            input: input.into(),
            profile,
        }
    }
}

macro_rules! assert_eq_ctx {
    ($ctx:expr, $actual:expr, $expected:expr) => {
        assert_eq!(
            $actual, $expected,
            "seed={SEED:#x} profile={} minimal_input={:?}",
            $ctx.profile, $ctx.input
        )
    };
}

fn unwrap_ctx<T, E: core::fmt::Debug>(result: Result<T, E>, ctx: &Ctx) -> T {
    result.unwrap_or_else(|e| {
        panic!(
            "seed={SEED:#x} profile={} minimal_input={:?}: unexpected error {e:?}",
            ctx.profile, ctx.input
        )
    })
}

fn expect_err_ctx<T: core::fmt::Debug>(result: Result<T, ParseError>, expected: ParseError, ctx: &Ctx) {
    match result {
        Err(e) => assert_eq_ctx!(ctx, e, expected),
        Ok(v) => panic!(
            "seed={SEED:#x} profile={} minimal_input={:?}: expected {expected:?}, got ok {v:?}",
            ctx.profile, ctx.input
        ),
    }
}

// ---------------------------------------------------------------------------
// Oracle self-check: anchor the independent algorithm to known constants
// ---------------------------------------------------------------------------

#[test]
fn oracle_self_check_anchors() {
    assert_eq!(ref_days_from_civil(1970, 1, 1), 0);
    assert_eq!(ref_days_from_civil(1600, 1, 1), -135_140);
    assert_eq!(ref_days_from_civil(0, 1, 1), -719_528);
    assert_eq!(ref_days_from_civil(2000, 2, 29), 11_016);
    assert_eq!(ref_days_from_civil(2401, 1, 1), 157_420);
    // the oracle must be its own inverse across the whole supported range
    let mut days = ref_days_from_civil(0, 1, 1);
    let end = ref_days_from_civil(2401, 1, 1);
    while days < end {
        let (y, m, d) = ref_civil_from_days(days);
        assert_eq!(
            ref_days_from_civil(y, m, d),
            days,
            "oracle inverse failed for days={days}"
        );
        days += 997;
    }
}

// ---------------------------------------------------------------------------
// Date: exhaustive roundtrip + differential timestamp, 1600..=2400
// ---------------------------------------------------------------------------

#[test]
fn date_parse_format_parse_roundtrip_1600_to_2400() {
    let profile = "Date::parse_str_rfc3339";
    for year in 1600..=2400i64 {
        for month in 1..=12u32 {
            let dim = ref_days_in_month(year, month);
            for day in 1..=dim {
                let input = format!("{year:04}-{month:02}-{day:02}");
                let ctx = Ctx::new(&input, profile);
                let parsed = unwrap_ctx(Date::parse_str_rfc3339(&input), &ctx);
                assert_eq_ctx!(
                    ctx,
                    (parsed.year as i64, parsed.month as u32, parsed.day as u32),
                    (year, month, day)
                );
                // format must be the canonical fixed-width form
                let formatted = parsed.to_string();
                assert_eq_ctx!(ctx, formatted, input);
                // parse(format(parse(x))) == parse(x)
                let reparsed = unwrap_ctx(Date::parse_str_rfc3339(&formatted), &ctx);
                assert_eq_ctx!(ctx, reparsed, parsed);
                // differential: unix timestamp vs the independent civil algorithm
                assert_eq_ctx!(ctx, parsed.timestamp(), ref_days_from_civil(year, month, day) * 86400);
            }
        }
    }
}

/// Day 0 and day-after-month-end must be rejected for every month of every
/// year in range — this is the directed month-end and leap-century coverage
/// (e.g. `2100-02-29` appears here as an invalid input).
#[test]
fn date_invalid_day_boundaries_rejected_1600_to_2400() {
    let profile = "Date::parse_str_rfc3339";
    for year in 1600..=2400i64 {
        for month in 1..=12u32 {
            let dim = ref_days_in_month(year, month);
            for bad_day in [0u32, dim + 1] {
                let input = format!("{year:04}-{month:02}-{bad_day:02}");
                let ctx = Ctx::new(&input, profile);
                expect_err_ctx(Date::parse_str_rfc3339(&input), ParseError::OutOfRangeDay, &ctx);
            }
        }
    }
    for bad_month in [0u32, 13] {
        let input = format!("2021-{bad_month:02}-01");
        let ctx = Ctx::new(&input, profile);
        expect_err_ctx(Date::parse_str_rfc3339(&input), ParseError::OutOfRangeMonth, &ctx);
    }
}

/// Directed leap-century semantics: years divisible by 100 are not leap
/// unless divisible by 400. Asserted as speedate's own documented semantics.
#[test]
fn date_leap_century_directed() {
    let profile = "Date::parse_str_rfc3339";
    for (year, is_leap) in [
        (1600, true),
        (1700, false),
        (1800, false),
        (1900, false),
        (2000, true),
        (2100, false),
        (2200, false),
        (2300, false),
        (2400, true),
    ] {
        let input = format!("{year}-02-29");
        let ctx = Ctx::new(&input, profile);
        if is_leap {
            let d = unwrap_ctx(Date::parse_str_rfc3339(&input), &ctx);
            assert_eq_ctx!(ctx, (d.year, d.month, d.day), (year as u16, 2, 29));
        } else {
            expect_err_ctx(Date::parse_str_rfc3339(&input), ParseError::OutOfRangeDay, &ctx);
        }
    }
}

// ---------------------------------------------------------------------------
// Time: roundtrip on both sides of the day boundary, offsets and fractions
// ---------------------------------------------------------------------------

fn truncate_config() -> TimeConfig {
    TimeConfig {
        microseconds_precision_overflow_behavior: MicrosecondsPrecisionOverflowBehavior::Truncate,
        unix_timestamp_offset: None,
    }
}

/// Expected microseconds for a fraction string under the truncate policy.
fn ref_fraction_micros(fraction: &str) -> u32 {
    let mut micros = 0u32;
    for (i, b) in fraction.bytes().enumerate() {
        if i < 6 {
            micros = micros * 10 + (b - b'0') as u32;
        }
    }
    if fraction.len() < 6 {
        micros *= 10u32.pow(6 - fraction.len() as u32);
    }
    micros
}

/// Expected tz offset in seconds for the offset forms generated below.
fn ref_offset_seconds(offset: &str) -> Option<i32> {
    match offset {
        "" => None,
        "Z" => Some(0),
        _ => {
            let sign = if offset.starts_with('-') { -1 } else { 1 };
            let hours: i32 = offset[1..3].parse().unwrap();
            let minutes: i32 = offset[4..6].parse().unwrap();
            Some(sign * (hours * 3600 + minutes * 60))
        }
    }
}

fn check_time_roundtrip(input: &str, h: u32, m: u32, s: u32, fraction: &str, offset: &str, profile: &'static str) {
    let ctx = Ctx::new(input, profile);
    let parsed = unwrap_ctx(
        Time::parse_bytes_with_config(input.as_bytes(), &truncate_config()),
        &ctx,
    );
    assert_eq_ctx!(ctx, parsed.hour as u32, h);
    assert_eq_ctx!(ctx, parsed.minute as u32, m);
    assert_eq_ctx!(ctx, parsed.second as u32, s);
    assert_eq_ctx!(ctx, parsed.microsecond, ref_fraction_micros(fraction));
    assert_eq_ctx!(ctx, parsed.tz_offset, ref_offset_seconds(offset));
    assert_eq_ctx!(ctx, parsed.total_seconds(), h * 3600 + m * 60 + s);
    // parse(format(parse(x))) == parse(x)
    let formatted = parsed.to_string();
    let reparsed = unwrap_ctx(
        Time::parse_bytes_with_config(formatted.as_bytes(), &truncate_config()),
        &Ctx::new(&formatted, profile),
    );
    assert_eq_ctx!(ctx, reparsed, parsed);
}

#[test]
fn time_parse_format_parse_roundtrip_day_boundaries() {
    let profile = "Time::parse_bytes_with_config(truncate)";
    // times hugging both sides of the day boundary, fractions up to
    // nanosecond precision (9 digits, truncated to microseconds)
    let fractions = ["", "5", "000001", "999999", "123456789"];
    let offsets = ["", "Z", "+00:00", "-00:00", "+23:59", "-23:59"];
    for h in [0u32, 1, 22, 23] {
        for m in [0u32, 1, 58, 59] {
            for s in [0u32, 1, 58, 59] {
                for (fi, fraction) in fractions.iter().enumerate() {
                    // rotate offsets over the fraction axis to keep the
                    // cross-product bounded while covering every offset form
                    let offset = offsets[(h as usize + m as usize + s as usize + fi) % offsets.len()];
                    let frac_part = if fraction.is_empty() {
                        String::new()
                    } else {
                        format!(".{fraction}")
                    };
                    let input = format!("{h:02}:{m:02}:{s:02}{frac_part}{offset}");
                    check_time_roundtrip(&input, h, m, s, fraction, offset, profile);
                }
            }
        }
    }
    // pseudo-random times with random minute-aligned offsets within ±24h
    let mut rng = Rng(SEED ^ 0x71);
    for _ in 0..5_000 {
        let h = rng.below(24) as u32;
        let m = rng.below(60) as u32;
        let s = rng.below(60) as u32;
        let digits = 1 + rng.below(9) as usize;
        let mut fraction = String::new();
        for _ in 0..digits {
            fraction.push((b'0' + rng.below(10) as u8) as char);
        }
        let offset_minutes = rng.below(2 * 24 * 60 - 1) as i32 - (24 * 60 - 1);
        let offset = match rng.below(3) {
            0 => String::new(),
            1 => "Z".to_string(),
            _ => {
                let sign = if offset_minutes < 0 { '-' } else { '+' };
                format!(
                    "{sign}{:02}:{:02}",
                    offset_minutes.abs() / 60,
                    offset_minutes.abs() % 60
                )
            }
        };
        let input = format!("{h:02}:{m:02}:{s:02}.{fraction}{offset}");
        check_time_roundtrip(&input, h, m, s, &fraction, &offset, profile);
    }
}

// ---------------------------------------------------------------------------
// DateTime: roundtrip + differential timestamp over the safe range
// ---------------------------------------------------------------------------

fn check_datetime(input: &str, profile: &'static str) {
    let ctx = Ctx::new(input, profile);
    let parsed = unwrap_ctx(DateTime::parse_str(input), &ctx);
    // parse(format(parse(x))) == parse(x)
    let formatted = parsed.to_string();
    let reparsed = unwrap_ctx(DateTime::parse_str(&formatted), &Ctx::new(&formatted, profile));
    assert_eq_ctx!(ctx, reparsed, parsed);
    // differential: absolute timestamp vs the independent civil algorithm
    let (y, mo, d) = (
        parsed.date.year as i64,
        parsed.date.month as u32,
        parsed.date.day as u32,
    );
    let hms = parsed.time.total_seconds() as i64;
    let offset = parsed.time.tz_offset.unwrap_or(0) as i64;
    let expected = ref_days_from_civil(y, mo, d) * 86400 + hms - offset;
    assert_eq_ctx!(ctx, parsed.timestamp_tz(), expected);
    // self-consistency of the millisecond view
    assert_eq_ctx!(
        ctx,
        parsed.timestamp_ms(),
        parsed.timestamp() * 1000 + (parsed.time.microsecond / 1000) as i64
    );
}

#[test]
fn datetime_roundtrip_and_timestamp_differential() {
    let profile = "DateTime::parse_str(default)";
    // directed: range extremes, leap-century neighbourhood, day boundaries
    let directed_dates = [
        "1600-01-01",
        "1600-02-29",
        "1700-02-28",
        "1900-02-28",
        "2000-02-29",
        "2100-02-28",
        "2100-03-01",
        "2400-02-29",
        "2400-12-31",
    ];
    let directed_times = ["00:00:00", "23:59:59", "00:00:00.000001", "23:59:59.999999"];
    let directed_offsets = ["", "Z", "+23:59", "-23:59", "+05:30", "-08:00"];
    for date in directed_dates {
        for time in directed_times {
            for offset in directed_offsets {
                check_datetime(&format!("{date}T{time}{offset}"), profile);
            }
        }
    }
    // pseudo-random datetimes, uniformly over 1600..=2400
    let mut rng = Rng(SEED ^ 0xD7);
    for _ in 0..20_000 {
        let year = 1600 + rng.below(801) as i64;
        let month = 1 + rng.below(12) as u32;
        let day = 1 + rng.below(ref_days_in_month(year, month) as u64) as u32;
        let h = rng.below(24);
        let m = rng.below(60);
        let s = rng.below(60);
        let micros = rng.below(1_000_000);
        let frac_part = if micros == 0 {
            String::new()
        } else {
            format!(".{micros:06}")
        };
        let offset_minutes = rng.below(2 * 24 * 60 - 1) as i32 - (24 * 60 - 1);
        let offset = match rng.below(3) {
            0 => String::new(),
            1 => "Z".to_string(),
            _ => {
                let sign = if offset_minutes < 0 { '-' } else { '+' };
                format!(
                    "{sign}{:02}:{:02}",
                    offset_minutes.abs() / 60,
                    offset_minutes.abs() % 60
                )
            }
        };
        let input = format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}{frac_part}{offset}");
        check_datetime(&input, profile);
    }
}

// ---------------------------------------------------------------------------
// Timestamp -> DateTime/Date: differential against the independent algorithm
// ---------------------------------------------------------------------------

fn check_timestamp_seconds(t: i64, config: &DateTimeConfig, date_config: &DateConfig, profile: &'static str) {
    let ctx = Ctx::new(t.to_string(), profile);
    let dt = unwrap_ctx(DateTime::from_timestamp_with_config(t, 0, config), &ctx);
    let ((y, mo, d), h, mi, s) = ref_unix_to_civil_hms(t);
    assert_eq_ctx!(
        ctx,
        (
            dt.date.year as i64,
            dt.date.month as u32,
            dt.date.day as u32,
            dt.time.hour as u32,
            dt.time.minute as u32,
            dt.time.second as u32,
        ),
        (y, mo, d, h, mi, s)
    );
    assert_eq_ctx!(ctx, dt.time.microsecond, 0);
    assert_eq_ctx!(ctx, dt.timestamp(), t);
    // the Date view must agree on the civil date
    let date = unwrap_ctx(Date::from_timestamp(t, false, date_config), &ctx);
    assert_eq_ctx!(ctx, (date.year as i64, date.month as u32, date.day as u32), (y, mo, d));
}

#[test]
fn timestamp_to_datetime_differential_safe_range() {
    let profile = "DateTime::from_timestamp_with_config(s)";
    let dt_config = DateTimeConfig {
        timestamp_unit: TimestampUnit::Second,
        time_config: TimeConfig::default(),
    };
    let date_config = DateConfig {
        timestamp_unit: TimestampUnit::Second,
    };
    let ts_min = ref_days_from_civil(1600, 1, 1) * 86400;
    let ts_max = ref_days_from_civil(2401, 1, 1) * 86400 - 1;
    // every day boundary in 1600..=2400
    let mut t = ts_min;
    while t <= ts_max {
        check_timestamp_seconds(t, &dt_config, &date_config, profile);
        t += 86400;
    }
    // dense sweep across the epoch, exercising negative timestamps
    for t in -100_000..=100_000 {
        check_timestamp_seconds(t, &dt_config, &date_config, profile);
    }
    // pseudo-random seconds over the whole safe range
    let mut rng = Rng(SEED ^ 0x5E);
    let span = (ts_max - ts_min) as u64;
    for _ in 0..50_000 {
        let t = ts_min + rng.below(span) as i64;
        check_timestamp_seconds(t, &dt_config, &date_config, profile);
    }
}

// ---------------------------------------------------------------------------
// Second/millisecond inference watershed: per-value enumeration, both signs
// ---------------------------------------------------------------------------

#[test]
fn timestamp_unit_inference_watershed_enumeration() {
    let profile = "TimestampUnit::Infer";
    let dt_config = DateTimeConfig::default();
    let date_config = DateConfig::default();
    for delta in -2_000..=2_000i64 {
        for base in [MS_WATERSHED, -MS_WATERSHED] {
            let v = base + delta;
            let ctx = Ctx::new(v.to_string(), profile);
            let (exp_secs, exp_micros) = ref_split_infer(v);
            let ((y, mo, d), h, mi, s) = ref_unix_to_civil_hms(exp_secs);
            let dt = unwrap_ctx(DateTime::from_timestamp_with_config(v, 0, &dt_config), &ctx);
            assert_eq_ctx!(
                ctx,
                (
                    dt.date.year as i64,
                    dt.date.month as u32,
                    dt.date.day as u32,
                    dt.time.hour as u32,
                    dt.time.minute as u32,
                    dt.time.second as u32,
                    dt.time.microsecond,
                ),
                (y, mo, d, h, mi, s, exp_micros)
            );
            let date = unwrap_ctx(Date::from_timestamp(v, false, &date_config), &ctx);
            assert_eq_ctx!(ctx, (date.year as i64, date.month as u32, date.day as u32), (y, mo, d));
        }
    }
}

/// Negative millisecond timestamps must use floored division: `-1ms` is
/// `1969-12-31T23:59:59.999`, not a truncation towards zero. These directed
/// cases pin the sign handling of the unit split.
#[test]
fn timestamp_negative_unit_division_directed() {
    let ms_config = DateTimeConfig {
        timestamp_unit: TimestampUnit::Millisecond,
        time_config: TimeConfig::default(),
    };
    let s_config = DateTimeConfig {
        timestamp_unit: TimestampUnit::Second,
        time_config: TimeConfig::default(),
    };
    let profile = "DateTime::from_timestamp_with_config(ms|s)";
    for (v, config, expected) in [
        (-1, &ms_config, "1969-12-31T23:59:59.999000"),
        (-999, &ms_config, "1969-12-31T23:59:59.001000"),
        (-1_000, &ms_config, "1969-12-31T23:59:59"),
        (-1_001, &ms_config, "1969-12-31T23:59:58.999000"),
        (-86_400_000, &ms_config, "1969-12-31T00:00:00"),
        (-86_400_001, &ms_config, "1969-12-30T23:59:59.999000"),
        (-1, &s_config, "1969-12-31T23:59:59"),
        (-86_400, &s_config, "1969-12-31T00:00:00"),
        (-86_401, &s_config, "1969-12-30T23:59:59"),
    ] {
        let ctx = Ctx::new(v.to_string(), profile);
        let dt = unwrap_ctx(DateTime::from_timestamp_with_config(v, 0, config), &ctx);
        assert_eq_ctx!(ctx, dt.to_string(), expected);
    }
    // the Date view agrees, and exactness is enforced only when requested
    let ms_date_config = DateConfig {
        timestamp_unit: TimestampUnit::Millisecond,
    };
    let ctx = Ctx::new("-1", "Date::from_timestamp(ms)");
    let d = unwrap_ctx(Date::from_timestamp(-1, false, &ms_date_config), &ctx);
    assert_eq_ctx!(ctx, d.to_string(), "1969-12-31");
    expect_err_ctx(
        Date::from_timestamp(-1, true, &ms_date_config),
        ParseError::DateNotExact,
        &ctx,
    );
}

// ---------------------------------------------------------------------------
// Directed: ±24h timezone offset boundary
// ---------------------------------------------------------------------------

#[test]
fn tz_offset_24h_boundary_directed() {
    let profile = "Time::parse_str";
    for (input, expected_offset) in [
        ("12:00:00+23:59", 86_340),
        ("12:00:00-23:59", -86_340),
        ("12:00:00+00:00", 0),
        ("12:00:00-00:00", 0),
        ("12:00:00Z", 0),
        ("12:00:00+2359", 86_340),
        ("12:00:00-0001", -60),
    ] {
        let ctx = Ctx::new(input, profile);
        let t = unwrap_ctx(Time::parse_str(input), &ctx);
        assert_eq_ctx!(ctx, t.tz_offset, Some(expected_offset));
    }
    // |offset| == 24h is out of range (the bound is strict, matching python)
    for input in [
        "12:00:00+24:00",
        "12:00:00-24:00",
        "12:00:00+24:01",
        "12:00:00-24:01",
        "12:00:00+99:00",
    ] {
        let ctx = Ctx::new(input, profile);
        expect_err_ctx(Time::parse_str(input), ParseError::OutOfRangeTz, &ctx);
    }
    for input in ["12:00:00+01:60", "12:00:00-00:60"] {
        let ctx = Ctx::new(input, profile);
        expect_err_ctx(Time::parse_str(input), ParseError::OutOfRangeTzMinute, &ctx);
    }
    // datetime forms go through the same offset parser
    let ctx = Ctx::new("2020-01-01T12:00:00+24:00", "DateTime::parse_str");
    expect_err_ctx(
        DateTime::parse_str("2020-01-01T12:00:00+24:00"),
        ParseError::OutOfRangeTz,
        &ctx,
    );
    // programmatic offset adjustment enforces the same strict <24h bound
    let t = Time::parse_str("12:00:00Z").unwrap();
    let profile = "Time::with_timezone_offset";
    for (offset, ok) in [(86_399, true), (-86_399, true), (86_400, false), (-86_400, false)] {
        let ctx = Ctx::new(offset.to_string(), profile);
        match t.with_timezone_offset(Some(offset)) {
            Ok(adjusted) => {
                assert!(ok, "seed={SEED:#x} profile={profile} minimal_input={offset}");
                assert_eq_ctx!(ctx, adjusted.tz_offset, Some(offset));
            }
            Err(e) => {
                assert!(!ok, "seed={SEED:#x} profile={profile} minimal_input={offset}");
                assert_eq_ctx!(ctx, e, ParseError::OutOfRangeTz);
            }
        }
    }
    let ctx = Ctx::new("86400", "Time::in_timezone");
    expect_err_ctx(t.in_timezone(86_400), ParseError::OutOfRangeTz, &ctx);
}

// ---------------------------------------------------------------------------
// Directed: fraction overflow policy (truncate vs error)
// ---------------------------------------------------------------------------

#[test]
fn fraction_overflow_truncate_vs_error_directed() {
    let error_config = TimeConfig::default(); // Error is the default policy
    let truncate_config = TimeConfig {
        microseconds_precision_overflow_behavior: MicrosecondsPrecisionOverflowBehavior::Truncate,
        unix_timestamp_offset: None,
    };
    let profile = "TimeConfig{microseconds_precision_overflow_behavior}";
    // 7+ fraction digits: rejected under Error, truncated (never rounded) under Truncate
    let ctx = Ctx::new("12:00:00.1234567", profile);
    expect_err_ctx(
        Time::parse_bytes_with_config(b"12:00:00.1234567", &error_config),
        ParseError::SecondFractionTooLong,
        &ctx,
    );
    let t = unwrap_ctx(
        Time::parse_bytes_with_config(b"12:00:00.1234567", &truncate_config),
        &ctx,
    );
    assert_eq_ctx!(ctx, t.microsecond, 123_456);
    assert_eq_ctx!(ctx, t.to_string(), "12:00:00.123456");
    let ctx = Ctx::new("12:00:00.999999999", profile);
    let t = unwrap_ctx(
        Time::parse_bytes_with_config(b"12:00:00.999999999", &truncate_config),
        &ctx,
    );
    assert_eq_ctx!(ctx, t.microsecond, 999_999);
    let ctx = Ctx::new("12:00:00.0000009", profile);
    let t = unwrap_ctx(
        Time::parse_bytes_with_config(b"12:00:00.0000009", &truncate_config),
        &ctx,
    );
    assert_eq_ctx!(ctx, t.microsecond, 0);
    // exactly 6 digits is accepted by both policies
    let ctx = Ctx::new("12:00:00.123456", profile);
    for config in [&error_config, &truncate_config] {
        let t = unwrap_ctx(Time::parse_bytes_with_config(b"12:00:00.123456", config), &ctx);
        assert_eq_ctx!(ctx, t.microsecond, 123_456);
    }
    // a fraction separator with no digits is an error under both policies
    for config in [&error_config, &truncate_config] {
        let ctx = Ctx::new("12:00:00.", profile);
        expect_err_ctx(
            Time::parse_bytes_with_config(b"12:00:00.", config),
            ParseError::SecondFractionMissing,
            &ctx,
        );
    }
    // datetime unix-timestamp float path: 6 digits max for seconds, 3 for milliseconds
    let dt_error = DateTimeConfig::default();
    let dt_truncate = DateTimeConfig {
        timestamp_unit: TimestampUnit::Infer,
        time_config: truncate_config,
    };
    let profile = "DateTimeConfig{overflow}(unix float)";
    let ctx = Ctx::new("1654646404.1234567", profile);
    expect_err_ctx(
        DateTime::parse_str_with_config("1654646404.1234567", &dt_error),
        ParseError::SecondFractionTooLong,
        &ctx,
    );
    unwrap_ctx(
        DateTime::parse_str_with_config("1654646404.1234567", &dt_truncate),
        &ctx,
    );
    let ctx = Ctx::new("1654646404123.4567", profile);
    expect_err_ctx(
        DateTime::parse_str_with_config("1654646404123.4567", &dt_error),
        ParseError::MillisecondFractionTooLong,
        &ctx,
    );
    unwrap_ctx(
        DateTime::parse_str_with_config("1654646404123.4567", &dt_truncate),
        &ctx,
    );
}

// ---------------------------------------------------------------------------
// Directed: leap seconds and 24:00 are extensions speedate rejects
// ---------------------------------------------------------------------------

#[test]
fn leap_second_and_2400_extension_semantics() {
    // Asserted directly as speedate's own semantics — no stdlib oracle is
    // involved, since these inputs are valid under ISO 8601 extensions.
    let profile = "Time::parse_str";
    for input in ["23:59:60", "23:59:60Z", "23:59:60.5"] {
        let ctx = Ctx::new(input, profile);
        expect_err_ctx(Time::parse_str(input), ParseError::OutOfRangeSecond, &ctx);
    }
    for input in ["24:00", "24:00:00", "24:00:00.000000"] {
        let ctx = Ctx::new(input, profile);
        expect_err_ctx(Time::parse_str(input), ParseError::OutOfRangeHour, &ctx);
    }
    let profile = "DateTime::parse_str";
    // 2016-12-31T23:59:60Z was a real leap second
    let ctx = Ctx::new("2016-12-31T23:59:60Z", profile);
    expect_err_ctx(
        DateTime::parse_str("2016-12-31T23:59:60Z"),
        ParseError::OutOfRangeSecond,
        &ctx,
    );
    let ctx = Ctx::new("2020-01-01T24:00:00", profile);
    expect_err_ctx(
        DateTime::parse_str("2020-01-01T24:00:00"),
        ParseError::OutOfRangeHour,
        &ctx,
    );
}

// ---------------------------------------------------------------------------
// Directed: NUL bytes embedded in byte input
// ---------------------------------------------------------------------------

#[test]
fn nul_bytes_embedded_in_input() {
    // NUL is not special-cased anywhere: it must behave like any other
    // invalid byte, at every position where it can appear.
    let profile = "Date::parse_bytes";
    for (input, expected) in [
        (&b"2020-01-01\0"[..], ParseError::ExtraCharacters),
        (b"2020\0-01-01", ParseError::InvalidCharDateSep),
        (b"2020-01-0\0", ParseError::InvalidCharDay),
        (b"\0", ParseError::TooShort),
        // numeric timestamp form: NUL makes it a non-number, and the
        // RFC3339 fallback error is surfaced
        (b"1577836800\0", ParseError::InvalidCharDateSep),
    ] {
        let ctx = Ctx::new(format!("{input:?}"), profile);
        expect_err_ctx(Date::parse_bytes(input), expected, &ctx);
    }
    let profile = "Time::parse_bytes";
    for (input, expected) in [
        (&b"12:00:00\0"[..], ParseError::InvalidCharTzSign),
        (b"12:00:00Z\0", ParseError::ExtraCharacters),
        (b"12\0:00", ParseError::InvalidCharTimeSep),
        (b"12:00\0", ParseError::InvalidCharTzSign),
        (b"12:00:00\0+01:00", ParseError::InvalidCharTzSign),
        (b"12:00:00.12\0", ParseError::InvalidCharTzSign),
    ] {
        let ctx = Ctx::new(format!("{input:?}"), profile);
        expect_err_ctx(Time::parse_bytes(input), expected, &ctx);
    }
    let profile = "DateTime::parse_bytes";
    for (input, expected) in [
        (&b"2020-01-01\0T12:00:00"[..], ParseError::InvalidCharDateTimeSep),
        (b"2020-01-01T12:00:00\0", ParseError::InvalidCharTzSign),
        (b"2020-01-01T12:00:00Z\0", ParseError::ExtraCharacters),
    ] {
        let ctx = Ctx::new(format!("{input:?}"), profile);
        expect_err_ctx(DateTime::parse_bytes(input), expected, &ctx);
    }
}

// ---------------------------------------------------------------------------
// Duration: bounded parse -> format -> parse roundtrip + signed totals
// ---------------------------------------------------------------------------

fn check_duration_roundtrip(positive: bool, day: u32, second: u32, microsecond: u32, profile: &'static str) {
    let ctx = Ctx::new(format!("sign={positive} {day}d {second}s {microsecond}us"), profile);
    let d = unwrap_ctx(Duration::new(positive, day, second, microsecond), &ctx);
    let formatted = d.to_string();
    let reparsed = unwrap_ctx(Duration::parse_str(&formatted), &Ctx::new(&formatted, profile));
    assert_eq_ctx!(ctx, reparsed, d);
    // differential: signed totals against plain arithmetic
    let sign: i64 = if positive { 1 } else { -1 };
    assert_eq_ctx!(
        ctx,
        d.signed_total_seconds(),
        sign * (day as i64 * 86400 + second as i64)
    );
    assert_eq_ctx!(ctx, d.signed_microseconds(), (sign * microsecond as i64) as i32);
}

#[test]
fn duration_parse_format_parse_roundtrip() {
    let profile = "Duration::parse_str";
    // directed boundaries, including the 999_999_999-day limit and negative zero
    for &(positive, day, second, microsecond) in &[
        (true, 0, 0, 0),
        (false, 0, 0, 0),
        (false, 0, 0, 1),
        (true, 0, 86_399, 999_999),
        (false, 999_999_999, 86_399, 999_999),
        (true, 365, 0, 0),
        (false, 366, 1, 1),
    ] {
        check_duration_roundtrip(positive, day, second, microsecond, profile);
    }
    let mut rng = Rng(SEED ^ 0xD0);
    for _ in 0..20_000 {
        let day = rng.below(1_000_000_000) as u32;
        let second = rng.below(86_400) as u32;
        let microsecond = rng.below(1_000_000) as u32;
        let positive = rng.below(2) == 0;
        check_duration_roundtrip(positive, day, second, microsecond, profile);
    }
    // one day beyond the limit is rejected
    let ctx = Ctx::new("1000000000d", "Duration::new");
    expect_err_ctx(
        Duration::new(true, 1_000_000_000, 0, 0),
        ParseError::DurationDaysTooLarge,
        &ctx,
    );
}
