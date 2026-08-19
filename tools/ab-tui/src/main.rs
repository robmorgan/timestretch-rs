//! Blind A/B listening TUI — the interactive half of `scripts/ab.sh`.
//!
//! Steps through the conditions of a rendered blind set
//! (`target/ab/<name>/blind/<track>/<rate>/arm_{A..}.wav`), plays arms with
//! POSITION-SYNCED hot-switching (press another letter — or space on the
//! hovered arm — mid-play and hear the same passage through the other arm,
//! gapless), loops by default (`l` toggles), takes free-text notes
//! per arm plus an optional winner pick, and saves everything to a
//! machine-readable `results.json` — unblinded against `BLIND_KEY.json`
//! at save time, never before.
//!
//! Usage:
//!   ab-tui <set_dir> [--results <path>] [--no-unblind]
//!
//! `<set_dir>` is the `target/ab/<name>` directory `scripts/ab.sh render`
//! produced. On save the results path is printed as the LAST stdout line
//! (machine-parseable); exit code 0 after a save, 2 on usage errors.
//! Launching with an existing results file resumes the session.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use serde::{Deserialize, Serialize};

/// Seek step for the arrow keys, in seconds.
const SEEK_SECS: f64 = 5.0;
/// The reference arm's label (always last; it is not blind).
const SOURCE_LABEL: char = 'S';

// ---------------------------------------------------------------------------
// Set discovery and results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Condition {
    /// "track/rate" — the key used by BLIND_KEY.json and results.json.
    id: String,
    dir: PathBuf,
    /// Blind arm letters (A, B, ...), sorted; `S` (source) is implicit.
    letters: Vec<char>,
    notes: BTreeMap<char, String>,
    winner: Option<char>,
}

/// Walks `<set_dir>/blind/<track>/<rate>/arm_*.wav` into sorted conditions.
fn discover(set_dir: &Path) -> Result<Vec<Condition>, String> {
    let blind = set_dir.join("blind");
    let mut conditions = Vec::new();
    let tracks =
        std::fs::read_dir(&blind).map_err(|e| format!("no blind set at {blind:?}: {e}"))?;
    let mut track_dirs: Vec<PathBuf> = tracks
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    track_dirs.sort();
    for track_dir in track_dirs {
        let mut rate_dirs: Vec<PathBuf> = std::fs::read_dir(&track_dir)
            .map_err(|e| format!("unreadable {track_dir:?}: {e}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        rate_dirs.sort();
        for dir in rate_dirs {
            let mut letters: Vec<char> = std::fs::read_dir(&dir)
                .map_err(|e| format!("unreadable {dir:?}: {e}"))?
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    name.strip_prefix("arm_")
                        .and_then(|s| s.strip_suffix(".wav"))
                        .and_then(|s| s.chars().next())
                })
                .collect();
            letters.sort_unstable();
            if letters.is_empty() {
                continue;
            }
            let track = track_dir.file_name().unwrap().to_string_lossy();
            let rate = dir.file_name().unwrap().to_string_lossy();
            conditions.push(Condition {
                id: format!("{track}/{rate}"),
                dir: dir.clone(),
                letters,
                notes: BTreeMap::new(),
                winner: None,
            });
        }
    }
    if conditions.is_empty() {
        return Err(format!("no conditions found under {blind:?}"));
    }
    Ok(conditions)
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct ResultsFile {
    set: String,
    conditions: BTreeMap<String, ConditionResult>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct ConditionResult {
    #[serde(default)]
    notes: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    winner: Option<String>,
    /// Letter -> arm name, merged from BLIND_KEY.json at save time.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    arms: BTreeMap<String, String>,
}

fn results_from(conditions: &[Condition], set_name: &str, key: Option<&BlindKey>) -> ResultsFile {
    let mut out = ResultsFile {
        set: set_name.to_string(),
        conditions: BTreeMap::new(),
    };
    for c in conditions {
        if c.notes.is_empty() && c.winner.is_none() {
            continue;
        }
        let arms = key
            .and_then(|k| k.0.get(&c.id))
            .map(|m| m.iter().map(|(l, a)| (l.clone(), a.clone())).collect())
            .unwrap_or_default();
        out.conditions.insert(
            c.id.clone(),
            ConditionResult {
                notes: c
                    .notes
                    .iter()
                    .map(|(l, n)| (l.to_string(), n.clone()))
                    .collect(),
                winner: c.winner.map(|w| w.to_string()),
                arms,
            },
        );
    }
    out
}

fn resume_into(conditions: &mut [Condition], prior: &ResultsFile) {
    for c in conditions.iter_mut() {
        if let Some(r) = prior.conditions.get(&c.id) {
            c.notes = r
                .notes
                .iter()
                .filter_map(|(l, n)| l.chars().next().map(|ch| (ch, n.clone())))
                .collect();
            c.winner = r.winner.as_deref().and_then(|w| w.chars().next());
        }
    }
}

/// The sealed key: condition id -> letter -> arm name. Read ONLY at save.
struct BlindKey(BTreeMap<String, BTreeMap<String, String>>);

fn read_key(set_dir: &Path) -> Option<BlindKey> {
    let raw = std::fs::read_to_string(set_dir.join("BLIND_KEY.json")).ok()?;
    serde_json::from_str(&raw).ok().map(BlindKey)
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

struct PlayerShared {
    /// Playback cursor in frames (shared with the audio callback).
    cursor: AtomicUsize,
    /// Index into the arm buffer list.
    active: AtomicUsize,
    playing: AtomicBool,
    looping: AtomicBool,
}

/// One condition's loaded audio and its output stream. Rebuilt per
/// condition (sample rate or channel count may differ between sets).
struct Player {
    _stream: Stream,
    shared: Arc<PlayerShared>,
    frames: usize,
    sample_rate: u32,
    /// Ordered labels matching the buffer list (blind letters then `S`).
    labels: Vec<char>,
}

impl Player {
    fn load(condition: &Condition, looping: bool) -> Result<Self, String> {
        let mut labels: Vec<char> = condition.letters.clone();
        let mut paths: Vec<PathBuf> = labels
            .iter()
            .map(|l| condition.dir.join(format!("arm_{l}.wav")))
            .collect();
        let source = condition.dir.join("source.wav");
        if source.exists() {
            labels.push(SOURCE_LABEL);
            paths.push(source);
        }

        let mut buffers: Vec<Arc<Vec<f32>>> = Vec::with_capacity(paths.len());
        let mut sample_rate = 0u32;
        let mut channels = 0usize;
        for path in &paths {
            let buf = timestretch::io::read_wav_file(
                path.to_str().ok_or_else(|| format!("bad path {path:?}"))?,
            )
            .map_err(|e| format!("read {path:?}: {e:?}"))?;
            if sample_rate == 0 {
                sample_rate = buf.sample_rate;
                channels = buf.channels.count();
            } else if buf.sample_rate != sample_rate || buf.channels.count() != channels {
                return Err(format!("arm format mismatch at {path:?}"));
            }
            buffers.push(Arc::new(buf.data));
        }
        // Arms are rendered to a common length; hold the invariant anyway.
        let frames = buffers
            .iter()
            .map(|b| b.len() / channels.max(1))
            .min()
            .unwrap_or(0);

        let shared = Arc::new(PlayerShared {
            cursor: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            playing: AtomicBool::new(false),
            looping: AtomicBool::new(looping),
        });

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no audio output device".to_string())?;
        let config = StreamConfig {
            channels: channels as u16,
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let cb_shared = shared.clone();
        let cb_buffers = buffers;
        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    data.fill(0.0);
                    if !cb_shared.playing.load(Ordering::Acquire) {
                        return;
                    }
                    let arm = &cb_buffers[cb_shared.active.load(Ordering::Acquire)];
                    let mut cursor = cb_shared.cursor.load(Ordering::Acquire);
                    let looping = cb_shared.looping.load(Ordering::Acquire);
                    for frame in data.chunks_exact_mut(channels) {
                        if cursor >= frames {
                            if looping {
                                cursor = 0;
                            } else {
                                cb_shared.playing.store(false, Ordering::Release);
                                break;
                            }
                        }
                        let base = cursor * channels;
                        frame.copy_from_slice(&arm[base..base + channels]);
                        cursor += 1;
                    }
                    cb_shared.cursor.store(cursor, Ordering::Release);
                },
                move |err| eprintln!("audio stream error: {err}"),
                None,
            )
            .map_err(|e| format!("open output stream: {e}"))?;
        stream.play().map_err(|e| format!("start stream: {e}"))?;

        Ok(Self {
            _stream: stream,
            shared,
            frames,
            sample_rate,
            labels,
        })
    }

    fn select(&self, label: char) {
        if let Some(idx) = self.labels.iter().position(|&l| l == label) {
            self.shared.active.store(idx, Ordering::Release);
            if self.shared.cursor.load(Ordering::Acquire) >= self.frames {
                self.shared.cursor.store(0, Ordering::Release);
            }
            self.shared.playing.store(true, Ordering::Release);
        }
    }

    fn active_label(&self) -> char {
        self.labels[self
            .shared
            .active
            .load(Ordering::Acquire)
            .min(self.labels.len() - 1)]
    }

    fn toggle_play(&self) {
        let playing = self.shared.playing.load(Ordering::Acquire);
        if !playing && self.shared.cursor.load(Ordering::Acquire) >= self.frames {
            self.shared.cursor.store(0, Ordering::Release);
        }
        self.shared.playing.store(!playing, Ordering::Release);
    }

    fn seek(&self, delta_secs: f64) {
        let delta = (delta_secs * self.sample_rate as f64) as i64;
        let cursor = self.shared.cursor.load(Ordering::Acquire) as i64;
        let next = (cursor + delta).clamp(0, self.frames.saturating_sub(1) as i64);
        self.shared.cursor.store(next as usize, Ordering::Release);
    }

    fn restart(&self) {
        self.shared.cursor.store(0, Ordering::Release);
    }

    fn set_loop(&self, on: bool) {
        self.shared.looping.store(on, Ordering::Release);
    }

    fn position_secs(&self) -> f64 {
        self.shared.cursor.load(Ordering::Acquire) as f64 / self.sample_rate as f64
    }

    fn duration_secs(&self) -> f64 {
        self.frames as f64 / self.sample_rate as f64
    }

    fn is_playing(&self) -> bool {
        self.shared.playing.load(Ordering::Acquire)
    }

    fn is_looping(&self) -> bool {
        self.shared.looping.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pane {
    Conditions,
    Arms,
}

/// What space does, given where the cursor is versus what is playing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpaceAction {
    /// Hot-switch to a different arm (and play it).
    Switch(char),
    /// Pause/resume whatever is already active.
    Toggle,
}

/// Space auditions the hovered arm; on the arm already playing it pauses.
fn space_action(pane: Pane, focused: Option<char>, active: char) -> SpaceAction {
    match focused {
        Some(label) if pane == Pane::Arms && label != active => SpaceAction::Switch(label),
        _ => SpaceAction::Toggle,
    }
}

/// An in-progress note edit: the arm being annotated, the text, and a
/// cursor held as a byte index that always sits on a char boundary.
#[derive(Debug, Clone)]
struct NoteEdit {
    label: char,
    text: String,
    cursor: usize,
}

impl NoteEdit {
    fn new(label: char, text: String) -> Self {
        let cursor = text.len();
        Self {
            label,
            text,
            cursor,
        }
    }

    fn insert(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    fn backspace(&mut self) {
        if let Some(c) = self.text[..self.cursor].chars().next_back() {
            self.cursor -= c.len_utf8();
            self.text.remove(self.cursor);
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.text.len() {
            self.text.remove(self.cursor);
        }
    }

    fn left(&mut self) {
        if let Some(c) = self.text[..self.cursor].chars().next_back() {
            self.cursor -= c.len_utf8();
        }
    }

    fn right(&mut self) {
        if let Some(c) = self.text[self.cursor..].chars().next() {
            self.cursor += c.len_utf8();
        }
    }

    fn home(&mut self) {
        self.cursor = 0;
    }

    fn end(&mut self) {
        self.cursor = self.text.len();
    }

    /// The text with the caret glyph drawn at the cursor.
    fn rendered(&self) -> String {
        format!(
            "{}\u{258f}{}",
            &self.text[..self.cursor],
            &self.text[self.cursor..]
        )
    }
}

struct App {
    set_dir: PathBuf,
    set_name: String,
    results_path: PathBuf,
    unblind: bool,
    conditions: Vec<Condition>,
    current: usize,
    player: Option<Player>,
    player_error: Option<String>,
    /// Which pane the ↑/↓ cursor lives in (Tab toggles).
    pane: Pane,
    /// Arm whose note row is focused (↑/↓ in the Arms pane).
    focus: usize,
    /// In-progress note edit.
    editing: Option<NoteEdit>,
    /// Loop preference, owned by the app so it survives condition changes
    /// (the per-condition `Player` mirrors it).
    looping: bool,
    dirty: bool,
    confirm_quit: bool,
    status: String,
}

impl App {
    fn condition(&self) -> &Condition {
        &self.conditions[self.current]
    }

    fn arm_labels(&self) -> Vec<char> {
        match &self.player {
            Some(p) => p.labels.clone(),
            None => {
                let mut l = self.condition().letters.clone();
                l.push(SOURCE_LABEL);
                l
            }
        }
    }

    fn load_player(&mut self) {
        match Player::load(self.condition(), self.looping) {
            Ok(p) => {
                self.player = Some(p);
                self.player_error = None;
            }
            Err(e) => {
                self.player = None;
                self.player_error = Some(e);
            }
        }
    }

    fn goto(&mut self, idx: usize) {
        if idx < self.conditions.len() && idx != self.current {
            self.current = idx;
            self.focus = 0;
            self.editing = None;
            self.load_player();
        }
    }

    fn save(&mut self) -> Result<(), String> {
        let key = if self.unblind {
            read_key(&self.set_dir)
        } else {
            None
        };
        let results = results_from(&self.conditions, &self.set_name, key.as_ref());
        let json = serde_json::to_string_pretty(&results).map_err(|e| e.to_string())?;
        std::fs::write(&self.results_path, json + "\n").map_err(|e| e.to_string())?;
        self.dirty = false;
        self.status = format!("saved {}", self.results_path.display());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

fn draw(frame: &mut ratatui::Frame, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(30), Constraint::Min(20)])
        .split(frame.area());

    // Condition list.
    let items: Vec<ListItem> = app
        .conditions
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let marks = format!(
                "{}{}",
                if c.notes.is_empty() { " " } else { "✓" },
                if c.winner.is_some() { "★" } else { " " },
            );
            let style = if i == app.current {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(format!("{marks} {}", c.id))).style(style)
        })
        .collect();
    let pane_border = |focused: bool| {
        if focused {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default()
        }
    };
    frame.render_widget(
        List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(pane_border(app.pane == Pane::Conditions))
                .title(format!(" {} ", app.set_name)),
        ),
        cols[0],
    );

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(4),
            Constraint::Length(3),
        ])
        .split(cols[1]);

    // Transport.
    let transport = match (&app.player, &app.player_error) {
        (Some(p), _) => {
            let pos = p.position_secs();
            let dur = p.duration_secs();
            format!(
                "{} {:5.1}s / {:5.1}s   arm {}   loop {}",
                if p.is_playing() { "▶" } else { "⏸" },
                pos,
                dur,
                p.active_label(),
                if p.is_looping() { "on" } else { "off" },
            )
        }
        (None, Some(e)) => format!("audio unavailable: {e}"),
        (None, None) => "loading…".to_string(),
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(app.condition().id.clone(), Style::default().fg(Color::Cyan)),
            Span::raw("   "),
            Span::raw(transport),
        ])),
        rows[0],
    );

    // Arms.
    let labels = app.arm_labels();
    let arm_items: Vec<ListItem> = labels
        .iter()
        .enumerate()
        .map(|(i, &label)| {
            let playing = app
                .player
                .as_ref()
                .map(|p| p.is_playing() && p.active_label() == label)
                .unwrap_or(false);
            let winner = app.condition().winner == Some(label);
            let note = if let Some(edit) = &app.editing {
                if edit.label == label {
                    edit.rendered()
                } else {
                    app.condition()
                        .notes
                        .get(&label)
                        .cloned()
                        .unwrap_or_default()
                }
            } else {
                app.condition()
                    .notes
                    .get(&label)
                    .cloned()
                    .unwrap_or_default()
            };
            let mut spans = vec![
                Span::styled(
                    format!(" {} ", label),
                    if playing {
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().add_modifier(Modifier::BOLD)
                    },
                ),
                Span::raw(if playing { "▶ " } else { "  " }),
                Span::raw(if winner { "★ " } else { "  " }),
                Span::raw(note),
            ];
            if label == SOURCE_LABEL {
                spans.insert(
                    1,
                    Span::styled("(source) ", Style::default().fg(Color::DarkGray)),
                );
            }
            let style = if i == app.focus && app.pane == Pane::Arms {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(spans)).style(style)
        })
        .collect();
    frame.render_widget(
        List::new(arm_items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(pane_border(app.pane == Pane::Arms))
                .title(" arms "),
        ),
        rows[1],
    );

    // Footer.
    let footer = if app.confirm_quit {
        "unsaved notes — press q again to discard, Ctrl-S to save first".to_string()
    } else if app.editing.is_some() {
        "editing note — ←/→ move  Home/End  Enter commit  Esc cancel".to_string()
    } else {
        format!(
            "Tab pane  ↑/↓ move  Enter note  a-{}/s play arm  space play/switch  l loop  \
             ←/→ seek  0 restart  w winner  W clear  Ctrl-S save  q quit   {}",
            labels
                .iter()
                .rfind(|&&l| l != SOURCE_LABEL)
                .map(|l| l.to_ascii_lowercase())
                .unwrap_or('e'),
            app.status,
        )
    };
    frame.render_widget(
        Paragraph::new(footer).block(Block::default().borders(Borders::TOP)),
        rows[2],
    );
}

fn run_tui(mut app: App) -> Result<(), String> {
    let mut terminal = ratatui::init();
    app.load_player();
    loop {
        terminal
            .draw(|frame| draw(frame, &app))
            .map_err(|e| e.to_string())?;
        if !event::poll(Duration::from_millis(50)).map_err(|e| e.to_string())? {
            continue;
        }
        let Event::Key(key) = event::read().map_err(|e| e.to_string())? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        // Note-edit mode swallows everything.
        if let Some(edit) = &mut app.editing {
            match key.code {
                KeyCode::Enter => {
                    let (label, text) = (edit.label, edit.text.clone());
                    if text.is_empty() {
                        app.conditions[app.current].notes.remove(&label);
                    } else {
                        app.conditions[app.current].notes.insert(label, text);
                    }
                    app.editing = None;
                    app.dirty = true;
                }
                KeyCode::Esc => app.editing = None,
                KeyCode::Backspace => edit.backspace(),
                KeyCode::Delete => edit.delete(),
                KeyCode::Left => edit.left(),
                KeyCode::Right => edit.right(),
                KeyCode::Home => edit.home(),
                KeyCode::End => edit.end(),
                // Modified chords (Ctrl-S and friends) are not note text.
                KeyCode::Char(c)
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    edit.insert(c)
                }
                _ => {}
            }
            continue;
        }

        let labels = app.arm_labels();
        match (key.code, key.modifiers) {
            (KeyCode::Char('s'), m) if m.contains(KeyModifiers::CONTROL) => {
                app.save()?;
            }
            (KeyCode::Char('q'), _) => {
                if app.dirty && !app.confirm_quit {
                    app.confirm_quit = true;
                } else {
                    break;
                }
            }
            (KeyCode::Char(' '), _) => {
                if let Some(p) = &app.player {
                    let focused = labels.get(app.focus.min(labels.len() - 1)).copied();
                    match space_action(app.pane, focused, p.active_label()) {
                        SpaceAction::Switch(label) => p.select(label),
                        SpaceAction::Toggle => p.toggle_play(),
                    }
                }
            }
            (KeyCode::Char('l'), _) => {
                app.looping = !app.looping;
                if let Some(p) = &app.player {
                    p.set_loop(app.looping);
                }
            }
            (KeyCode::Left, _) => {
                if let Some(p) = &app.player {
                    p.seek(-SEEK_SECS);
                }
            }
            (KeyCode::Right, _) => {
                if let Some(p) = &app.player {
                    p.seek(SEEK_SECS);
                }
            }
            (KeyCode::Char('0'), _) => {
                if let Some(p) = &app.player {
                    p.restart();
                }
            }
            (KeyCode::Char('n'), _) => {
                let next = (app.current + 1).min(app.conditions.len() - 1);
                app.goto(next);
            }
            (KeyCode::Char('p'), _) => {
                let prev = app.current.saturating_sub(1);
                app.goto(prev);
            }
            (KeyCode::Down, _) => match app.pane {
                Pane::Conditions => {
                    let next = (app.current + 1).min(app.conditions.len() - 1);
                    app.goto(next);
                }
                Pane::Arms => {
                    app.focus = (app.focus + 1).min(labels.len() - 1);
                }
            },
            (KeyCode::Up, _) => match app.pane {
                Pane::Conditions => {
                    let prev = app.current.saturating_sub(1);
                    app.goto(prev);
                }
                Pane::Arms => {
                    app.focus = app.focus.saturating_sub(1);
                }
            },
            (KeyCode::Tab, _) => {
                app.pane = match app.pane {
                    Pane::Conditions => Pane::Arms,
                    Pane::Arms => Pane::Conditions,
                };
            }
            (KeyCode::Enter, _) => match app.pane {
                Pane::Conditions => {
                    // The condition is already loaded by goto(); Enter
                    // moves the cursor into it.
                    app.pane = Pane::Arms;
                }
                Pane::Arms => {
                    let label = labels[app.focus.min(labels.len() - 1)];
                    if label != SOURCE_LABEL {
                        let existing = app
                            .condition()
                            .notes
                            .get(&label)
                            .cloned()
                            .unwrap_or_default();
                        app.editing = Some(NoteEdit::new(label, existing));
                    }
                }
            },
            (KeyCode::Char('w'), m) if m.contains(KeyModifiers::SHIFT) => {
                app.conditions[app.current].winner = None;
                app.dirty = true;
            }
            (KeyCode::Char('w'), _) => {
                let label = labels[app.focus.min(labels.len() - 1)];
                if label != SOURCE_LABEL {
                    app.conditions[app.current].winner = Some(label);
                    app.dirty = true;
                }
            }
            (KeyCode::Char(c), _) => {
                // Arm selection: lowercase letters map to blind arms, `s`
                // to the source reference.
                let target = c.to_ascii_uppercase();
                if labels.contains(&target)
                    && let Some(p) = &app.player
                {
                    p.select(target);
                    if let Some(idx) = labels.iter().position(|&l| l == target) {
                        app.focus = idx;
                        app.pane = Pane::Arms;
                    }
                }
            }
            _ => {}
        }
        if !matches!(key.code, KeyCode::Char('q')) {
            app.confirm_quit = false;
        }
    }
    ratatui::restore();
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!("usage: ab-tui <set_dir> [--results <path>] [--no-unblind]");
    std::process::exit(2);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut set_dir: Option<PathBuf> = None;
    let mut results_path: Option<PathBuf> = None;
    let mut unblind = true;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--results" => {
                results_path = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            "--no-unblind" => unblind = false,
            "--help" | "-h" => usage(),
            _ if set_dir.is_none() => set_dir = Some(PathBuf::from(arg)),
            _ => usage(),
        }
    }
    let Some(set_dir) = set_dir else { usage() };
    let set_name = set_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "ab".to_string());
    let results_path = results_path.unwrap_or_else(|| set_dir.join("results.json"));

    let mut conditions = match discover(&set_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Ok(raw) = std::fs::read_to_string(&results_path)
        && let Ok(prior) = serde_json::from_str::<ResultsFile>(&raw)
    {
        resume_into(&mut conditions, &prior);
    }

    let app = App {
        set_dir,
        set_name,
        results_path: results_path.clone(),
        unblind,
        conditions,
        current: 0,
        player: None,
        player_error: None,
        pane: Pane::Conditions,
        focus: 0,
        editing: None,
        looping: true,
        dirty: false,
        confirm_quit: false,
        status: String::new(),
    };

    match run_tui(app) {
        Ok(_) => {
            // The results path is the LAST stdout line, machine-parseable.
            let mut stdout = std::io::stdout();
            let _ = writeln!(stdout, "{}", results_path.display());
        }
        Err(e) => {
            ratatui::restore();
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (pure parts only — no terminal, no audio device)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn scaffold(dir: &Path, conds: &[(&str, &str, &[char])]) {
        for (track, rate, letters) in conds {
            let d = dir.join("blind").join(track).join(rate);
            std::fs::create_dir_all(&d).unwrap();
            for l in *letters {
                std::fs::write(d.join(format!("arm_{l}.wav")), b"x").unwrap();
            }
            std::fs::write(d.join("source.wav"), b"x").unwrap();
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ab_tui_test_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn discovery_sorts_conditions_and_letters() {
        let dir = temp_dir("disc");
        scaffold(
            &dir,
            &[
                ("trackb", "-8pct", &['B', 'A']),
                ("tracka", "+8pct", &['A', 'B', 'C']),
            ],
        );
        let conds = discover(&dir).unwrap();
        assert_eq!(conds.len(), 2);
        assert_eq!(conds[0].id, "tracka/+8pct");
        assert_eq!(conds[0].letters, vec!['A', 'B', 'C']);
        assert_eq!(conds[1].id, "trackb/-8pct");
        assert_eq!(conds[1].letters, vec!['A', 'B']);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_set_errors() {
        let dir = temp_dir("empty");
        std::fs::create_dir_all(dir.join("blind")).unwrap();
        assert!(discover(&dir).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn results_merge_unblind_and_resume_round_trip() {
        let dir = temp_dir("res");
        scaffold(&dir, &[("t", "-8pct", &['A', 'B'])]);
        let mut conds = discover(&dir).unwrap();
        conds[0].notes.insert('A', "clean".into());
        conds[0].notes.insert('B', "roboty".into());
        conds[0].winner = Some('A');

        let key: BTreeMap<String, BTreeMap<String, String>> =
            serde_json::from_str(r#"{"t/-8pct": {"A": "current", "B": "base"}}"#).unwrap();
        let results = results_from(&conds, "myset", Some(&BlindKey(key)));
        let json = serde_json::to_string(&results).unwrap();
        let parsed: ResultsFile = serde_json::from_str(&json).unwrap();
        let r = &parsed.conditions["t/-8pct"];
        assert_eq!(r.notes["A"], "clean");
        assert_eq!(r.winner.as_deref(), Some("A"));
        assert_eq!(r.arms["B"], "base");

        // Resume restores notes + winner onto a fresh discovery.
        let mut fresh = discover(&dir).unwrap();
        resume_into(&mut fresh, &parsed);
        assert_eq!(fresh[0].notes[&'B'], "roboty");
        assert_eq!(fresh[0].winner, Some('A'));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn note_edit_cursor_moves_and_edits_mid_string() {
        let mut e = NoteEdit::new('A', "abcd".into());
        assert_eq!(e.cursor, 4);
        e.left();
        e.left();
        e.insert('X');
        assert_eq!(e.text, "abXcd");
        assert_eq!(e.rendered(), "abX\u{258f}cd");
        e.backspace();
        assert_eq!(e.text, "abcd");
        e.delete();
        assert_eq!(e.text, "abd");
        e.home();
        assert_eq!(e.cursor, 0);
        e.end();
        assert_eq!(e.cursor, e.text.len());
    }

    #[test]
    fn note_edit_clamps_at_both_ends() {
        let mut e = NoteEdit::new('A', "ab".into());
        e.right();
        assert_eq!(e.cursor, 2);
        e.backspace();
        e.backspace();
        e.backspace();
        assert_eq!(e.text, "");
        assert_eq!(e.cursor, 0);
        e.left();
        assert_eq!(e.cursor, 0);
        e.delete();
        assert_eq!(e.text, "");
    }

    #[test]
    fn note_edit_is_char_boundary_safe() {
        let mut e = NoteEdit::new('A', "café".into());
        e.left();
        assert_eq!(e.cursor, 3);
        e.insert('é');
        assert_eq!(e.text, "caféé");
        assert_eq!(e.rendered(), "café\u{258f}é");
        e.backspace();
        assert_eq!(e.text, "café");
        e.left();
        e.delete();
        assert_eq!(e.text, "caé");
    }

    #[test]
    fn space_switches_to_hovered_arm_else_toggles() {
        assert_eq!(
            space_action(Pane::Arms, Some('B'), 'A'),
            SpaceAction::Switch('B')
        );
        assert_eq!(
            space_action(Pane::Arms, Some('A'), 'A'),
            SpaceAction::Toggle
        );
        assert_eq!(
            space_action(Pane::Conditions, Some('B'), 'A'),
            SpaceAction::Toggle
        );
        assert_eq!(space_action(Pane::Arms, None, 'A'), SpaceAction::Toggle);
    }

    #[test]
    fn unnoted_conditions_are_omitted_from_results() {
        let dir = temp_dir("omit");
        scaffold(
            &dir,
            &[("t", "-8pct", &['A', 'B']), ("t", "+8pct", &['A', 'B'])],
        );
        let mut conds = discover(&dir).unwrap();
        conds[0].notes.insert('A', "x".into());
        let results = results_from(&conds, "s", None);
        assert_eq!(results.conditions.len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
