"""
Morse EME Signal Simulator by DL3WDG
==================================
Sends a user-typed message as Morse code, modulating the tone generator
from the EME Signal Simulator (DL3WDG).  The same Rayleigh Lorentzian
fading and AWGN noise pipeline is applied so the output sounds like a
weak-signal CW transmission over a faded channel.

Signal chain:
    MorseKeyer  ->  ToneGen (complex IQ)  ->  FadeEnvelope  ->
    imag() + AWGN  ->  sounddevice output

No delay line is used (unlike the EME simulator).
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import configparser
import os
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 12000
CHUNK         = 512
BW_REF        = 2500.0
OUT_PEAK      = 0.5
LORENTZ_B     = 6.0
LORENTZ_CUT   = 3.0
XFADE_LEN     = 512          # crossfade samples between fading blocks (~43 ms)
DEFAULT_FREQ  = 700.0        # Hz  — classic CW pitch
DEFAULT_WPM   = 20           # words per minute
DEFAULT_SNR   = -10.0        # dB
DEFAULT_SPREAD = 2.0         # Hz  — Doppler spread
DEFAULT_LEVEL  = 0.8         # output level multiplier 0.0–1.0
DEFAULT_FILTER_BW = 500.0    # Hz  — CW filter bandwidth (0 = bypass)
DEFAULT_MYCALL    = "DL3WDG" # operator callsign

# ── Callsign list — edit this list to add/remove stations ─────────────────────
CALLSIGN_LIST = [
    "DL3WDG",
    "DL4KGC",
    "DG2YCB",
    "DL0ROI",
    "W3SZ",
    "KA1GT",
    "G3LTF",
    "G4CCH",
    "JA1TTT",
    "RW3BP",
    "AB7DX",
    "VK7MO",
    "DJ7FJ",
]

def pick_dx_call(mycall: str) -> str:
    """Pick a random callsign from CALLSIGN_LIST, excluding mycall."""
    import random
    candidates = [c for c in CALLSIGN_LIST if c.upper() != mycall.strip().upper()]
    if not candidates:
        candidates = CALLSIGN_LIST   # fallback if list only has one entry
    return random.choice(candidates)

# ──────────────────────────────────────────────────────────────────────────────
#  Settings persistence  (AppData\MorseEME\settings.ini)
# ──────────────────────────────────────────────────────────────────────────────
def _ini_path() -> str:
    base = os.environ.get('APPDATA') or os.path.expanduser('~')
    folder = os.path.join(base, 'MorseEME')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, 'settings.ini')

def load_settings() -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(_ini_path())
    s = cfg['settings'] if 'settings' in cfg else {}
    return {
        'message': s.get('message', 'CQ CQ DE DL3WDG'),
        'freq':    float(s.get('freq',   DEFAULT_FREQ)),
        'wpm':     float(s.get('wpm',    DEFAULT_WPM)),
        'snr':     float(s.get('snr',    DEFAULT_SNR)),
        'spread':  float(s.get('spread', DEFAULT_SPREAD)),
        'level':     float(s.get('level',     DEFAULT_LEVEL)),
        'filter_bw': float(s.get('filter_bw', DEFAULT_FILTER_BW)),
        'mycall':    s.get('mycall',    DEFAULT_MYCALL),
        'mode':      s.get('mode',      'trainer'),
        'device':    s.get('device', ''),
    }

def save_settings(d: dict):
    cfg = configparser.ConfigParser()
    cfg['settings'] = {k: str(v) for k, v in d.items()}
    try:
        with open(_ini_path(), 'w') as f:
            cfg.write(f)
    except Exception as e:
        print(f"[save_settings] {e}")

# Morse timing: PARIS standard — 1 dit = 1200 / wpm  ms
# dit=1, dah=3, inter-element=1, inter-char=3, inter-word=7  (in dits)
MORSE_TABLE = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',
    'E': '.',     'F': '..-.',  'G': '--.',   'H': '....',
    'I': '..',    'J': '.---',  'K': '-.-',   'L': '.-..',
    'M': '--',    'N': '-.',    'O': '---',   'P': '.--.',
    'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',
    'Y': '-.--',  'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', '/': '-..-.',
    '-': '-....-', '(': '-.--.',  ')': '-.--.-', '=': '-...-',
    '+': '.-.-.',  '@': '.--.-.',
}


# ──────────────────────────────────────────────────────────────────────────────
#  Fading envelope  (verbatim from EME simulator, minus the delay line)
# ──────────────────────────────────────────────────────────────────────────────
def _fade_block_len(fspread: float) -> int:
    MIN_SAMPLES = 10 * 8192
    if fspread <= 0:
        return MIN_SAMPLES
    samples = int(20.0 * SAMPLE_RATE / fspread)
    return max(MIN_SAMPLES, min(samples, SAMPLE_RATE * 60))


def generate_fade_envelope(fspread: float) -> np.ndarray:
    nfft = _fade_block_len(fspread)
    if fspread <= 0.0:
        return np.ones(nfft, dtype=np.complex64)
    nh      = nfft // 2
    df      = SAMPLE_RATE / nfft
    cspread = np.zeros(nfft, dtype=np.complex64)
    for i in range(1, nh):
        x = LORENTZ_B * (i * df) / fspread
        if x < LORENTZ_CUT:
            a    = float(np.sqrt(max(0.0, 1.111 / (1.0 + x * x) - 0.1)))
            phi1 = 2.0 * np.pi * np.random.rand()
            cspread[i]        = a * np.exp(1j * phi1)
            phi2 = 2.0 * np.pi * np.random.rand()
            cspread[nfft - i] = a * np.exp(1j * phi2)
    env = np.fft.ifft(cspread).astype(np.complex64)
    pwr = float(np.mean(np.abs(env) ** 2))
    if pwr > 0.0:
        env *= np.float32(1.0 / np.sqrt(pwr))
    return env


class FadeEnvelope:
    def __init__(self):
        self._fspread   = -1.0
        self._env       = np.ones(SAMPLE_RATE, dtype=np.complex64)
        self._next_env  = None
        self._pos       = 0
        self._xfade_buf = None
        self._xfade_pos = 0

    def set_fspread(self, fspread: float):
        if fspread != self._fspread:
            self._fspread   = fspread
            self._env       = generate_fade_envelope(fspread)
            self._next_env  = None
            self._pos       = 0
            self._xfade_buf = None
            self._xfade_pos = 0

    def _build_xfade(self):
        self._next_env  = generate_fade_envelope(self._fspread)
        n               = XFADE_LEN
        t               = np.linspace(0.0, 1.0, n, dtype=np.float32)
        ramp_out        = np.cos(0.5 * np.pi * t) ** 2
        ramp_in         = 1.0 - ramp_out
        tail = self._env[-n:]
        head = self._next_env[:n]
        self._xfade_buf = (ramp_out * tail + ramp_in * head).astype(np.complex64)
        self._xfade_pos = 0

    def get(self, n: int) -> np.ndarray:
        out    = np.empty(n, dtype=np.complex64)
        filled = 0
        while filled < n:
            if self._xfade_buf is not None:
                avail = len(self._xfade_buf) - self._xfade_pos
                take  = min(avail, n - filled)
                out[filled:filled + take] = (
                    self._xfade_buf[self._xfade_pos:self._xfade_pos + take])
                filled          += take
                self._xfade_pos += take
                if self._xfade_pos >= len(self._xfade_buf):
                    self._env       = self._next_env
                    self._next_env  = None
                    self._xfade_buf = None
                    self._xfade_pos = 0
                    self._pos       = XFADE_LEN
                continue
            stop  = len(self._env) - XFADE_LEN
            avail = max(0, stop - self._pos)
            take  = min(avail, n - filled)
            if take > 0:
                out[filled:filled + take] = (
                    self._env[self._pos:self._pos + take])
                filled    += take
                self._pos += take
            if self._xfade_buf is None and self._pos >= stop:
                self._build_xfade()
        return out


# ──────────────────────────────────────────────────────────────────────────────
#  SNR → signal scale  (same formula as EME simulator)
# ──────────────────────────────────────────────────────────────────────────────
def snr_to_scale(snr_db: float) -> float:
    raw = float(np.sqrt(2.0 * BW_REF / 6000.0) * 10.0 ** (0.05 * snr_db))
    return raw * (OUT_PEAK / 4.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Continuous-phase tone generator
# ──────────────────────────────────────────────────────────────────────────────
class ToneGen:
    def __init__(self):
        self._phi  = 0.0
        self._dphi = 2.0 * np.pi * DEFAULT_FREQ / SAMPLE_RATE

    def set_freq(self, freq: float):
        self._dphi = 2.0 * np.pi * float(freq) / SAMPLE_RATE
        self._phi  = 0.0

    def get(self, n: int) -> np.ndarray:
        phi       = self._phi + self._dphi * np.arange(n, dtype=np.float64)
        self._phi = float(phi[-1] + self._dphi) % (2.0 * np.pi)
        return np.exp(1j * phi).astype(np.complex64)


# ──────────────────────────────────────────────────────────────────────────────
#  Morse keyer
#  Converts a text message into a sample-accurate on/off key signal.
#  Returns a boolean numpy array: True = tone on, False = tone off.
# ──────────────────────────────────────────────────────────────────────────────
class MorseKeyer:
    """Pre-renders the entire message as a boolean key array."""

    # Element durations in dits
    DIT       = 1
    DAH       = 3
    ELEM_GAP  = 1   # between elements within a character
    CHAR_GAP  = 3   # between characters (includes trailing ELEM_GAP)
    WORD_GAP  = 7   # between words (includes trailing CHAR_GAP)

    def __init__(self, message: str, wpm: float):
        self._key    = self._encode(message.upper(), wpm)
        self._pos    = 0
        self._done   = False

    def _dit_samples(self, wpm: float) -> int:
        """1200 / wpm gives dit length in ms; convert to samples."""
        ms = 1200.0 / max(1.0, float(wpm))
        return max(1, int(round(ms * SAMPLE_RATE / 1000.0)))

    def _encode(self, text: str, wpm: float) -> np.ndarray:
        dit = self._dit_samples(wpm)
        segments = []   # list of (n_samples, is_on)

        words = text.split(' ')
        for wi, word in enumerate(words):
            for ci, ch in enumerate(word):
                code = MORSE_TABLE.get(ch)
                if code is None:
                    continue
                for ei, elem in enumerate(code):
                    # Mark
                    dur = (self.DAH if elem == '-' else self.DIT) * dit
                    segments.append((dur, True))
                    # Inter-element gap (not after last element)
                    if ei < len(code) - 1:
                        segments.append((self.ELEM_GAP * dit, False))
                # Inter-character gap (not after last char in word)
                if ci < len(word) - 1:
                    segments.append((self.CHAR_GAP * dit, False))
            # Inter-word gap (not after last word); CHAR_GAP already added if
            # the word was non-empty, so we only need 4 more dits here
            if wi < len(words) - 1:
                extra = (self.WORD_GAP - self.CHAR_GAP) * dit
                segments.append((extra, False))

        if not segments:
            return np.zeros(dit, dtype=bool)

        total = sum(n for n, _ in segments)
        key   = np.zeros(total, dtype=bool)
        pos   = 0
        for n, on in segments:
            if on:
                key[pos:pos + n] = True
            pos += n
        return key

    def get(self, n: int) -> np.ndarray:
        """Return n key samples.  Pads with False once message is finished."""
        if self._done:
            return np.zeros(n, dtype=bool)
        remaining = len(self._key) - self._pos
        if remaining <= 0:
            self._done = True
            return np.zeros(n, dtype=bool)
        take = min(n, remaining)
        out  = np.zeros(n, dtype=bool)
        out[:take] = self._key[self._pos:self._pos + take]
        self._pos += take
        if self._pos >= len(self._key):
            self._done = True
        return out

    @property
    def done(self) -> bool:
        return self._done

    @property
    def total_samples(self) -> int:
        return len(self._key)


# ──────────────────────────────────────────────────────────────────────────────
#  Butterworth bandpass filter  (chunk-by-chunk, stateful)
# ──────────────────────────────────────────────────────────────────────────────
class BandpassFilter:
    """4th-order Butterworth BPF centred on freq_hz with bandwidth bw_hz.
    Processes audio chunk by chunk, preserving filter state between calls
    so there are no discontinuities at chunk boundaries.
    bw_hz <= 0 means bypass (no filtering)."""

    ORDER = 4

    def __init__(self):
        self._sos  = None
        self._zi   = None
        self._freq = DEFAULT_FREQ
        self._bw   = DEFAULT_FILTER_BW
        self._bypass = False
        self._rebuild()

    def set_params(self, freq_hz: float, bw_hz: float):
        changed = (freq_hz != self._freq or bw_hz != self._bw)
        self._freq = freq_hz
        self._bw   = bw_hz
        if changed:
            self._rebuild()

    def _rebuild(self):
        self._bypass = (self._bw <= 0.0)
        if self._bypass:
            self._sos = None
            self._zi  = None
            return
        nyq   = SAMPLE_RATE / 2.0
        half  = self._bw / 2.0
        low   = max(10.0,        self._freq - half)
        high  = min(nyq - 10.0, self._freq + half)
        # Guard against degenerate band (e.g. very low freq + very wide BW)
        if low >= high:
            self._bypass = True
            return
        try:
            sos = butter(self.ORDER, [low / nyq, high / nyq],
                         btype='bandpass', output='sos')
            self._sos = sos
            # Initialise state to zero (reset on every rebuild)
            self._zi  = sosfilt_zi(sos).astype(np.float64)
        except Exception as e:
            print(f"[BandpassFilter] design error: {e}  — bypassing")
            self._bypass = True

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._bypass or self._sos is None:
            return x
        y, self._zi = sosfilt(self._sos, x.astype(np.float64), zi=self._zi)
        return y.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  DSP thread
#  Pulls key samples from MorseKeyer, generates tone, applies fading + noise,
#  and pushes float32 audio to out_q.
# ──────────────────────────────────────────────────────────────────────────────
class DSPThread(threading.Thread):
    # ~0.5 s of audio pre-buffered: enough headroom without runaway memory
    QUEUE_DEPTH = 12   # 12 * 512 / 12000 = 0.51 s

    def __init__(self):
        super().__init__(daemon=True)
        self.out_q   = queue.Queue(maxsize=self.QUEUE_DEPTH)
        self._tone   = ToneGen()
        self._fade   = FadeEnvelope()
        self._fade.set_fspread(DEFAULT_SPREAD)

        self._snr_db   = DEFAULT_SNR
        self._level    = DEFAULT_LEVEL
        self._filter   = BandpassFilter()
        self._filter_bw = DEFAULT_FILTER_BW
        self._running  = False
        self._keyer   = None
        self._lock    = threading.Lock()
        self._abort   = threading.Event()  # set to unblock a stuck _put()

        self.out_level  = 0.0
        self.done_event = threading.Event()
        self.start()

    # ── queue helpers ─────────────────────────────────────────────
    def _put(self, chunk: np.ndarray) -> bool:
        """Non-deadlocking put: retries every 50 ms, returns False if aborted."""
        while not self._abort.is_set():
            try:
                self.out_q.put(chunk, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def flush_queue(self):
        """Drain all buffered chunks (call after stop_message)."""
        while True:
            try:    self.out_q.get_nowait()
            except queue.Empty: break

    # ── parameter setters ────────────────────────────────────────
    def set_freq(self, v):
        self._tone.set_freq(float(v))
        self._filter.set_params(float(v), self._filter_bw)
    def set_snr(self, v):
        self._snr_db = float(v)
    def set_fspread(self, v):
        self._fade.set_fspread(float(v))
    def set_level(self, v):
        self._level = max(0.0, min(1.0, float(v)))
    def set_filter(self, freq_hz: float, bw_hz: float):
        self._filter_bw = float(bw_hz)
        self._filter.set_params(float(freq_hz), float(bw_hz))

    def send_message(self, message: str, wpm: float):
        """Start a new transmission (flush_queue must be called first)."""
        self._abort.clear()
        with self._lock:
            self._keyer   = MorseKeyer(message, wpm)
            self._running = True
            self.done_event.clear()

    def stop_message(self):
        """Stop transmission and immediately unblock any stuck _put()."""
        self._abort.set()
        with self._lock:
            self._keyer   = None
            self._running = False
        self.done_event.set()

    def run(self):
        import time
        silence = np.zeros(CHUNK, dtype=np.float32)
        while True:
            with self._lock:
                running = self._running
                keyer   = self._keyer

            if not running or keyer is None:
                self.out_level = 0.0
                # Idle: sleep one chunk period so we don't spin the CPU
                time.sleep(CHUNK / SAMPLE_RATE)
                continue

            # ── generate one chunk ────────────────────────────────
            key     = keyer.get(CHUNK)
            tone_iq = self._tone.get(CHUNK)
            tone_iq[~key] = 0.0

            env  = self._fade.get(CHUNK)
            cdat = env * tone_iq

            sig_scale = snr_to_scale(self._snr_db)
            noise     = (np.random.randn(CHUNK) * (OUT_PEAK / 4.0)).astype(np.float32)
            dat       = sig_scale * np.imag(cdat).astype(np.float32) + noise
            dat      *= self._level
            dat       = self._filter.process(dat)
            np.clip(dat, -1.0, 1.0, out=dat)

            self.out_level = float(np.sqrt(np.mean(dat ** 2)) + 1e-12)

            if not self._put(dat):
                continue    # abort fired mid-message; discard this chunk

            # ── check if message finished ─────────────────────────
            if keyer.done:
                # Send 1 second of noise at the same level before stopping
                tail_chunks = int(SAMPLE_RATE / CHUNK)   # ≈ 23 chunks @ 12 kHz
                for _ in range(tail_chunks):
                    if self._abort.is_set():
                        break
                    noise_tail = (np.random.randn(CHUNK) * (OUT_PEAK / 4.0)).astype(np.float32)
                    noise_tail *= self._level
                    noise_tail  = self._filter.process(noise_tail)
                    np.clip(noise_tail, -1.0, 1.0, out=noise_tail)
                    self.out_level = float(np.sqrt(np.mean(noise_tail ** 2)) + 1e-12)
                    if not self._put(noise_tail):
                        break
                with self._lock:
                    self._running = False
                    self._keyer   = None
                self.done_event.set()


# ──────────────────────────────────────────────────────────────────────────────
#  Audio engine  — persistent stream, never opened/closed between messages
# ──────────────────────────────────────────────────────────────────────────────
class AudioEngine:
    def __init__(self, dsp: DSPThread):
        self._dsp     = dsp
        self._stream  = None
        self._silence = np.zeros(CHUNK, dtype=np.float32)
        self._cur_dev = None

    def open(self, out_dev):
        """Open stream on out_dev; reopen only if device changed."""
        if self._stream is not None and out_dev == self._cur_dev:
            return
        self.close()
        self._stream = sd.OutputStream(
            samplerate = SAMPLE_RATE,
            blocksize  = CHUNK,
            dtype      = 'float32',
            channels   = 1,
            device     = out_dev,
            callback   = self._cb,
        )
        self._stream.start()
        self._cur_dev = out_dev

    def close(self):
        """Only called on exit or device change — never between messages."""
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception as e:
                print(f"[audio close] {e}")
            self._stream  = None
            self._cur_dev = None

    def _cb(self, outdata, frames, time, status):
        try:
            if status:
                print(f"[audio cb] {status}")
            try:
                out = self._dsp.out_q.get_nowait()
            except queue.Empty:
                out = self._silence
            outdata[:, 0] = out
        except Exception as e:
            print(f"[audio cb exception] {e}")
            outdata[:] = 0


# ──────────────────────────────────────────────────────────────────────────────
#  Tooltip  (lightweight hover popup, no extra libraries needed)
# ──────────────────────────────────────────────────────────────────────────────
class Tooltip:
    """Show a small popup label when the mouse hovers over a widget."""
    PAD  = 4
    DELAY = 600   # ms before tooltip appears

    def __init__(self, widget, text: str):
        self._widget = widget
        self._text   = text
        self._id     = None
        self._win    = None
        widget.bind("<Enter>",  self._schedule, add="+")
        widget.bind("<Leave>",  self._cancel,   add="+")
        widget.bind("<ButtonPress>", self._cancel, add="+")

    def _schedule(self, event=None):
        self._cancel()
        self._id = self._widget.after(self.DELAY, self._show)

    def _cancel(self, event=None):
        if self._id:
            self._widget.after_cancel(self._id)
            self._id = None
        if self._win:
            self._win.destroy()
            self._win = None

    def _show(self):
        if self._win:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self._text, justify='left',
                 bg="#fffde7", fg="#333333",
                 font=("Segoe UI", 9, "bold"),
                 relief='solid', bd=1,
                 padx=self.PAD, pady=self.PAD).pack()


# ──────────────────────────────────────────────────────────────────────────────
#  GUI
# ──────────────────────────────────────────────────────────────────────────────
class MorseGUI:
    # Light theme
    BG       = "#f0f2f5"
    PANEL    = "#ffffff"
    ACCENT   = "#0066cc"
    ACCENT2  = "#004999"
    TEXT     = "#000000"
    LABEL    = "#555e6b"
    RED      = "#cc2200"
    ENTRY_BG = "#f8f9fb"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Morse EME Signal Simulator by DL3WDG")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)
        self.root.minsize(700, 600)

        self.dsp    = DSPThread()
        self.engine = AudioEngine(self.dsp)
        self.running = False
        self._settings = load_settings()
        self._current_dx = ""   # DX call picked for current round
        self.mode_var = tk.StringVar(value=self._settings.get('mode', 'trainer'))

        self._build_device_lists()
        self._build_gui()
        # Open the audio stream immediately on the default device so it
        # stays open for the lifetime of the program.
        self._open_stream()
        self._poll()

    # ── device enumeration ───────────────────────────────────────
    def _build_device_lists(self):
        devs = sd.query_devices()
        self._out_names = []
        self._out_idx   = []
        for i, d in enumerate(devs):
            if d['max_output_channels'] > 0:
                self._out_names.append(f"{d['name']}")
                self._out_idx.append(i)

    # ── stream helpers ──────────────────────────────────────────
    def _open_stream(self):
        """Open (or reopen) the audio stream for the currently selected device."""
        idx = self.out_menu.current()
        if idx < 0:
            return
        out_dev = self._out_idx[idx]
        try:
            self.engine.open(out_dev)
        except Exception as e:
            self.status_var.set(f"Audio open error: {e}")
            print(f"[open_stream] {e}")

    # ── GUI layout ───────────────────────────────────────────────
    def _build_gui(self):
        root = self.root
        PAD  = dict(padx=16, pady=6)

        def section_label(parent, text):
            tk.Label(parent, text=text, bg=self.BG, fg=self.ACCENT2,
                     font=("Segoe UI", 9, "bold")).pack(anchor='w', padx=16, pady=(10, 0))

        def row_frame(parent):
            f = tk.Frame(parent, bg=self.PANEL,
                         highlightbackground=self.ACCENT2,
                         highlightthickness=1)
            f.pack(fill='x', padx=16, pady=2)
            return f

        def lbl(parent, text, width=18):
            w = tk.Label(parent, text=text, bg=self.PANEL, fg=self.LABEL,
                         font=("Segoe UI", 10, "bold"), width=width, anchor='w')
            w.pack(side='left', padx=(10, 0), pady=4)
            return w

        def val_label(parent, var, width=8):
            w = tk.Label(parent, textvariable=var, bg=self.PANEL, fg=self.ACCENT,
                         font=("Segoe UI", 10, "bold"), width=width, anchor='w')
            w.pack(side='left')
            return w

        def slider(parent, var, from_, to_, res, cmd):
            w = tk.Scale(parent, variable=var, from_=from_, to=to_,
                         resolution=res, orient='horizontal', command=cmd,
                         bg=self.PANEL, fg=self.TEXT, troughcolor=self.BG,
                         activebackground=self.ACCENT, highlightthickness=0,
                         bd=0, showvalue=False)
            w.pack(side='left', padx=8, fill='x', expand=True)
            return w

        # ── Title bar ──────────────────────────────────────────
        title_f = tk.Frame(root, bg=self.ACCENT2)
        title_f.pack(fill='x')
        tk.Label(title_f,
                 text="  MORSE EME SIGNAL SIMULATOR  by DL3WDG",
                 bg=self.ACCENT2, fg="#ffffff",
                 font=("Segoe UI", 13, "bold")).pack(side='left', pady=6)
        tk.Label(title_f,
                 text="Rayleigh Lorentzian fading · AWGN · CW tone  ",
                 bg=self.ACCENT2, fg="#cce0ff",
                 font=("Segoe UI", 9, "bold")).pack(side='right', pady=6)

        # ── Callsign fields ────────────────────────────────────
        # ── Mode selector ──────────────────────────────────────
        section_label(root, "── MODE ─────────────────────────────────────────")
        mf = tk.Frame(root, bg=self.PANEL,
                      highlightbackground=self.ACCENT2, highlightthickness=1)
        mf.pack(fill='x', padx=16, pady=2)
        rb_trainer = tk.Radiobutton(mf, text="Trainer  (mycall DE dxcall)",
                           variable=self.mode_var, value='trainer',
                           bg=self.PANEL, fg=self.TEXT,
                           selectcolor=self.ENTRY_BG,
                           activebackground=self.PANEL,
                           font=("Segoe UI", 10, "bold"),
                           command=self._on_mode_change)
        rb_trainer.pack(side='left', padx=(10, 20), pady=6)
        Tooltip(rb_trainer, "Trainer mode: simulates receiving a call\n"
                            "from a DX station addressed to you.\n"
                            "Message: MYCALL MYCALL MYCALL DE DXCALL DXCALL DXCALL K K K")
        rb_cq = tk.Radiobutton(mf, text="CQ  (CQ CQ CQ DE dxcall)",
                           variable=self.mode_var, value='cq',
                           bg=self.PANEL, fg=self.TEXT,
                           selectcolor=self.ENTRY_BG,
                           activebackground=self.PANEL,
                           font=("Segoe UI", 10, "bold"),
                           command=self._on_mode_change)
        rb_cq.pack(side='left', padx=(0, 10), pady=6)
        Tooltip(rb_cq, "CQ mode: simulates receiving a CQ call\n"
                       "from an unknown station.\n"
                       "Message: CQ CQ CQ DE DXCALL DXCALL DXCALL K K K")

        section_label(root, "── CALLSIGNS ────────────────────────────────────")
        cf = tk.Frame(root, bg=self.PANEL,
                      highlightbackground=self.ACCENT2, highlightthickness=1)
        cf.pack(fill='x', padx=16, pady=2)
        tk.Label(cf, text="My Call:", bg=self.PANEL, fg=self.LABEL,
                 font=("Segoe UI", 10, "bold"), width=10, anchor='w'
                 ).pack(side='left', padx=(10, 0), pady=6)
        self.mycall_var = tk.StringVar(value=self._settings['mycall'])
        _w = tk.Entry(cf, textvariable=self.mycall_var,
                 bg=self.ENTRY_BG, fg=self.TEXT,
                 insertbackground=self.ACCENT,
                 font=("Segoe UI", 11, "bold"), bd=0, relief='flat',
                 width=12)
        _w.pack(side='left', padx=(4, 20), pady=6)
        Tooltip(_w, "Your own callsign.\nUsed in the transmitted message.")
        tk.Label(cf, text="DX Call:", bg=self.PANEL, fg=self.LABEL,
                 font=("Segoe UI", 10, "bold"), width=10, anchor='w'
                 ).pack(side='left', padx=(0, 0), pady=6)
        self._dxcall_display = tk.StringVar(value="????????")
        _w = tk.Label(cf, textvariable=self._dxcall_display,
                 bg=self.ENTRY_BG, fg=self.LABEL,
                 font=("Segoe UI", 11, "bold"), width=12, anchor='w')
        _w.pack(side='left', padx=4, pady=6)
        Tooltip(_w, "The DX station callsign hidden in the message.\nRevealed when you answer correctly.")
        # Hidden msg_var still needed by _update_morse_preview and send logic
        self.msg_var = tk.StringVar(value="")

        # ── Signal parameters ──────────────────────────────────
        section_label(root, "── SIGNAL PARAMETERS ────────────────────────────")

        # Tone frequency
        rf = row_frame(root)
        self.freq_var = tk.DoubleVar(value=DEFAULT_FREQ)
        self._freq_lbl = tk.StringVar(value=f"{DEFAULT_FREQ:.0f} Hz")
        tip_txt = "CW tone pitch (200–3000 Hz).\nTakes effect on next play."
        Tooltip(lbl(rf, "Tone freq (Hz):"), tip_txt)
        Tooltip(slider(rf, self.freq_var, 200, 3000, 10, self._on_freq), tip_txt)
        Tooltip(val_label(rf, self._freq_lbl), tip_txt)

        # Morse speed
        rf = row_frame(root)
        self.wpm_var = tk.DoubleVar(value=DEFAULT_WPM)
        self._wpm_lbl = tk.StringVar(value=f"{DEFAULT_WPM} WPM")
        tip_txt = "Morse speed in words per minute (1–30).\nTakes effect on next play."
        Tooltip(lbl(rf, "Speed (WPM):"), tip_txt)
        Tooltip(slider(rf, self.wpm_var, 1, 30, 1, self._on_wpm), tip_txt)
        Tooltip(val_label(rf, self._wpm_lbl), tip_txt)

        # SNR
        rf = row_frame(root)
        self.snr_var = tk.DoubleVar(value=DEFAULT_SNR)
        self._snr_lbl = tk.StringVar(value=f"{DEFAULT_SNR:+.0f} dB")
        tip_txt = "Signal-to-noise ratio in dB (−30 to +10).\nLive — adjustable during playback."
        Tooltip(lbl(rf, "SNR (dB):"), tip_txt)
        Tooltip(slider(rf, self.snr_var, -30, 10, 0.5, self._on_snr), tip_txt)
        Tooltip(val_label(rf, self._snr_lbl), tip_txt)

        # Doppler spread
        rf = row_frame(root)
        self.spread_var = tk.DoubleVar(value=DEFAULT_SPREAD)
        self._spread_lbl = tk.StringVar(value=f"{DEFAULT_SPREAD:.1f} Hz")
        tip_txt = "Rayleigh/Lorentzian Doppler spread (0–400 Hz).\n0 = no fading. Live — adjustable during playback."
        Tooltip(lbl(rf, "Spread (Hz):"), tip_txt)
        Tooltip(slider(rf, self.spread_var, 0, 400, 0.5, self._on_spread), tip_txt)
        Tooltip(val_label(rf, self._spread_lbl), tip_txt)

        # Output level
        rf = row_frame(root)
        self.level_var = tk.DoubleVar(value=self._settings['level'])
        self._level_lbl = tk.StringVar(value=f"{int(self._settings['level']*100)} %")
        tip_txt = "Output volume (0–100%).\nLive — adjustable during playback."
        Tooltip(lbl(rf, "Level:"), tip_txt)
        Tooltip(slider(rf, self.level_var, 0.0, 1.0, 0.01, self._on_level), tip_txt)
        Tooltip(val_label(rf, self._level_lbl), tip_txt)

        # Filter bandwidth
        rf = row_frame(root)
        self.filter_var = tk.DoubleVar(value=self._settings['filter_bw'])
        self._filter_lbl = tk.StringVar(value=self._fmt_bw(self._settings['filter_bw']))
        tip_txt = "Butterworth bandpass filter bandwidth (0–2500 Hz).\n0 = bypass. Centred on tone frequency.\nLive — adjustable during playback."
        Tooltip(lbl(rf, "Filter BW (Hz):"), tip_txt)
        Tooltip(slider(rf, self.filter_var, 0, 2500, 10, self._on_filter), tip_txt)
        Tooltip(val_label(rf, self._filter_lbl), tip_txt)

        # ── Output device ──────────────────────────────────────
        section_label(root, "── OUTPUT DEVICE ────────────────────────────────")
        df = row_frame(root)
        lbl(df, "Audio output:")
        self.out_menu = ttk.Combobox(df, values=self._out_names,
                                     state='readonly', width=46,
                                     font=("Segoe UI", 10, "bold"))
        self.out_menu.pack(side='left', padx=8, pady=6)
        Tooltip(self.out_menu, "Select the audio output device.\nChanging this reopens the audio stream.")
        # Select default output device
        try:
            default_out = sd.default.device[1]
            out_pos = self._out_idx.index(default_out)
            self.out_menu.current(out_pos)
        except Exception:
            if self._out_names:
                self.out_menu.current(0)
        self.out_menu.bind("<<ComboboxSelected>>", lambda e: self._open_stream())

        # Restore saved settings to all controls
        self.msg_var.set(self._settings['message'])
        self.freq_var.set(self._settings['freq'])
        self.wpm_var.set(self._settings['wpm'])
        self.snr_var.set(self._settings['snr'])
        self.spread_var.set(self._settings['spread'])
        self.level_var.set(self._settings['level'])
        self.filter_var.set(self._settings['filter_bw'])
        self.mycall_var.set(self._settings['mycall'])
        self._freq_lbl.set(f"{self._settings['freq']:.0f} Hz")
        self._wpm_lbl.set(f"{int(self._settings['wpm'])} WPM")
        self._snr_lbl.set(f"{self._settings['snr']:+.1f} dB")
        self._spread_lbl.set(f"{self._settings['spread']:.1f} Hz")
        self._level_lbl.set(f"{int(self._settings['level']*100)} %")
        self._filter_lbl.set(self._fmt_bw(self._settings['filter_bw']))
        self.dsp.set_freq(self._settings['freq'])
        self.dsp.set_snr(self._settings['snr'])
        self.dsp.set_fspread(self._settings['spread'])
        self.dsp.set_level(self._settings['level'])
        self.dsp.set_filter(self._settings['freq'], self._settings['filter_bw'])
        # Restore saved device selection
        saved_dev = self._settings['device']
        if saved_dev and saved_dev in self._out_names:
            self.out_menu.current(self._out_names.index(saved_dev))

        # ── Status / level ─────────────────────────────────────
        section_label(root, "── STATUS ───────────────────────────────────────")
        sf = tk.Frame(root, bg=self.PANEL,
                      highlightbackground=self.ACCENT2, highlightthickness=1)
        sf.pack(fill='x', padx=16, pady=2)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(sf, textvariable=self.status_var,
                 bg=self.PANEL, fg=self.ACCENT,
                 font=("Segoe UI", 10, "bold"), anchor='w').pack(side='left',
                                                       padx=10, pady=4, fill='x', expand=True)
        # Level bar
        self._level_canvas = tk.Canvas(sf, width=160, height=14,
                                       bg=self.PANEL, highlightthickness=0)
        self._level_canvas.pack(side='right', padx=10, pady=4)
        self._init_level_meter()

        # ── Play button ────────────────────────────────────────
        btn_f = tk.Frame(root, bg=self.BG)
        btn_f.pack(fill='x', padx=16, pady=(12, 4))
        self.send_btn = tk.Button(btn_f,
                                  text="▶  PLAY",
                                  font=("Segoe UI", 14, "bold"),
                                  bg=self.ACCENT, fg=self.BG,
                                  activebackground=self.ACCENT2,
                                  activeforeground=self.TEXT,
                                  bd=0, padx=24, pady=8,
                                  command=self._on_send_stop)
        self.send_btn.pack(side='left')
        Tooltip(self.send_btn, "Play a new message with a randomly\npicked DX callsign from the list.\nClick again to stop.")

        # Morse preview label
        self._morse_var = tk.StringVar(value="")
        tk.Label(btn_f, textvariable=self._morse_var,
                 bg=self.BG, fg=self.ACCENT2,
                 font=("Segoe UI", 9, "bold"), wraplength=540, justify='left'
                 ).pack(side='left', padx=16)

        # ── Answer section ─────────────────────────────────────
        section_label(root, "── YOUR ANSWER ──────────────────────────────────")
        af = tk.Frame(root, bg=self.PANEL,
                      highlightbackground=self.ACCENT2, highlightthickness=1)
        af.pack(fill='x', padx=16, pady=2)
        tk.Label(af, text="DX Callsign:", bg=self.PANEL, fg=self.LABEL,
                 font=("Segoe UI", 10, "bold"), width=14, anchor='w'
                 ).pack(side='left', padx=(10, 0), pady=8)
        self.answer_var = tk.StringVar()
        self.answer_entry = tk.Entry(af, textvariable=self.answer_var,
                                     bg=self.ENTRY_BG, fg=self.TEXT,
                                     insertbackground=self.ACCENT,
                                     font=("Segoe UI", 13, "bold"),
                                     bd=0, relief='flat', width=14,
                                     state='disabled',
                                     disabledbackground='#d8dce4',
                                     disabledforeground='#aaaaaa')
        self.answer_entry.pack(side='left', padx=8, pady=8)
        self.answer_entry.bind("<Return>", lambda e: self._on_check())
        Tooltip(self.answer_entry, "Type the DX callsign you heard.\nPress Enter or CHECK to submit.")
        self.check_btn = tk.Button(af, text="CHECK",
                                   font=("Segoe UI", 11, "bold"),
                                   bg=self.LABEL, fg=self.BG,
                                   activebackground=self.ACCENT2,
                                   activeforeground="#ffffff",
                                   bd=0, padx=12, pady=4,
                                   state='disabled',
                                   command=self._on_check)
        self.check_btn.pack(side='left', padx=4)
        Tooltip(self.check_btn, "Submit your answer.\nOr press Enter in the answer field.")
        self.replay_btn = tk.Button(af, text="↺  REPLAY",
                                    font=("Segoe UI", 11, "bold"),
                                    bg=self.ACCENT2, fg="#ffffff",
                                    activebackground=self.ACCENT,
                                    activeforeground=self.BG,
                                    bd=0, padx=12, pady=4,
                                    state='disabled',
                                    command=self._on_replay)
        self.replay_btn.pack(side='left', padx=4)
        Tooltip(self.replay_btn, "Replay the SAME message again.\nThe DX callsign does not change.")
        self.result_var = tk.StringVar(value="")
        self.result_lbl = tk.Label(af, textvariable=self.result_var,
                                   bg=self.PANEL,
                                   font=("Segoe UI", 12, "bold"), anchor='w')
        self.result_lbl.pack(side='left', padx=12)

        root.after(1, self._update_morse_preview)

    # ── slider callbacks ─────────────────────────────────────────
    def _on_freq(self, v):
        self._freq_lbl.set(f"{float(v):.0f} Hz")
        self.dsp.set_freq(float(v))   # also recentres the filter
        if self.running:
            self.status_var.set("Tone frequency: takes effect on next play.")

    def _on_wpm(self, v):
        self._wpm_lbl.set(f"{int(float(v))} WPM")
        self.root.after(100, self._update_morse_preview)
        if self.running:
            self.status_var.set("Speed: takes effect on next play.")

    def _on_snr(self, v):
        self._snr_lbl.set(f"{float(v):+.1f} dB")
        self.dsp.set_snr(float(v))
        if self.running:
            self.status_var.set(f"SNR → {float(v):+.1f} dB  (live)")

    def _on_spread(self, v):
        fv = float(v)
        self._spread_lbl.set(f"{fv:.1f} Hz")
        self.dsp.set_fspread(fv)
        if self.running:
            self.status_var.set(f"Spread → {fv:.1f} Hz  (live)")

    def _on_level(self, v):
        fv = float(v)
        self._level_lbl.set(f"{int(fv*100)} %")
        self.dsp.set_level(fv)
        if self.running:
            self.status_var.set(f"Level → {int(fv*100)} %  (live)")

    @staticmethod
    def _fmt_bw(bw: float) -> str:
        if bw <= 0:
            return "Bypass"
        return f"{int(bw)} Hz"

    def _on_filter(self, v):
        bw = float(v)
        self._filter_lbl.set(self._fmt_bw(bw))
        self.dsp.set_filter(self.freq_var.get(), bw)
        if self.running:
            self.status_var.set(f"Filter BW → {self._fmt_bw(bw)}  (live)")

    # ── morse preview ────────────────────────────────────────────
    def _update_morse_preview(self):
        has_call = self.mycall_var.get().strip() or self.mode_var.get() == 'cq'
        msg = self._build_message() if (has_call and self._current_dx) else self.msg_var.get()
        msg = msg.upper()
        parts = []
        for ch in msg:
            if ch == ' ':
                parts.append('/')
            else:
                code = MORSE_TABLE.get(ch)
                if code:
                    parts.append(code)
                else:
                    parts.append('?')
        preview = '  '.join(parts)
        # Estimate duration
        wpm = max(1.0, float(self.wpm_var.get()))
        dit_ms = 1200.0 / wpm
        n_dits = 0
        for ch in msg:
            if ch == ' ':
                n_dits += MorseKeyer.WORD_GAP
            else:
                code = MORSE_TABLE.get(ch)
                if code:
                    for e in code:
                        n_dits += (3 if e == '-' else 1)
                    n_dits += len(code) - 1   # inter-element gaps
                    n_dits += MorseKeyer.CHAR_GAP
        dur_s = n_dits * dit_ms / 1000.0
        self._morse_var.set(f"{preview}\n≈ {dur_s:.1f} s  @ {int(wpm)} WPM")

    # ── send / stop ──────────────────────────────────────────────
    def _build_message(self) -> str:
        """Build message depending on mode."""
        dx = self._current_dx.upper()
        if self.mode_var.get() == 'cq':
            return f"CQ CQ CQ DE {dx} {dx} {dx} K K K"
        else:
            mc = self.mycall_var.get().strip().upper() or "DE"
            return f"{mc} {mc} {mc} DE {dx} {dx} {dx} K K K"

    def _on_send_stop(self):
        if not self.running:
            mycall = self.mycall_var.get().strip()
            if self.mode_var.get() == 'trainer' and not mycall:
                self.status_var.set("Enter your callsign first.")
                return
            # Pick a new DX call (exclude mycall in trainer mode)
            self._current_dx = pick_dx_call(mycall if mycall else "NONE")
            self._dxcall_display.set("????????")
            msg = self._build_message()
            self.msg_var.set(msg)
            # Grey out answer while playing
            self._set_answer_state(False)
            self.result_var.set("")
            try:
                self.dsp.set_freq(self.freq_var.get())
                self.dsp.set_snr(self.snr_var.get())
                self.dsp.set_fspread(self.spread_var.get())
                self.dsp.stop_message()
                self.dsp.flush_queue()
                self.dsp.send_message(msg, self.wpm_var.get())
                self.running = True
                self.send_btn.config(text="■  STOP", bg=self.RED, fg=self.BG)
                wpm = int(self.wpm_var.get())
                snr = self.snr_var.get()
                self.status_var.set(
                    f"Playing  |  {wpm} WPM  |  SNR {snr:+.1f} dB")
                self._save()
                self.root.after(200, self._check_done)
            except Exception as e:
                self.status_var.set(f"Error: {e}")
                print(f"[start ERROR] {e}")
        else:
            self._stop()

    def _check_done(self):
        if not self.running:
            return
        if self.dsp.done_event.is_set():
            # Poll until the audio callback has drained all queued chunks
            self.root.after(80, self._wait_drain)
        else:
            self.root.after(200, self._check_done)

    def _wait_drain(self):
        """Keep checking until the output queue is empty, then finish."""
        if not self.running:
            return
        if self.dsp.out_q.empty():
            # One extra audio-device latency period for the final chunk to play
            self.root.after(150, self._finish)
        else:
            self.root.after(80, self._wait_drain)

    def _finish(self):
        self._stop()
        self.status_var.set("Message finished — enter the DX callsign above.")
        self._set_answer_state(True)   # both modes require an answer

    def _stop(self):
        self.dsp.stop_message()   # sets abort + clears keyer
        self.dsp.flush_queue()    # drain any buffered chunks; stream keeps running
        self.running = False
        self.send_btn.config(text="▶  PLAY", bg=self.ACCENT, fg=self.BG)

    # ── answer helpers ──────────────────────────────────────
    def _set_answer_state(self, enabled: bool):
        state = 'normal' if enabled else 'disabled'
        self.answer_entry.config(state=state)
        self.check_btn.config(state=state)
        self.replay_btn.config(state=state)
        if enabled:
            self.answer_var.set("")
            self.answer_entry.focus_set()

    def _on_mode_change(self):
        """Called when mode radio button changes."""
        self.result_var.set("")
        self._dxcall_display.set("????????")
        self._set_answer_state(False)
        mode = self.mode_var.get()
        if mode == 'cq':
            self.status_var.set("CQ mode — press PLAY to hear a CQ call.")
        else:
            self.status_var.set("Trainer mode — press PLAY to hear a call to your station.")

    def _on_replay(self):
        """Replay the same message without picking a new DX call."""
        self._set_answer_state(False)
        self.result_var.set("")
        msg = self._build_message()   # uses same _current_dx
        self.msg_var.set(msg)
        try:
            self.dsp.set_freq(self.freq_var.get())
            self.dsp.set_snr(self.snr_var.get())
            self.dsp.set_fspread(self.spread_var.get())
            self.dsp.stop_message()
            self.dsp.flush_queue()
            self.dsp.send_message(msg, self.wpm_var.get())
            self.running = True
            self.send_btn.config(text="■  STOP", bg=self.RED, fg=self.BG)
            self.status_var.set("Replaying same message...")
            self.root.after(200, self._check_done)
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            print(f"[replay ERROR] {e}")

    def _on_check(self):
        answer = self.answer_var.get().strip().upper()
        if not answer:
            return
        if answer == self._current_dx.upper():
            self.result_var.set("✓  CORRECT!")
            self.result_lbl.config(fg="#007700")
            self.status_var.set("Correct! Press PLAY for a new callsign.")
            self._set_answer_state(False)
            self._dxcall_display.set(self._current_dx)
        else:
            self.result_var.set("✗  Wrong — press REPLAY")
            self.result_lbl.config(fg=self.RED)
            self.answer_var.set("")
            self.status_var.set("Incorrect — press REPLAY to hear the same message again.")

    # ── poll loop (GUI refresh @ 80 ms) ──────────────────────────
    def _poll(self):
        self._draw_level(self.dsp.out_level)
        self.root.after(80, self._poll)

    def _init_level_meter(self):
        """Create the canvas items once; _draw_level updates coords/colour."""
        c    = self._level_canvas
        W, H = 160, 14
        c.create_rectangle(0, 0, W, H, fill='#e0e4ea', outline='', tags='bg')
        c.create_rectangle(1, 2, 1,    H - 2,        fill=self.ACCENT, tags='bar')
        c.create_text(W // 2, H // 2, text="-∞ dBFS",
                      fill=self.LABEL, font=("Segoe UI", 8), tags='txt')

    def _draw_level(self, rms: float):
        c    = self._level_canvas
        W, H = 160, 14
        db   = 20.0 * np.log10(max(rms, 1e-9))
        frac = max(0.0, min(1.0, (db + 40.0) / 40.0))
        bar_w  = max(0, int(frac * (W - 2)))
        colour = self.ACCENT if frac < 0.8 else self.RED
        c.coords('bar', 1, 2, 1 + bar_w, H - 2)
        c.itemconfig('bar', fill=colour)
        c.itemconfig('txt', text=f"{db:.0f} dBFS")

    def _save(self):
        idx = self.out_menu.current()
        dev_name = self._out_names[idx] if idx >= 0 else ''
        save_settings({
            'message': self.msg_var.get(),
            'freq':    self.freq_var.get(),
            'wpm':     self.wpm_var.get(),
            'snr':     self.snr_var.get(),
            'spread':  self.spread_var.get(),
            'level':     self.level_var.get(),
            'filter_bw': self.filter_var.get(),
            'mycall':    self.mycall_var.get(),
            'mode':      self.mode_var.get(),
            'device':    dev_name,
        })

    def _on_close(self):
        self._save()
        self.dsp.stop_message()
        self.dsp.flush_queue()
        self.engine.close()       # only here do we close the stream
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        print(f"sounddevice : {sd.__version__}")
        MorseGUI().run()
    except Exception as e:
        import traceback
        print("\n*** STARTUP ERROR ***")
        traceback.print_exc()
        input("Press Enter to close...")