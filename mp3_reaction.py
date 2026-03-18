"""
mp3_reaction.py — Réactivité audio depuis un fichier MP3.

Charge et joue un fichier MP3, extrait en temps réel les bandes
de fréquences (bass / mid / high / vol) pour piloter le visualiseur.

Dépendances supplémentaires :
    pip install pydub sounddevice numpy
    + ffmpeg dans le PATH (pour le décodage MP3)

Utilisation rapide :
    from mp3_reaction import MP3Reaction

    mp3 = MP3Reaction()
    mp3.load("mon_fichier.mp3")
    mp3.play()

    # Dans la boucle de rendu :
    bands = mp3.get_bands()   # {'bass': 0.0–1.0, 'mid': …, 'high': …, 'vol': …}
    spectrum = mp3.get_spectrum()  # tableau numpy normalisé (512 bins)
"""

from __future__ import annotations

import threading
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Tentative d'import pydub (nécessite ffmpeg dans le PATH)
# ---------------------------------------------------------------------------
try:
    from pydub import AudioSegment
    _PYDUB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYDUB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constantes FFT
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100     # Hz
CHUNK = 1024            # Taille d'une fenêtre d'analyse (samples)
FFT_SIZE = 2048         # Zéro-padding pour une meilleure résolution
SPECTRUM_BINS = 512     # Taille du spectre renvoyé à l'appelant

# Plages de fréquences (Hz) pour les 3 bandes
BASS_RANGE  = (20,   300)
MID_RANGE   = (300,  4000)
HIGH_RANGE  = (4000, 20000)

# Seuil d'avertissement pour les fichiers volumineux (octets)
_LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100 Mo


def _fmt_time(seconds: float) -> str:
    """Formate un nombre de secondes en ``M:SS``."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


class MP3Reaction:
    """Lecteur MP3 avec extraction de bandes fréquentielles en temps réel.

    Callbacks disponibles
    ---------------------
    on_beat(bands, spectrum)
        Appelé à chaque chunk audio (≈ 23 ms à 44 100 Hz / 1024 samples).
        ``bands`` est un dict ``{'bass', 'mid', 'high', 'vol'}`` ∈ [0, 1].
        ``spectrum`` est un tableau numpy float32 de ``SPECTRUM_BINS`` valeurs.

    on_finish()
        Appelé une fois quand la lecture atteint la fin du fichier
        (non déclenché si ``loop=True``).

    on_error(exc)
        Appelé quand une exception survient dans ``on_beat``.
        Par défaut : affiche un avertissement et désactive ``on_beat``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, smoothing: float = 0.75) -> None:
        """
        Paramètres
        ----------
        smoothing : facteur de lissage exponentiel des bandes et du spectre.
            0 = aucun lissage (réactif), 0.9 = très lissé (inertiel).
            Valeur par défaut : 0.75.
        """
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing doit être dans [0, 1[")

        self.smoothing: float = smoothing

        self._samples: Optional[np.ndarray] = None  # float32 mono
        self._sample_rate: int = SAMPLE_RATE
        self._duration: float = 0.0
        self._metadata: dict[str, str] = {}

        self._position: int = 0
        self._paused: bool = False
        self._playing: bool = False
        self._loop: bool = False
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None

        self._spectrum = np.zeros(SPECTRUM_BINS, dtype=np.float32)
        self._bands: dict[str, float] = {"bass": 0.0, "mid": 0.0, "high": 0.0, "vol": 0.0}

        # Callbacks publics
        self.on_beat: Optional[Callable[[dict[str, float], np.ndarray], None]] = None
        self.on_finish: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = self._default_on_error

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> None:
        """Charge un fichier audio (MP3, WAV, FLAC, OGG…).

        Arrête la lecture en cours si nécessaire.

        Paramètres
        ----------
        path : chemin vers le fichier audio
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        if not _PYDUB_AVAILABLE:
            raise ImportError(
                "pydub est requis pour charger des fichiers audio.\n"
                "  pip install pydub\n"
                "  + ffmpeg dans le PATH  (https://ffmpeg.org/download.html)"
            )

        file_size = path.stat().st_size
        if file_size > _LARGE_FILE_THRESHOLD:
            warnings.warn(
                f"Fichier volumineux ({file_size / 1_048_576:.0f} Mo) — "
                "le chargement peut prendre quelques secondes.",
                stacklevel=2,
            )

        if self._playing or self._paused:
            self.stop()

        try:
            audio: AudioSegment = AudioSegment.from_file(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"Impossible de décoder '{path.name}'.\n"
                "Vérifiez que ffmpeg est installé et dans le PATH.\n"
                f"Détail : {exc}"
            ) from exc

        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        raw = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)

        with self._lock:
            self._samples = raw / 32768.0
            self._sample_rate = SAMPLE_RATE
            self._duration = len(self._samples) / SAMPLE_RATE
            self._position = 0
            self._metadata = {"filename": path.name, "format": path.suffix.lstrip(".").upper()}

        # Tentative d'extraction des tags ID3 (optionnel — mutagen non requis)
        self._load_tags(path)

    def load_and_play(self, path: str | Path, loop: bool = False) -> None:
        """Raccourci : charge puis démarre la lecture immédiatement."""
        self.load(path)
        self.play(loop=loop)

    # ------------------------------------------------------------------
    # Contrôle de lecture
    # ------------------------------------------------------------------

    def play(self, loop: bool = False) -> None:
        """Démarre la lecture depuis la position courante (non-bloquant).

        Paramètres
        ----------
        loop : si True, remet la lecture au début à la fin du fichier
        """
        if self._samples is None:
            raise RuntimeError("Aucun fichier chargé — appelez load() d'abord.")
        if self._playing:
            return

        with self._lock:
            self._loop = loop
            self._paused = False
            self._playing = True

        self._open_stream()

    def stop(self) -> None:
        """Arrête la lecture et remet la tête de lecture au début."""
        with self._lock:
            self._playing = False
            self._paused = False

        self._close_stream()

        with self._lock:
            self._position = 0

        self._reset_analysis()

    def pause(self) -> None:
        """Suspend la lecture sans changer la position."""
        if not self._playing:
            return

        with self._lock:
            self._playing = False
            self._paused = True

        self._close_stream()

    def resume(self) -> None:
        """Reprend après un ``pause()``."""
        if not self._paused or self._samples is None:
            return

        with self._lock:
            self._playing = True
            self._paused = False

        self._open_stream()

    def toggle_pause(self) -> None:
        """Bascule entre pause et lecture."""
        if self._paused:
            self.resume()
        else:
            self.pause()

    def seek(self, seconds: float) -> None:
        """Déplace la tête de lecture.

        Paramètres
        ----------
        seconds : position en secondes (clampée à [0, duration])
        """
        if self._samples is None:
            return
        seconds = max(0.0, min(seconds, self._duration))
        with self._lock:
            self._position = int(seconds * self._sample_rate)

    def seek_pct(self, pct: float) -> None:
        """Déplace la tête de lecture par pourcentage.

        Paramètres
        ----------
        pct : position dans [0.0, 1.0]
        """
        self.seek(pct * self._duration)

    # ------------------------------------------------------------------
    # Données d'analyse
    # ------------------------------------------------------------------

    def get_bands(self) -> dict[str, float]:
        """Retourne les niveaux de bandes normalisés dans [0, 1].

        Clés : ``'bass'``, ``'mid'``, ``'high'``, ``'vol'``
        """
        with self._lock:
            return dict(self._bands)

    def get_spectrum(self) -> np.ndarray:
        """Retourne le spectre de fréquences normalisé (SPECTRUM_BINS valeurs dans [0, 1])."""
        with self._lock:
            return self._spectrum.copy()

    # ------------------------------------------------------------------
    # Propriétés
    # ------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        """True si la lecture est active (pas en pause)."""
        return self._playing

    @property
    def is_paused(self) -> bool:
        """True si la lecture est suspendue par ``pause()``."""
        return self._paused

    @property
    def loaded(self) -> bool:
        """True si un fichier a été chargé."""
        return self._samples is not None

    @property
    def position(self) -> float:
        """Position de lecture en secondes."""
        return self._position / self._sample_rate

    @property
    def duration(self) -> float:
        """Durée totale du fichier en secondes."""
        return self._duration

    @property
    def remaining(self) -> float:
        """Temps restant en secondes."""
        return max(0.0, self._duration - self.position)

    @property
    def position_pct(self) -> float:
        """Position de lecture dans [0.0, 1.0]."""
        if self._duration == 0:
            return 0.0
        return self.position / self._duration

    @property
    def position_formatted(self) -> str:
        """Position de lecture formatée ``'M:SS / M:SS'``."""
        return f"{_fmt_time(self.position)} / {_fmt_time(self._duration)}"

    @property
    def metadata(self) -> dict[str, str]:
        """Métadonnées du fichier (filename, format, et tags ID3 si disponibles)."""
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Internals — stream
    # ------------------------------------------------------------------

    def _open_stream(self) -> None:
        self._close_stream()
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=CHUNK,
            callback=self._audio_callback,
            finished_callback=self._on_stream_finished,
        )
        self._stream.start()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ------------------------------------------------------------------
    # Internals — audio callback
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if not self._playing or self._samples is None:
            outdata[:] = 0
            return

        with self._lock:
            start = self._position
            end = start + frames

            if end >= len(self._samples):
                chunk = self._samples[start:]
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0] = 0.0

                if self._loop:
                    self._position = frames - len(chunk)
                else:
                    self._playing = False
                    self._position = 0
                    # on_finish sera déclenché dans _on_stream_finished
            else:
                chunk = self._samples[start:end]
                outdata[:, 0] = chunk
                self._position = end

            self._analyse(chunk)

    def _analyse(self, chunk: np.ndarray) -> None:
        """Calcule le spectre et les bandes depuis un bloc PCM (appelé sous _lock)."""
        n = len(chunk)
        if n == 0:
            return

        window = np.hanning(n)
        fft_data = np.abs(np.fft.rfft(chunk * window, n=FFT_SIZE))

        fft_db = 20 * np.log10(fft_data + 1e-9)
        fft_norm = np.clip((fft_db + 80) / 80, 0.0, 1.0).astype(np.float32)

        indices = np.linspace(0, len(fft_norm) - 1, SPECTRUM_BINS).astype(int)
        new_spectrum = fft_norm[indices]

        s = self.smoothing
        self._spectrum = s * self._spectrum + (1 - s) * new_spectrum

        freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / self._sample_rate)

        def band_energy(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.mean(fft_norm[mask])) if mask.any() else 0.0

        new_bands = {
            "bass":  band_energy(*BASS_RANGE),
            "mid":   band_energy(*MID_RANGE),
            "high":  band_energy(*HIGH_RANGE),
            "vol":   float(np.mean(np.abs(chunk))),
        }
        for key in self._bands:
            self._bands[key] = s * self._bands[key] + (1 - s) * new_bands[key]

        if self.on_beat is not None:
            try:
                self.on_beat(dict(self._bands), self._spectrum.copy())
            except Exception as exc:
                if self.on_error is not None:
                    self.on_error(exc)

    def _reset_analysis(self) -> None:
        with self._lock:
            self._spectrum[:] = 0.0
            for k in self._bands:
                self._bands[k] = 0.0

    def _on_stream_finished(self) -> None:
        was_playing = self._playing
        self._playing = False

        # Déclencher on_finish seulement si la fin est naturelle (pas stop/pause)
        if not self._paused and self.on_finish is not None:
            try:
                self.on_finish()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internals — métadonnées
    # ------------------------------------------------------------------

    def _load_tags(self, path: Path) -> None:
        """Tente d'extraire les tags ID3 via mutagen (non obligatoire)."""
        try:
            from mutagen import File as MutagenFile  # type: ignore

            tags = MutagenFile(str(path), easy=True)
            if tags is None:
                return
            for key, attr in (("title", "title"), ("artist", "artist"), ("album", "album")):
                values = tags.get(key)
                if values:
                    self._metadata[attr] = str(values[0])
        except ImportError:
            pass  # mutagen optionnel
        except Exception:
            pass  # tags corrompus ou format non supporté

    @staticmethod
    def _default_on_error(exc: Exception) -> None:
        warnings.warn(
            f"Exception dans on_beat (callback désactivé) : {exc}",
            stacklevel=1,
        )


# ---------------------------------------------------------------------------
# Démonstration en ligne de commande
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python mp3_reaction.py <fichier.mp3>")
        sys.exit(1)

    player = MP3Reaction()

    print(f"Chargement de '{sys.argv[1]}'…")
    try:
        player.load(sys.argv[1])
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    meta = player.metadata
    title = meta.get("title", meta.get("filename", "?"))
    artist = meta.get("artist", "")
    print(f"  {title}{' — ' + artist if artist else ''}")
    print(f"  Durée : {player.position_formatted.split('/')[1].strip()}")
    print("Lecture… (Ctrl+C pour arrêter, Entrée pour pause/reprise)")

    def afficher(bands: dict[str, float], _spectrum: np.ndarray) -> None:
        bar = lambda v: "█" * int(v * 20)  # noqa: E731
        print(
            f"\r  {player.position_formatted}  "
            f"bass [{bar(bands['bass']):<20}]  "
            f"mid [{bar(bands['mid']):<20}]  "
            f"high [{bar(bands['high']):<20}]",
            end="",
            flush=True,
        )

    def on_fin() -> None:
        print("\nLecture terminée.")

    player.on_beat = afficher
    player.on_finish = on_fin
    player.play()

    # Thread pour écouter Entrée (pause/reprise) sans bloquer
    def _input_loop() -> None:
        while True:
            try:
                input()
            except EOFError:
                break
            player.toggle_pause()
            state = "En pause" if player.is_paused else "Lecture"
            print(f"\n[{state}]", end="", flush=True)

    t = threading.Thread(target=_input_loop, daemon=True)
    t.start()

    try:
        while player.is_playing or player.is_paused:
            time.sleep(0.05)
    except KeyboardInterrupt:
        player.stop()
        print("\nArrêté.")
