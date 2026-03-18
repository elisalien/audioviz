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
SAMPLE_RATE = 44100          # Hz
CHUNK = 1024                 # Taille d'une fenêtre d'analyse (samples)
FFT_SIZE = 2048              # Zéro-padding pour une meilleure résolution
SPECTRUM_BINS = 512          # Taille du spectre renvoyé à l'appelant
SMOOTHING = 0.75             # Lissage exponentiel (0 = aucun, <1 = rapide, ~0.9 = lent)

# Plages de fréquences (Hz) pour les 3 bandes
BASS_RANGE  = (20,   300)
MID_RANGE   = (300,  4000)
HIGH_RANGE  = (4000, 20000)


class MP3Reaction:
    """Lecteur MP3 avec extraction de bandes fréquentielles en temps réel."""

    # ------------------------------------------------------------------
    # Construction / chargement
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._samples: Optional[np.ndarray] = None  # float32, shape (N,) mono
        self._sample_rate: int = SAMPLE_RATE
        self._duration: float = 0.0

        self._position: int = 0       # index courant dans _samples
        self._playing: bool = False
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None

        # Résultats d'analyse (mis à jour par le thread audio)
        self._spectrum = np.zeros(SPECTRUM_BINS, dtype=np.float32)
        self._bands: dict[str, float] = {"bass": 0.0, "mid": 0.0, "high": 0.0, "vol": 0.0}

        # Callback optionnel appelé à chaque chunk : cb(bands, spectrum)
        self.on_beat: Optional[Callable[[dict[str, float], np.ndarray], None]] = None

    def load(self, path: str | Path) -> None:
        """Charge un fichier MP3 (ou WAV/FLAC/OGG si pydub est dispo).

        Paramètres
        ----------
        path : chemin vers le fichier audio
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        if not _PYDUB_AVAILABLE:
            raise ImportError(
                "pydub est requis pour charger des MP3.\n"
                "  pip install pydub\n"
                "  + ffmpeg dans le PATH"
            )

        audio: AudioSegment = AudioSegment.from_file(str(path))
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)

        raw = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
        self._samples = raw / 32768.0          # normalisation → [-1, 1]
        self._sample_rate = SAMPLE_RATE
        self._duration = len(self._samples) / SAMPLE_RATE
        self._position = 0

    # ------------------------------------------------------------------
    # Contrôle de lecture
    # ------------------------------------------------------------------

    def play(self, loop: bool = False) -> None:
        """Démarre la lecture (non-bloquant).

        Paramètres
        ----------
        loop : si True, remet la lecture au début à la fin du fichier
        """
        if self._samples is None:
            raise RuntimeError("Aucun fichier chargé — appelez load() d'abord.")
        if self._playing:
            return

        self._playing = True
        self._loop = loop

        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=CHUNK,
            callback=self._audio_callback,
            finished_callback=self._on_stream_finished,
        )
        self._stream.start()

    def stop(self) -> None:
        """Arrête la lecture et remet la tête de lecture au début."""
        self._playing = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._position = 0
        self._reset_analysis()

    def pause(self) -> None:
        """Suspend la lecture sans changer la position."""
        self._playing = False
        if self._stream is not None:
            self._stream.stop()

    def resume(self) -> None:
        """Reprend après un pause()."""
        if self._samples is None or self._playing:
            return
        self._playing = True
        if self._stream is not None:
            self._stream.start()

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
        return self._playing

    @property
    def position(self) -> float:
        """Position de lecture en secondes."""
        return self._position / self._sample_rate

    @property
    def duration(self) -> float:
        """Durée totale du fichier en secondes."""
        return self._duration

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,   # noqa: ANN001
        status,      # noqa: ANN001
    ) -> None:
        """Callback appelé par sounddevice pour chaque bloc audio."""
        if not self._playing or self._samples is None:
            outdata[:] = 0
            return

        with self._lock:
            start = self._position
            end = start + frames

            if end >= len(self._samples):
                # Fin du fichier
                chunk = self._samples[start:]
                pad = frames - len(chunk)
                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0] = 0.0

                if getattr(self, "_loop", False):
                    self._position = pad
                else:
                    self._playing = False
                    self._position = 0
            else:
                chunk = self._samples[start:end]
                outdata[:, 0] = chunk
                self._position = end

            # Analyse FFT sur ce chunk
            self._analyse(chunk)

    def _analyse(self, chunk: np.ndarray) -> None:
        """Calcule le spectre et les bandes depuis un bloc PCM (appelé sous _lock)."""
        n = len(chunk)
        if n == 0:
            return

        # Fenêtre de Hann + FFT
        window = np.hanning(n)
        fft_data = np.abs(np.fft.rfft(chunk * window, n=FFT_SIZE))

        # Normalisation logarithmique
        fft_db = 20 * np.log10(fft_data + 1e-9)
        fft_norm = np.clip((fft_db + 80) / 80, 0.0, 1.0).astype(np.float32)

        # Rééchantillonnage vers SPECTRUM_BINS
        indices = np.linspace(0, len(fft_norm) - 1, SPECTRUM_BINS).astype(int)
        new_spectrum = fft_norm[indices]

        # Lissage exponentiel
        self._spectrum = SMOOTHING * self._spectrum + (1 - SMOOTHING) * new_spectrum

        # Extraction des bandes
        freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / self._sample_rate)

        def band_energy(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                return 0.0
            return float(np.mean(fft_norm[mask]))

        new_bands = {
            "bass":  band_energy(*BASS_RANGE),
            "mid":   band_energy(*MID_RANGE),
            "high":  band_energy(*HIGH_RANGE),
            "vol":   float(np.mean(np.abs(chunk))),
        }
        # Lissage des bandes
        for key in self._bands:
            self._bands[key] = SMOOTHING * self._bands[key] + (1 - SMOOTHING) * new_bands[key]

        # Callback utilisateur (hors lock pour éviter deadlock)
        if self.on_beat is not None:
            try:
                self.on_beat(dict(self._bands), self._spectrum.copy())
            except Exception:  # noqa: BLE001
                pass

    def _reset_analysis(self) -> None:
        with self._lock:
            self._spectrum[:] = 0.0
            for k in self._bands:
                self._bands[k] = 0.0

    def _on_stream_finished(self) -> None:
        self._playing = False


# ---------------------------------------------------------------------------
# Démonstration en ligne de commande
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python mp3_reaction.py <fichier.mp3>")
        sys.exit(1)

    player = MP3Reaction()
    player.load(sys.argv[1])
    print(f"Durée : {player.duration:.1f}s — lecture en cours…  (Ctrl+C pour arrêter)")

    def afficher(bands: dict[str, float], _spectrum: np.ndarray) -> None:  # noqa: ANN001
        bar = lambda v: "█" * int(v * 20)  # noqa: E731
        print(
            f"\r  bass [{bar(bands['bass']):<20}]  "
            f"mid [{bar(bands['mid']):<20}]  "
            f"high [{bar(bands['high']):<20}]  "
            f"vol [{bar(bands['vol']):<20}]",
            end="",
            flush=True,
        )

    player.on_beat = afficher
    player.play()

    try:
        while player.is_playing:
            time.sleep(0.05)
    except KeyboardInterrupt:
        player.stop()
        print("\nArrêté.")
