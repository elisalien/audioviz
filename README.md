# AureactiveViz

Visualiseur audioreactif Windows — sortie Spout vers Resolume.

## Installation rapide

1. Installe **Python 3.11** depuis https://python.org  
   (cocher "Add to PATH" lors de l'installation)

2. Double-clic sur **install.bat**

3. Double-clic sur **run.bat**

---

## Deux fenêtres

| Fenêtre | Rôle |
|---|---|
| **AureactiveViz — Contrôles** | Panneau UI (à garder sur ton écran) |
| **AureactiveViz — Preview** | Rendu OpenGL (à mettre sur le projecteur ou monitor 2) |

---

## Entrée audio

| Mode | Quand l'utiliser |
|---|---|
| **Loopback** | Capturer ce qui sort de la carte son (son du set DJ, DAW, etc.) |
| **Input physique** | Micro, interface audio Focusrite/RME/etc. |

Pour le loopback : sélectionne un périphérique de **sortie** dans la liste  
(ils apparaissent avec le suffixe `[loopback]`).

> Sur certaines configs Windows, il faut activer **"Écoute stéréo"** (Stereo Mix)  
> dans Paramètres son → Enregistrement → clic droit → Afficher les périphériques désactivés.

---

## Shaders

| Preset | Style |
|---|---|
| **Spectrum** | Barres de fréquences miroir avec glow — style Winamp classique |
| **Waveform** | Oscilloscope multi-couches — style CRT |
| **Tunnel** | Tunnel psychédélique — style MilkDrop |
| **Plasma** | Interférences d'ondes — style démo |
| **Particles** | Champ de particules en orbite — style VJ |

Paramètres communs :
- **Vitesse** — cadence temporelle du shader
- **Complexité** — densité / amplitude du motif
- **Color shift** — décalage de teinte global

---

## Sprites PNG / GIF

- Clic **Importer** → sélectionne ton fichier
- **Scale** — taille en proportion de l'image (réactive à l'audio)
- **React** — quelle bande fréquentielle pilote la réactivité (bass / mid / high / vol)
- Checkbox gauche = masquer/afficher
- ✕ = supprimer

Les GIF animés jouent automatiquement à leur vitesse native.

---

## Sortie Spout → Resolume

1. Cocher **Spout** dans le panneau
2. Note le **nom** du sender (ex. `AureactiveViz`)
3. Dans Resolume : `Sources → Spout → AureactiveViz`
4. Change la **Résolution** pour matcher ta sortie Resolume

---

## Raccourcis fenêtre Preview

| Touche | Action |
|---|---|
| `F` | Plein écran |
| `Échap` | Sortir du plein écran |

---

## Dépendances

```
moderngl        — rendu OpenGL
glfw            — fenêtre OpenGL
numpy           — FFT
sounddevice     — capture audio (WASAPI loopback)
Pillow          — PNG / GIF
dearpygui       — UI
SpoutGL         — sortie Spout (optionnel)
```
"# audioviz" 
